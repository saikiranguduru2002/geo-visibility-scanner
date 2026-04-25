"""
scanner.py

A lightweight GEO (Generative Engine Optimization) visibility scanner for a brand.

This module intentionally avoids paid third-party APIs (except Google's Knowledge Graph
Search API, which has a free tier). All network operations are best-effort and include
retries/backoff for resiliency.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup


logger = logging.getLogger(__name__)


DEFAULT_TIMEOUT_S = 12
DEFAULT_UA = (
    "Mozilla/5.0 (compatible; GEOScanner/1.0; +https://example.com/bot) "
    "requests"
)

KG_POINTS_FULL = 25
KG_POINTS_ESTIMATED = 12


class ScannerError(Exception):
    """Base exception for scanner-level errors (rarely raised; mostly logged)."""


@dataclass(frozen=True)
class HttpResult:
    url: str
    status_code: Optional[int]
    elapsed_s: Optional[float]
    text: Optional[str]
    headers: Dict[str, str]
    error: Optional[str]


def _request_with_retries(
    method: str,
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    max_attempts: int = 3,
    backoff_s: float = 1.2,
    allow_redirects: bool = True,
) -> HttpResult:
    """
    Make an HTTP request with small retry/backoff and graceful error capture.

    Retries on:
    - transient network errors
    - HTTP 429 / 5xx

    Returns an HttpResult that always exists; never raises requests exceptions.
    """
    hdrs = {"User-Agent": DEFAULT_UA}
    if headers:
        hdrs.update(headers)

    last_error: Optional[str] = None
    for attempt in range(1, max_attempts + 1):
        t0 = time.perf_counter()
        try:
            resp = requests.request(
                method=method.upper(),
                url=url,
                headers=hdrs,
                params=params,
                timeout=timeout_s,
                allow_redirects=allow_redirects,
            )
            elapsed = time.perf_counter() - t0

            # Rate-limit / transient server errors: backoff and retry.
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < max_attempts:
                retry_after = resp.headers.get("Retry-After")
                sleep_s = float(retry_after) if retry_after and retry_after.isdigit() else backoff_s * attempt
                logger.warning(
                    "HTTP %s %s -> %s (attempt %s/%s). Backing off %.2fs",
                    method.upper(),
                    url,
                    resp.status_code,
                    attempt,
                    max_attempts,
                    sleep_s,
                )
                time.sleep(sleep_s)
                continue

            return HttpResult(
                url=str(resp.url),
                status_code=resp.status_code,
                elapsed_s=elapsed,
                text=resp.text,
                headers={k: v for k, v in resp.headers.items()},
                error=None,
            )
        except requests.RequestException as e:
            elapsed = time.perf_counter() - t0
            last_error = f"{type(e).__name__}: {e}"
            logger.warning(
                "HTTP %s %s failed (attempt %s/%s): %s",
                method.upper(),
                url,
                attempt,
                max_attempts,
                last_error,
            )
            if attempt < max_attempts:
                time.sleep(backoff_s * attempt)
                continue
            return HttpResult(
                url=url,
                status_code=None,
                elapsed_s=elapsed,
                text=None,
                headers={},
                error=last_error,
            )

    # Should be unreachable
    return HttpResult(url=url, status_code=None, elapsed_s=None, text=None, headers={}, error=last_error)


def _normalize_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return url
    parsed = urlparse(url)
    if not parsed.scheme:
        return "https://" + url.lstrip("/")
    return url


def _is_https(url: str) -> bool:
    try:
        return urlparse(url).scheme.lower() == "https"
    except Exception:
        return False


def _extract_json_ld_blocks(html: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Extract JSON-LD blocks from HTML. Returns (json_objects, errors).

    - Handles both JSON objects and JSON arrays within script tags.
    - Best-effort: parsing failures go to errors list, extraction continues.
    """
    errors: List[str] = []
    objects: List[Dict[str, Any]] = []

    soup = BeautifulSoup(html or "", "html.parser")
    scripts = soup.find_all("script", attrs={"type": re.compile(r"^application/ld\+json$", re.I)})

    for idx, tag in enumerate(scripts):
        raw = tag.string if tag.string is not None else tag.get_text()
        if not raw or not raw.strip():
            continue
        raw = raw.strip()
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                objects.append(parsed)
            elif isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        objects.append(item)
            else:
                errors.append(f"JSON-LD block {idx} is neither object nor array")
        except json.JSONDecodeError as e:
            # Fallback: try to find the first {...} region and parse it (common when multiple JSON objects are concatenated).
            errors.append(f"JSON-LD block {idx} JSON decode error: {e}")
            candidate = _try_regex_json_object(raw)
            if candidate is not None:
                objects.append(candidate)

    return objects, errors


def _expand_json_ld_nodes(objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Flatten @graph and pull nested organization-like nodes (publisher, provider, brand).
    """
    expanded: List[Dict[str, Any]] = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        expanded.append(obj)
        graph = obj.get("@graph")
        if isinstance(graph, list):
            for node in graph:
                if isinstance(node, dict):
                    expanded.append(node)
        for key in ("publisher", "provider", "brand", "manufacturer", "isPartOf"):
            child = obj.get(key)
            if isinstance(child, dict):
                expanded.append(child)
            elif isinstance(child, list):
                for c in child:
                    if isinstance(c, dict):
                        expanded.append(c)
    return expanded


def _try_regex_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Very small heuristic to salvage a JSON object from a broken JSON-LD block.
    This is intentionally conservative to avoid returning garbage.
    """
    if not text:
        return None
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    snippet = m.group(0).strip()
    try:
        parsed = json.loads(snippet)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _looks_like_org_schema(obj: Dict[str, Any]) -> bool:
    """
    Detect Organization schema candidates.
    Accepts:
    - {"@type":"Organization"} or array types
    - "Corporation" and other org-like types as Organization variants
    """
    typ = obj.get("@type")
    if isinstance(typ, list):
        types = {str(t).lower() for t in typ}
    elif typ is None:
        types = set()
    else:
        types = {str(typ).lower()}

    org_types = {
        "organization",
        "corporation",
        "localbusiness",
        "newsmediaorganization",
        "educationalorganization",
        "governmentorganization",
        "sportsorganization",
        "airline",
        "consortium",
        "ngo",
        "brand",
    }
    return bool(types & org_types)


def _get_title_and_meta_description(html: str) -> Tuple[str, str]:
    soup = BeautifulSoup(html or "", "html.parser")
    title = ""
    if soup.title is not None:
        title = (soup.title.get_text(strip=True) or "").strip()
    meta_desc = ""
    md = soup.find("meta", attrs={"name": re.compile(r"^description$", re.I)})
    if md and md.get("content") is not None:
        meta_desc = str(md.get("content") or "").strip()
    return title, meta_desc
def _estimate_mention_signals(brand_name: str, html: str) -> Dict[str, Any]:
    """
    Cheap mention signal estimator (no paid APIs).
    Returns counts of brand occurrences and a qualitative label.
    """
    brand = (brand_name or "").strip()
    if not brand or not html:
        return {"brand": brand, "occurrences": 0, "signal": "unknown"}

    # Normalize whitespace; do case-insensitive substring counts.
    haystack = re.sub(r"\s+", " ", html).lower()
    needle = brand.lower()
    occurrences = haystack.count(needle)

    if occurrences >= 20:
        signal = "strong"
    elif occurrences >= 5:
        signal = "medium"
    elif occurrences >= 1:
        signal = "weak"
    else:
        signal = "none"

    return {"brand": brand, "occurrences": occurrences, "signal": signal}

def check_knowledge_graph(brand_name: str) -> Dict[str, Any]:
    """
    Check if a brand appears in Google's Knowledge Graph via the KG Search API.

    Environment variable:
    - GOOGLE_KG_API_KEY: API key for the Knowledge Graph Search API

    Returns a dict: found, confidence, description, rate_limited, kg_score_points (int, for scoring)
    """
    out: Dict[str, Any] = {
        "found": False,
        "confidence": 0.0,
        "description": "",
        "rate_limited": False,
        "kg_score_points": 0,
    }
    
    # Validate brand name
    brand = (brand_name or "").strip()
    if not brand:
        out["description"] = "brand_name is empty"
        return out

    # Load API key
    import os
    api_key = (os.getenv("GOOGLE_KG_API_KEY") or "").strip()
    if not api_key:
        out["description"] = "Missing GOOGLE_KG_API_KEY env var"
        return out

    endpoint = "https://kgsearch.googleapis.com/v1/entities:search"
    params = {
        "query": brand,
        "key": api_key,
        "limit": 5,
        "indent": "True",
        "languages": "en",
    }

    res = _request_with_retries("GET", endpoint, params=params, timeout_s=DEFAULT_TIMEOUT_S, max_attempts=4)
    if res.error:
        out["description"] = f"Knowledge Graph request error: {res.error}"
        return out
    if res.status_code is None:
        out["description"] = "Knowledge Graph request failed"
        return out

    if res.status_code == 403:
        out["description"] = "Knowledge Graph API 403 (check API key, enabled API, or quota)"
        return out
    if res.status_code == 429:
        out["rate_limited"] = True
        out["confidence"] = 0.35
        out["kg_score_points"] = KG_POINTS_ESTIMATED
        out["description"] = (
            "Knowledge Graph rate-limited (HTTP 429) after retries; using estimated KG contribution in score"
        )
        return out
    if res.status_code >= 400:
        out["description"] = f"Knowledge Graph HTTP {res.status_code}"
        return out

    try:
        data = json.loads(res.text or "{}")
    except json.JSONDecodeError as e:
        out["description"] = f"Knowledge Graph JSON parse error: {e}"
        return out

    if not isinstance(data, dict):
        out["description"] = "Knowledge Graph response was not a JSON object"
        return out

    items = data.get("itemListElement") or []
    if not isinstance(items, list) or not items:
        out["description"] = "No Knowledge Graph results"
        return out

    top: Dict[str, Any] = items[0] if isinstance(items[0], dict) else {}
    if not top:
        out["description"] = "Knowledge Graph first result was empty"
        return out

    score_raw = 0.0
    try:
        score_raw = float(top.get("resultScore") or 0.0)
    except (TypeError, ValueError):
        score_raw = 0.0

    confidence = max(0.0, min(1.0, score_raw / 1000.0))

    result: Dict[str, Any] = {}
    r = top.get("result")
    if isinstance(r, dict):
        result = r
    desc = ""
    detailed = result.get("detailedDescription")
    if isinstance(detailed, dict) and detailed.get("articleBody") is not None:
        desc = str(detailed.get("articleBody") or "").strip()
    elif result.get("description") is not None:
        desc = str(result.get("description") or "").strip()

    name = (result.get("name") or "").strip() if result.get("name") is not None else ""
    found = bool(name) and (confidence >= 0.15 or score_raw >= 150)
    if not desc:
        desc = name or "Knowledge Graph result found"
    if found:
        out["kg_score_points"] = KG_POINTS_FULL
    out["found"] = found
    out["confidence"] = confidence
    out["description"] = desc
    return out

def check_schema_markup(url: str) -> Tuple[bool, Dict[str, Any], List[str]]:
    """
    Fetch homepage HTML and look for JSON-LD Organization schema.

    Returns:
    - has_schema (bool)
    - schema_data (dict): the best Organization-like JSON-LD object found (or {})
    - errors (list): parse/network warnings
    """
    homepage = _normalize_url(url)
    errors: List[str] = []
    if not homepage:
        return False, {}, ["url is empty"]

    res = _request_with_retries("GET", homepage, timeout_s=DEFAULT_TIMEOUT_S, max_attempts=3)
    if res.error:
        return False, {}, [f"Homepage fetch error: {res.error}"]
    if res.status_code is None or res.status_code >= 400:
        return False, {}, [f"Homepage fetch failed (HTTP {res.status_code})"]

    jsonld_objects, jsonld_errors = _extract_json_ld_blocks(res.text or "")
    errors.extend(jsonld_errors)

    flat_nodes = _expand_json_ld_nodes([o for o in jsonld_objects if isinstance(o, dict)])
    org_candidates = [obj for obj in flat_nodes if isinstance(obj, dict) and _looks_like_org_schema(obj)]
    if not org_candidates:
        return False, {}, errors

    # Prefer objects with "name" and "url"
    def rank(o: Dict[str, Any]) -> int:
        score = 0
        if o.get("name"):
            score += 2
        if o.get("url"):
            score += 2
        if o.get("sameAs"):
            score += 1
        if o.get("logo"):
            score += 1
        return score

    best = sorted(org_candidates, key=rank, reverse=True)[0]
    if not isinstance(best, dict):
        return False, {}, errors + ["Organization schema candidate was not a JSON object"]
    return True, best, errors


def check_wikidata(brand_name: str) -> Tuple[bool, str, str]:
    """
    Query Wikidata via SPARQL for an item whose label matches the brand name.

    Returns:
    - found (bool)
    - qid (str): like 'Q95' or ''
    - description (str): English description if available, else empty / error text
    """
    brand = (brand_name or "").strip()
    if not brand:
        return False, "", "brand_name is empty"

    endpoint = "https://query.wikidata.org/sparql"

    # Prefer exact label match; fallback is inherently fuzzy and can be added later.
    sparql = f"""
    SELECT ?item ?itemLabel ?itemDescription WHERE {{
      ?item rdfs:label "{brand}"@en .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT 5
    """.strip()

    headers = {
        "Accept": "application/sparql+json",
        "User-Agent": DEFAULT_UA,
    }

    res = _request_with_retries(
        "GET",
        endpoint,
        headers=headers,
        params={"query": sparql, "format": "json"},
        timeout_s=DEFAULT_TIMEOUT_S,
        max_attempts=4,
        backoff_s=1.5,
    )

    if res.error:
        return False, "", f"Wikidata request error: {res.error}"
    if res.status_code == 429:
        return False, "", "Wikidata rate-limited (HTTP 429)"
    if res.status_code is None or res.status_code >= 400:
        return False, "", f"Wikidata HTTP {res.status_code}"

    try:
        data = json.loads(res.text or "{}")
        bindings = (((data.get("results") or {}).get("bindings")) or [])
    except Exception as e:
        return False, "", f"Wikidata JSON parse error: {e}"

    if not isinstance(bindings, list) or not bindings:
        return False, "", "No matching Wikidata item"

    first = bindings[0] if isinstance(bindings[0], dict) else {}
    item_node = first.get("item")
    val = (item_node or {}).get("value") if isinstance(item_node, dict) else None
    item_uri = (str(val).strip() if val is not None else "")
    qid = item_uri.rsplit("/", 1)[-1] if item_uri else ""
    idesc = first.get("itemDescription")
    ilab = first.get("itemLabel")
    desc = ""
    if isinstance(idesc, dict) and idesc.get("value") is not None:
        desc = str(idesc.get("value") or "").strip()
    if not desc and isinstance(ilab, dict) and ilab.get("value") is not None:
        desc = str(ilab.get("value") or "").strip()

    return bool(qid), qid, desc


def check_crawlability(url: str) -> Tuple[bool, List[str]]:
    """
    Evaluate basic crawlability:
    - Fetch robots.txt and verify homepage is allowed
    - Check for sitemap.xml presence (and robots-declared sitemap if present)

    Returns:
    - crawlable (bool)
    - robots_issues (list)
    """
    homepage = _normalize_url(url)
    issues: List[str] = []
    if not homepage:
        return False, ["url is empty"]

    parsed = urlparse(homepage)
    base = f"{parsed.scheme}://{parsed.netloc}"
    robots_url = urljoin(base + "/", "robots.txt")

    # robots.txt fetch (best-effort). Some sites return 403 to non-browser or generic bots.
    robots_res = _request_with_retries("GET", robots_url, timeout_s=DEFAULT_TIMEOUT_S, max_attempts=3)
    robots_txt = robots_res.text or ""
    if robots_res.error:
        issues.append(f"robots.txt fetch error: {robots_res.error}")
    elif robots_res.status_code and robots_res.status_code >= 400:
        issues.append(f"robots.txt not accessible (HTTP {robots_res.status_code})")
        if robots_res.status_code == 403:
            # Retry with a search crawler UA; many CDNs only allow this for robots.
            alt_ua = {
                "User-Agent": "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"
            }
            alt = _request_with_retries(
                "GET",
                robots_url,
                headers=alt_ua,
                timeout_s=DEFAULT_TIMEOUT_S,
                max_attempts=2,
            )
            if not alt.error and alt.text and (alt.status_code or 500) < 400:
                robots_txt = alt.text or ""
                issues.append("robots.txt: initial fetch was 403; loaded with search-bot user-agent for rules")
            else:
                issues.append("robots.txt: could not read rules; treating crawl as unknown (permissive)")
                robots_txt = ""

    rp = RobotFileParser()
    if robots_txt.strip():
        try:
            rp.set_url(robots_url)
            rp.parse(robots_txt.splitlines())
        except Exception as e:
            issues.append(f"robots.txt parse error: {e}")

    # Check whether homepage is allowed (generic UA and Googlebot — many sites differ).
    try:
        allowed_user = rp.can_fetch("*", homepage) if robots_txt.strip() else True
        allowed_bot = rp.can_fetch("Googlebot", homepage) if robots_txt.strip() else True
    except Exception:
        allowed_user = True
        allowed_bot = True
        issues.append("robots.txt allow-check failed; assuming allowed")

    allowed = allowed_user or allowed_bot
    if robots_txt.strip() and not allowed_user and allowed_bot:
        issues.append("Generic user-agent may be disallowed, but Googlebot is allowed (typical for indexing)")

    if not allowed:
        issues.append("Homepage appears disallowed by robots.txt for both * and Googlebot")

    # Sitemap checks
    sitemap_url = urljoin(base + "/", "sitemap.xml")
    sitemap_res = _request_with_retries("GET", sitemap_url, timeout_s=DEFAULT_TIMEOUT_S, max_attempts=2)
    sitemap_ok = bool(sitemap_res.status_code and sitemap_res.status_code < 400 and (sitemap_res.text or "").strip())
    if not sitemap_ok:
        issues.append("sitemap.xml not found or inaccessible")

    # Also check for Sitemap: lines in robots.txt
    if robots_txt:
        declared = re.findall(r"(?im)^\s*sitemap:\s*(\S+)\s*$", robots_txt)
        if declared:
            # If declared, validate at least one works.
            any_ok = False
            for s in declared[:5]:
                r = _request_with_retries("GET", s, timeout_s=DEFAULT_TIMEOUT_S, max_attempts=2)
                if r.status_code and r.status_code < 400:
                    any_ok = True
                    break
            if not any_ok:
                issues.append("Robots-declared sitemap URL(s) appear inaccessible")

    crawlable = allowed
    return crawlable, issues


def calculate_geo_score(results: Dict[str, Any]) -> Tuple[int, str, List[str]]:
    """
    Calculate an overall GEO score and map failures to the 6-point exclusion framework.

    Expected (best-effort) inputs inside results dict:
    - knowledge_graph: {found: bool, confidence: float, description: str}
    - schema: {has_schema: bool, schema_data: dict, errors: list}
    - wikidata: {found: bool, qid: str, description: str}
    - crawlability: {crawlable: bool, robots_issues: list, sitemap_present: bool (optional)}
    - homepage: {url: str, https: bool, load_time_s: float, title: str, meta_description: str, mention_signals: dict}

    Returns:
    - score (0-100)
    - grade (A-F)
    - exclusion_points (list of strings like "#1 Not Crawled")
    """
    score = 0
    exclusion: List[str] = []

    kg = results.get("knowledge_graph") or {}
    schema = results.get("schema") or {}
    wd = results.get("wikidata") or {}
    crawl = results.get("crawlability") or {}
    home = results.get("homepage") or {}

    # 1) Knowledge Graph (up to +25; rate-limited uses partial points from check_knowledge_graph)
    try:
        kg_pts = int(kg.get("kg_score_points") or 0)
    except (TypeError, ValueError):
        kg_pts = 0
    if bool(kg.get("found")) and kg_pts <= 0:
        kg_pts = KG_POINTS_FULL
    score += min(KG_POINTS_FULL, max(0, kg_pts))
    if not bool(kg.get("found")) and not bool(kg.get("rate_limited")):
        exclusion.append("#3 Not Tokenized")
    # If rate_limited: partial kg points, no #3 for KG alone (Wikidata/schema may still add #3).

    # 2) Schema Markup (+20)
    if bool(schema.get("has_schema")):
        score += 20
    else:
        # schema absence contributes to tokenization too
        if "#3 Not Tokenized" not in exclusion:
            exclusion.append("#3 Not Tokenized")

    # 3) Wikidata (+20)
    if bool(wd.get("found")):
        score += 20
    else:
        if "#3 Not Tokenized" not in exclusion:
            exclusion.append("#3 Not Tokenized")

    # 4) Crawlable + Sitemap (+15)
    crawlable = bool(crawl.get("crawlable"))
    robots_issues = crawl.get("robots_issues") or []
    sitemap_present = "sitemap.xml not found or inaccessible" not in robots_issues
    if crawlable and sitemap_present:
        score += 15
    else:
        exclusion.append("#1 Not Crawled")
        if not crawlable:
            exclusion.append("#5 Not in Index")

    # 5) HTTPS + Fast load (+10)
    https_ok = bool(home.get("https"))
    load_time_s = home.get("load_time_s")
    fast = False
    try:
        fast = (load_time_s is not None) and float(load_time_s) <= 2.5
    except Exception:
        fast = False
    if https_ok:
        score += 5
    if fast:
        score += 5

    # 6) Content signals (+10): meta description present, title has brand
    title = (home.get("title") or "").strip()
    meta_desc = (home.get("meta_description") or "").strip()
    brand = (results.get("brand_name") or "").strip()
    title_has_brand = bool(brand) and (brand.lower() in title.lower())
    if meta_desc:
        score += 5
    if title_has_brand:
        score += 5
    else:
        # semantic mismatch proxy
        exclusion.append("#6 Semantic Mismatch")

    # Extra framework estimation signals (no extra points, just exclusions)
    mentions = (home.get("mention_signals") or {})
    signal = (mentions.get("signal") or "").strip().lower()
    if signal in ("none", "weak", "unknown"):
        exclusion.append("#4 Not Learned")

    # Thin content signals (very rough): missing title or extremely short meta
    if not title or (meta_desc and len(meta_desc) < 50) or not meta_desc:
        exclusion.append("#2 Content Quality")

    # Deduplicate while keeping order
    seen = set()
    exclusion_points: List[str] = []
    for x in exclusion:
        if x not in seen:
            exclusion_points.append(x)
            seen.add(x)

    score = max(0, min(100, int(round(score))))
    grade = _grade_from_score(score)
    return score, grade, exclusion_points


def _grade_from_score(score: int) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    if score >= 50:
        return "E"
    return "F"


def generate_playbook(exclusion_points: List[str]) -> List[Dict[str, str]]:
    """
    Convert exclusion points into an actionable fix list with priority and timeline.

    Uses the user's fix templates mapped to the 6-point framework.
    Output items:
    - point
    - priority: P0/P1/P2
    - timeline: e.g., "Today", "1-2 weeks", "2-6 weeks"
    - actions: concise checklist string
    """
    points = exclusion_points or []
    playbook: List[Dict[str, str]] = []

    def add(point: str, priority: str, timeline: str, actions: str) -> None:
        playbook.append(
            {
                "point": point,
                "priority": priority,
                "timeline": timeline,
                "actions": actions,
            }
        )

    for p in points:
        if p.startswith("#1"):
            add(
                p,
                "P0",
                "Today",
                "Run a robots.txt audit; ensure homepage and key paths are allowed. "
                "Add/validate sitemap(s) and submit in Google Search Console.",
            )
        elif p.startswith("#2"):
            add(
                p,
                "P1",
                "1-2 weeks",
                "Do a content quality audit: strengthen homepage and key landing pages, "
                "expand thin sections, add clear value props, FAQs, and entity-rich copy.",
            )
        elif p.startswith("#3"):
            add(
                p,
                "P0",
                "1-2 weeks",
                "Create/upgrade Wikidata entity (with references). Add Organization JSON-LD "
                "with name, url, logo, sameAs, and identifier fields.",
            )
        elif p.startswith("#4"):
            add(
                p,
                "P2",
                "2-6 weeks",
                "Build comparison + 'alternatives' content, publish PR/earned media, "
                "and grow community presence where LLMs ingest signals.",
            )
        elif p.startswith("#5"):
            add(
                p,
                "P0",
                "Today",
                "Fix indexing blockers: remove disallow rules, ensure canonical tags are correct, "
                "and use structured data to improve discoverability.",
            )
        elif p.startswith("#6"):
            add(
                p,
                "P1",
                "1-2 weeks",
                "Align semantics: make titles/meta reflect the category + brand intent. "
                "Create intent-specific pages for top queries and ensure consistent entity language.",
            )
        else:
            add(p, "P2", "2-6 weeks", "Investigate and address the underlying exclusion driver.")

    # Stable sort by priority then point
    prio_order = {"P0": 0, "P1": 1, "P2": 2}
    playbook.sort(key=lambda x: (prio_order.get(x["priority"], 9), x["point"]))
    return playbook


def scan_brand_geo(    brand_name: str, url: str) -> Dict[str, Any]:
    """
    Convenience orchestrator (optional): runs all checks and returns a consolidated result dict.

    This is not required by the prompt, but makes the module easier to consume.
    """
    homepage = _normalize_url(url)
    knowledge_graph = check_knowledge_graph(brand_name)
    schema_has, schema_data, schema_errors = check_schema_markup(homepage)
    wd_found, wd_qid, wd_desc = check_wikidata(brand_name)
    crawlable, robots_issues = check_crawlability(homepage)

    # Homepage signals for scoring (HTTPS, load time, title/meta, mention estimate)
    home_res = _request_with_retries("GET", homepage, timeout_s=DEFAULT_TIMEOUT_S, max_attempts=2)
    title, meta_desc = _get_title_and_meta_description(home_res.text or "")
    mention_signals = _estimate_mention_signals(brand_name, home_res.text or "")

    consolidated: Dict[str, Any] = {
        "brand_name": brand_name,
        "url": homepage,
        "knowledge_graph": knowledge_graph,
        "schema": {"has_schema": schema_has, "schema_data": schema_data, "errors": schema_errors},
        "wikidata": {"found": wd_found, "qid": wd_qid, "description": wd_desc},
        "crawlability": {
            "crawlable": crawlable,
            "robots_issues": robots_issues,
        },
        "homepage": {
            "url": home_res.url or homepage,
            "https": _is_https(home_res.url or homepage),
            "load_time_s": home_res.elapsed_s,
            "title": title,
            "meta_description": meta_desc,
            "mention_signals": mention_signals,
        },
    }

    score, grade, exclusion_points = calculate_geo_score(consolidated)
    consolidated["geo_score"] = {"score": score, "grade": grade, "exclusion_points": exclusion_points}
    consolidated["playbook"] = generate_playbook(exclusion_points)
    return consolidated

