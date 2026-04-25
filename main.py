"""
main.py

FastAPI backend for the GEO scanner.

Endpoints:
- POST /scan
- GET /health
- GET /demo-brands

Features:
- CORS for frontend access
- Simple per-IP rate limiting (5 req/min)
- In-memory cache (1 hour)
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from urllib.parse import urlparse

try:
    from dotenv import load_dotenv
    from pathlib import Path
    import os

    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(env_path)

    # DEBUG: Check if key loaded
    print(f"DEBUG: .env path = {env_path}")
    print(f"DEBUG: .env exists = {env_path.exists()}")
    print(f"DEBUG: GOOGLE_KG_API_KEY = {os.getenv('GOOGLE_KG_API_KEY')[:15] if os.getenv('GOOGLE_KG_API_KEY') else 'NOT FOUND'}")

except ImportError:
    pass
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, HttpUrl, ValidationError

import scanner


APP_VERSION = "1.0.0"

logger = logging.getLogger("geo-scanner-api")

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _utc_now_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_url_for_key(url: str) -> str:
    """
    Normalize URL for caching keys (scheme + netloc + path without trailing slash).
    scanner.scan_brand_geo also normalizes for fetch, but caching should be stable.
    """
    u = (url or "").strip()
    if not u:
        return u
    if "://" not in u:
        u = "https://" + u.lstrip("/")
    p = urlparse(u)
    path = p.path or "/"
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    return f"{p.scheme.lower()}://{p.netloc.lower()}{path}"


def _client_ip(request: Request) -> str:
    # Prefer X-Forwarded-For if present (common behind proxies).
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class ScanRequest(BaseModel):
    brand_name: str = Field(..., min_length=1, max_length=120)
    url: HttpUrl


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str


class PlaybookItem(BaseModel):
    exclusion: int
    title: str
    priority: str
    actions: List[str]
    timeline: str
    expected_impact: str


class ScanResponse(BaseModel):
    brand: str
    url: str
    score: int
    grade: str
    exclusion_points: List[int]
    checks: Dict[str, Any]
    playbook: List[PlaybookItem]
    scanned_at: str


# -----------------------------
# Rate limiting + caching
# -----------------------------

RATE_LIMIT_PER_MIN = 5
RATE_WINDOW_S = 60.0

# ip -> list[timestamps]
_rate_buckets: Dict[str, List[float]] = {}

CACHE_TTL_S = 3600.0
# key -> (stored_at_ts, payload)
_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}


def _rate_limit_check(ip: str) -> None:
    now = time.time()
    bucket = _rate_buckets.get(ip, [])
    bucket = [t for t in bucket if now - t < RATE_WINDOW_S]
    if len(bucket) >= RATE_LIMIT_PER_MIN:
        retry_after = int(max(1, RATE_WINDOW_S - (now - bucket[0])))
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Try again in ~{retry_after}s.",
            headers={"Retry-After": str(retry_after)},
        )
    bucket.append(now)
    _rate_buckets[ip] = bucket


def _cache_get(key: str) -> Optional[Dict[str, Any]]:
    now = time.time()
    item = _cache.get(key)
    if not item:
        return None
    stored_at, payload = item
    if now - stored_at > CACHE_TTL_S:
        _cache.pop(key, None)
        return None
    return payload


def _cache_set(key: str, payload: Dict[str, Any]) -> None:
    _cache[key] = (time.time(), payload)


# -----------------------------
# Exclusion + playbook mapping
# -----------------------------

def _exclusion_string_to_int(s: str) -> Optional[int]:
    # scanner uses strings like "#3 Not Tokenized"
    if not s:
        return None
    if s.startswith("#") and len(s) >= 2 and s[1].isdigit():
        try:
            return int(s[1])
        except ValueError:
            return None
    return None


_PLAYBOOK_TEMPLATES: Dict[int, Dict[str, Any]] = {
    1: {
        "title": "Fix Crawlability",
        "priority": "P0",
        "timeline": "2-4 hours",
        "expected_impact": "+10 points",
        "actions": [
            "Audit robots.txt for accidental disallows (/, homepage, key landing pages).",
            "Ensure sitemap.xml exists and is accessible (200 OK).",
            "Submit sitemap(s) in Google Search Console and request re-crawl of key URLs.",
        ],
    },
    2: {
        "title": "Improve Content Quality Signals",
        "priority": "P1",
        "timeline": "1-2 weeks",
        "expected_impact": "+5 to +10 points",
        "actions": [
            "Expand thin homepage sections with entity-rich copy (who/what/why).",
            "Add FAQs and clear product/service explanations.",
            "Strengthen internal linking to key intent pages.",
        ],
    },
    3: {
        "title": "Get Tokenized (Entity + Structured Data)",
        "priority": "P0",
        "timeline": "3-7 days",
        "expected_impact": "+15 to +25 points",
        "actions": [
            "Add/validate schema.org Organization JSON-LD (name, url, logo, sameAs, identifiers).",
            "Create or improve a Wikidata entity with reliable references.",
            "Ensure consistent brand naming across site, social profiles, and structured data.",
        ],
    },
    4: {
        "title": "Increase Learned Signals (Mentions/Context)",
        "priority": "P2",
        "timeline": "2-6 weeks",
        "expected_impact": "+5 to +15 points",
        "actions": [
            "Publish comparison pages (Brand vs X, alternatives, best-for) targeting real queries.",
            "Invest in PR/earned media and community participation where models ingest signals.",
            "Create consistent citations across profiles/directories (same name, site, and category).",
        ],
    },
    5: {
        "title": "Remove Indexing Blockers",
        "priority": "P0",
        "timeline": "2-4 hours",
        "expected_impact": "+10 points",
        "actions": [
            "Remove disallow rules blocking important sections.",
            "Confirm canonical tags and noindex directives are correct.",
            "Validate structured data and resubmit for indexing where applicable.",
        ],
    },
    6: {
        "title": "Fix Semantic Alignment",
        "priority": "P1",
        "timeline": "3-7 days",
        "expected_impact": "+5 to +10 points",
        "actions": [
            "Align title/meta to the brand + category intent (clear positioning).",
            "Create intent-specific landing pages for top queries and use consistent entity language.",
            "Ensure on-page copy matches the category you want the model to learn.",
        ],
    },
}


def _build_playbook(exclusion_points: List[int]) -> List[PlaybookItem]:
    items: List[PlaybookItem] = []
    for e in exclusion_points:
        t = _PLAYBOOK_TEMPLATES.get(e)
        if not t:
            continue
        items.append(
            PlaybookItem(
                exclusion=e,
                title=t["title"],
                priority=t["priority"],
                actions=list(t["actions"]),
                timeline=t["timeline"],
                expected_impact=t["expected_impact"],
            )
        )

    # Sort by priority then exclusion id
    prio_order = {"P0": 0, "P1": 1, "P2": 2}
    items.sort(key=lambda x: (prio_order.get(x.priority, 9), x.exclusion))
    return items


# -----------------------------
# FastAPI app
# -----------------------------

app = FastAPI(title="GEO Scanner API", version=APP_VERSION)

# Wildcard origin is invalid with credentials; public API + static UI use * without cookies.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "HEAD", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Retry-After"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(_req: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error="http_error", detail=str(exc.detail)).model_dump(),
        headers=getattr(exc, "headers", None) or {},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_req: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled_error err=%s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(error="internal_error", detail="Unexpected server error").model_dump(),
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=APP_VERSION)


@app.get("/", include_in_schema=False)
def index_page() -> FileResponse:
    """Serve the GEO scanner SPA from static/index.html."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="index.html not found")
    return FileResponse(index_path, media_type="text/html; charset=utf-8")


def _to_scan_response_payload(scan: Dict[str, Any]) -> Dict[str, Any]:
    geo = scan.get("geo_score") or {}
    exclusion_strs = (geo.get("exclusion_points") or [])
    exclusion_ints: List[int] = []
    for s in exclusion_strs:
        v = _exclusion_string_to_int(str(s))
        if v is not None:
            exclusion_ints.append(v)
    exclusion_ints = sorted(set(exclusion_ints))

    payload: Dict[str, Any] = {
        "brand": scan.get("brand_name") or "",
        "url": scan.get("url") or "",
        "score": int(geo.get("score") or 0),
        "grade": str(geo.get("grade") or "F"),
        "exclusion_points": exclusion_ints,
        "checks": {
            "knowledge_graph": scan.get("knowledge_graph") or {},
            "schema_markup": scan.get("schema") or {},
            "wikidata": scan.get("wikidata") or {},
            "crawlability": scan.get("crawlability") or {},
        },
        "playbook": [i.model_dump() for i in _build_playbook(exclusion_ints)],
        "scanned_at": scan.get("scanned_at") or _utc_now_z(),
    }
    return payload


def _run_scan(brand_name: str, url: str) -> Dict[str, Any]:
    # scanner.scan_brand_geo returns consolidated checks + geo_score + playbook
    scan = scanner.scan_brand_geo(brand_name, url)
    scan["scanned_at"] = _utc_now_z()
    return scan


@app.post(
    "/scan",
    response_model=ScanResponse,
    responses={
        400: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def scan_endpoint(req: Request) -> Any:
    ip = _client_ip(req)
    _rate_limit_check(ip)

    try:
        body = await req.json()
        parsed = ScanRequest.model_validate(body)
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON body")

    brand = parsed.brand_name.strip()
    url = str(parsed.url)

    cache_key = f"{brand.lower()}|{_normalize_url_for_key(url)}"
    cached = _cache_get(cache_key)
    if cached:
        logger.info("scan_request cached ip=%s brand=%s url=%s", ip, brand, url)
        return cached

    logger.info("scan_request ip=%s brand=%s url=%s", ip, brand, url)

    try:
        scan = _run_scan(brand, url)
        payload = _to_scan_response_payload(scan)
        _cache_set(cache_key, payload)
        return payload
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("scan_failed ip=%s brand=%s url=%s err=%s", ip, brand, url, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Scan failed unexpectedly. Check server logs for details.",
        )


_DEMO_BRANDS: List[Dict[str, Any]] = [
    {
        "brand": "AcmeProject",
        "url": "https://acmeproject.com",
        "score": 67,
        "grade": "C",
        "exclusion_points": [1, 4],
        "checks": {
            "knowledge_graph": {
                "found": False,
                "confidence": 0.0,
                "description": "No Knowledge Graph results",
                "rate_limited": False,
                "kg_score_points": 0,
            },
            "schema_markup": {"has_schema": True, "schema_data": {"@type": "Organization", "name": "AcmeProject"}, "errors": []},
            "wikidata": {"found": False, "qid": "", "description": "No matching Wikidata item"},
            "crawlability": {"crawlable": True, "robots_issues": []},
        },
        "playbook": [],  # filled below
        "scanned_at": "2026-04-24T09:00:00Z",
    },
    {
        "brand": "Contoso",
        "url": "https://contoso.com",
        "score": 82,
        "grade": "B",
        "exclusion_points": [6],
        "checks": {
            "knowledge_graph": {
                "found": True,
                "confidence": 0.78,
                "description": "Brand entity present",
                "rate_limited": False,
                "kg_score_points": 25,
            },
            "schema_markup": {"has_schema": True, "schema_data": {"@type": "Organization", "name": "Contoso"}, "errors": []},
            "wikidata": {"found": True, "qid": "Q12345", "description": "Example company"},
            "crawlability": {"crawlable": True, "robots_issues": []},
        },
        "playbook": [],
        "scanned_at": "2026-04-24T09:00:00Z",
    },
    {
        "brand": "Northwind",
        "url": "https://northwind.example",
        "score": 54,
        "grade": "E",
        "exclusion_points": [1, 2, 3, 5],
        "checks": {
            "knowledge_graph": {
                "found": False,
                "confidence": 0.0,
                "description": "Missing GOOGLE_KG_API_KEY env var",
                "rate_limited": False,
                "kg_score_points": 0,
            },
            "schema_markup": {"has_schema": False, "schema_data": {}, "errors": ["No Organization JSON-LD found"]},
            "wikidata": {"found": False, "qid": "", "description": "No matching Wikidata item"},
            "crawlability": {"crawlable": False, "robots_issues": ["Homepage appears disallowed by robots.txt", "sitemap.xml not found or inaccessible"]},
        },
        "playbook": [],
        "scanned_at": "2026-04-24T09:00:00Z",
    },
]

# Fill demo playbooks deterministically
for d in _DEMO_BRANDS:
    d["playbook"] = [i.model_dump() for i in _build_playbook(d.get("exclusion_points", []))]


@app.get("/demo-brands", response_model=List[ScanResponse])
def demo_brands() -> List[Dict[str, Any]]:
    return _DEMO_BRANDS


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

