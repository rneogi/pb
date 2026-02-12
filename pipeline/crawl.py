"""
Crawl Pipeline Stage
====================
Fetches content from configured sources with rate limiting and retry logic.
Supports HTTP (httpx) and JavaScript rendering (Playwright).
"""

import asyncio
import hashlib
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import urlparse, urljoin
import xml.etree.ElementTree as ET

import httpx

from .config import (
    load_sources_config, load_pipeline_config,
    ensure_week_dirs, SourceCategory, SourceEntry
)
from .database import update_crawl_state, get_crawl_state, upsert_artifact

# Rate limiting state per domain
_domain_last_request: Dict[str, float] = {}


def get_domain(url: str) -> str:
    """Extract domain from URL."""
    parsed = urlparse(url)
    return parsed.netloc


def normalize_url(url: str) -> str:
    """Normalize URL for canonical comparison."""
    parsed = urlparse(url)
    # Remove trailing slashes, fragments, normalize scheme
    path = parsed.path.rstrip("/") or "/"
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def compute_content_hash(content: str) -> str:
    """Compute SHA256 hash of content."""
    # Normalize whitespace for consistent hashing
    normalized = re.sub(r'\s+', ' ', content.strip())
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def compute_artifact_id(canonical_url: str, content_hash: str) -> str:
    """Compute unique artifact ID."""
    combined = f"{canonical_url}::{content_hash}"
    return hashlib.sha256(combined.encode('utf-8')).hexdigest()[:16]


async def rate_limit(domain: str, rate_limit_seconds: float):
    """Apply rate limiting per domain."""
    now = time.time()
    last_request = _domain_last_request.get(domain, 0)
    wait_time = rate_limit_seconds - (now - last_request)
    if wait_time > 0:
        await asyncio.sleep(wait_time)
    _domain_last_request[domain] = time.time()


async def fetch_http(
    url: str,
    config: Dict[str, Any],
    user_agent: str,
    etag: Optional[str] = None,
    last_modified: Optional[str] = None
) -> Tuple[Optional[str], int, Dict[str, str]]:
    """
    Fetch URL using httpx.
    Returns: (content, status_code, headers_dict)
    """
    timeout = config.get("timeout", 30)
    headers = {"User-Agent": user_agent}

    # Conditional GET headers
    if etag:
        headers["If-None-Match"] = etag
    if last_modified:
        headers["If-Modified-Since"] = last_modified

    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=True,
        http2=True
    ) as client:
        response = await client.get(url, headers=headers)

        response_headers = {
            "etag": response.headers.get("etag"),
            "last_modified": response.headers.get("last-modified"),
            "content_type": response.headers.get("content-type", "")
        }

        if response.status_code == 304:
            # Not modified
            return None, 304, response_headers

        return response.text, response.status_code, response_headers


async def fetch_playwright(
    url: str,
    config: Dict[str, Any]
) -> Tuple[Optional[str], int, Dict[str, str]]:
    """
    Fetch URL using Playwright for JavaScript rendering.
    Returns: (content, status_code, headers_dict)
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        print(f"WARNING: Playwright not installed, falling back to HTTP for {url}")
        return await fetch_http(url, config, config.get("user_agent", ""))

    pw_config = config.get("playwright", {})
    timeout = pw_config.get("timeout", 60000)
    wait_until = pw_config.get("wait_until", "networkidle")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=pw_config.get("headless", True))
        page = await browser.new_page()

        try:
            response = await page.goto(url, timeout=timeout, wait_until=wait_until)
            content = await page.content()
            status = response.status if response else 200
            await browser.close()
            return content, status, {}
        except Exception as e:
            await browser.close()
            print(f"Playwright error for {url}: {e}")
            return None, 0, {}


def parse_rss_feed(content: str, base_url: str) -> List[Dict[str, Any]]:
    """Parse RSS/Atom feed and extract entries."""
    entries = []
    try:
        root = ET.fromstring(content)

        # Handle different feed formats
        # RSS 2.0
        for item in root.findall(".//item"):
            entry = {
                "title": item.findtext("title", ""),
                "url": item.findtext("link", ""),
                "published_at": item.findtext("pubDate", ""),
                "description": item.findtext("description", "")
            }
            if entry["url"]:
                entries.append(entry)

        # Atom
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        for item in root.findall(".//atom:entry", ns):
            link_elem = item.find("atom:link[@rel='alternate']", ns)
            if link_elem is None:
                link_elem = item.find("atom:link", ns)
            url = link_elem.get("href", "") if link_elem is not None else ""

            entry = {
                "title": item.findtext("atom:title", "", ns),
                "url": url,
                "published_at": item.findtext("atom:updated", "", ns) or item.findtext("atom:published", "", ns),
                "description": item.findtext("atom:summary", "", ns)
            }
            if entry["url"]:
                # Handle relative URLs
                if not entry["url"].startswith("http"):
                    entry["url"] = urljoin(base_url, entry["url"])
                entries.append(entry)

    except ET.ParseError as e:
        print(f"RSS parse error: {e}")

    return entries


async def crawl_source(
    category: SourceCategory,
    entry: SourceEntry,
    week: str,
    dirs: Dict[str, Path],
    pipeline_config: Dict[str, Any],
    settings: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Crawl a single source entry.
    Returns list of artifact metadata dicts.
    """
    artifacts = []
    domain = get_domain(entry.url)
    rate_limit_seconds = category.rate_limit

    # Apply rate limiting
    await rate_limit(domain, rate_limit_seconds)

    user_agent = settings.get("user_agent", "PublicPitchBookObserver/1.0")
    crawl_config = pipeline_config

    # Check existing crawl state for conditional requests
    canonical_url = normalize_url(entry.url)
    crawl_state = get_crawl_state(canonical_url)
    etag = crawl_state["last_etag"] if crawl_state else None
    last_modified = crawl_state["last_modified"] if crawl_state else None

    print(f"  Crawling: {entry.name} ({entry.url})")

    # Fetch based on mode
    if entry.fetch_mode == "playwright":
        content, status, headers = await fetch_playwright(entry.url, crawl_config)
    else:
        content, status, headers = await fetch_http(
            entry.url, crawl_config, user_agent, etag, last_modified
        )

    if status == 304:
        print(f"    Not modified (304)")
        update_crawl_state(canonical_url, crawl_state["last_content_hash"], week,
                          etag, last_modified, unchanged=True)
        return artifacts

    if not content or status >= 400:
        print(f"    Failed with status {status}")
        return artifacts

    retrieved_at = datetime.utcnow().isoformat()

    # Handle RSS/Atom feeds - fetch individual articles
    if entry.format in ("rss", "atom"):
        feed_entries = parse_rss_feed(content, entry.url)
        print(f"    Found {len(feed_entries)} feed entries")

        for feed_entry in feed_entries[:20]:  # Limit per feed
            await rate_limit(get_domain(feed_entry["url"]), rate_limit_seconds)

            try:
                article_content, article_status, _ = await fetch_http(
                    feed_entry["url"], crawl_config, user_agent
                )
                if article_content and article_status == 200:
                    artifact = await save_artifact(
                        content=article_content,
                        url=feed_entry["url"],
                        source_name=entry.name,
                        source_kind=category.source_kind,
                        week=week,
                        dirs=dirs,
                        fetch_mode="http",
                        http_status=article_status,
                        published_at=feed_entry.get("published_at"),
                        title=feed_entry.get("title")
                    )
                    if artifact:
                        artifacts.append(artifact)
            except Exception as e:
                print(f"    Error fetching {feed_entry['url']}: {e}")

    else:
        # Direct HTML page
        artifact = await save_artifact(
            content=content,
            url=entry.url,
            source_name=entry.name,
            source_kind=category.source_kind,
            week=week,
            dirs=dirs,
            fetch_mode=entry.fetch_mode,
            http_status=status,
            etag=headers.get("etag"),
            last_modified=headers.get("last_modified")
        )
        if artifact:
            artifacts.append(artifact)

    return artifacts


async def save_artifact(
    content: str,
    url: str,
    source_name: str,
    source_kind: str,
    week: str,
    dirs: Dict[str, Path],
    fetch_mode: str,
    http_status: int,
    etag: Optional[str] = None,
    last_modified: Optional[str] = None,
    published_at: Optional[str] = None,
    title: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Save artifact to disk and database."""
    canonical_url = normalize_url(url)
    content_hash = compute_content_hash(content)
    artifact_id = compute_artifact_id(canonical_url, content_hash)
    domain = get_domain(url)
    retrieved_at = datetime.utcnow().isoformat()

    # Create source subdirectory
    source_dir_name = re.sub(r'[^\w\-]', '_', source_name)[:50]
    raw_dir = dirs["raw"] / source_dir_name
    meta_dir = dirs["meta"] / source_dir_name
    raw_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    # Determine file extension
    ext = ".html"
    if "xml" in (content[:100] if content else ""):
        ext = ".xml"

    # Save raw content
    raw_path = raw_dir / f"{artifact_id}{ext}"
    raw_path.write_text(content, encoding="utf-8")

    # Build metadata
    metadata = {
        "artifact_id": artifact_id,
        "url": url,
        "canonical_url": canonical_url,
        "content_hash": content_hash,
        "source_name": source_name,
        "source_kind": source_kind,
        "domain": domain,
        "week": week,
        "retrieved_at": retrieved_at,
        "published_at": published_at,
        "fetch_mode": fetch_mode,
        "http_status": http_status,
        "etag": etag,
        "last_modified": last_modified,
        "title": title,
        "raw_path": str(raw_path.relative_to(raw_path.parent.parent.parent.parent))
    }

    # Save metadata
    meta_path = meta_dir / f"{artifact_id}.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metadata["meta_path"] = str(meta_path.relative_to(meta_path.parent.parent.parent.parent))

    # Update database
    upsert_artifact(metadata)
    update_crawl_state(canonical_url, content_hash, week, etag, last_modified)

    print(f"    Saved: {artifact_id} ({domain})")
    return metadata


async def run_crawl(week: str, sources_config=None, pipeline_config=None) -> Dict[str, Any]:
    """
    Run the crawl stage for a given week.
    Returns summary of crawl results.
    """
    if sources_config is None:
        sources_config = load_sources_config()
    if pipeline_config is None:
        pipeline_config = load_pipeline_config()

    dirs = ensure_week_dirs(week)
    crawl_cfg = pipeline_config.crawl

    all_artifacts = []
    errors = []

    print(f"Starting crawl for week {week}")
    print(f"Output directories: {dirs['raw']}")

    enabled_sources = sources_config.get_all_enabled_sources()
    print(f"Found {len(enabled_sources)} enabled source entries")

    for category, entry in enabled_sources:
        try:
            artifacts = await crawl_source(
                category=category,
                entry=entry,
                week=week,
                dirs=dirs,
                pipeline_config=crawl_cfg,
                settings=sources_config.settings
            )
            all_artifacts.extend(artifacts)
        except Exception as e:
            error_msg = f"Error crawling {entry.name}: {e}"
            print(f"  ERROR: {error_msg}")
            errors.append(error_msg)

    summary = {
        "week": week,
        "total_artifacts": len(all_artifacts),
        "sources_crawled": len(enabled_sources),
        "errors": errors,
        "completed_at": datetime.utcnow().isoformat()
    }

    # Save crawl summary
    summary_path = dirs["runs"] / "crawl_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nCrawl complete: {len(all_artifacts)} artifacts from {len(enabled_sources)} sources")
    if errors:
        print(f"Errors: {len(errors)}")

    return summary


def main(week: str):
    """Entry point for crawl stage."""
    return asyncio.run(run_crawl(week))


if __name__ == "__main__":
    import sys
    from .config import get_current_week

    week = sys.argv[1] if len(sys.argv) > 1 else get_current_week()
    main(week)
