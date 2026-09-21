import base64
import logging

import httpx

from backend import config

logger = logging.getLogger(__name__)

GITHUB_API_BASE = "https://api.github.com"


def _headers() -> dict:
    headers = {"Accept": "application/vnd.github+json"}
    if config.GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {config.GITHUB_TOKEN}"
    return headers


def _decode_content(payload: dict) -> str | None:
    content = payload.get("content")
    if not content:
        return None
    try:
        # GitHub's contents API base64-encodes with embedded newlines
        return base64.b64decode(content).decode("utf-8", errors="replace")
    except Exception as e:
        logger.warning(f"[github_client] failed to decode content: {e}")
        return None


def get_readme(owner: str, repo: str, branch: str = "main") -> tuple[str, str] | None:
    """Returns (readme_text, github_html_url), or None if missing/unreachable.
    Uses GitHub's dedicated /readme endpoint, which resolves the actual
    README file regardless of exact casing/extension.
    """
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/readme"
    try:
        response = httpx.get(url, headers=_headers(), params={
                             "ref": branch}, timeout=15.0)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        payload = response.json()
        text = _decode_content(payload)
        if text is None:
            return None
        return text, payload.get("html_url", repo_url(owner, repo))
    except httpx.HTTPError as e:
        logger.warning(
            f"[github_client] get_readme failed for {owner}/{repo}: {e}")
        return None


def get_file(owner: str, repo: str, path: str, branch: str = "main") -> tuple[str, str] | None:
    """Returns (file_text, github_html_url), or None if missing/unreachable/a directory."""
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/contents/{path}"
    try:
        response = httpx.get(url, headers=_headers(), params={
                             "ref": branch}, timeout=15.0)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, list):  # path pointed at a directory, not a file
            return None
        text = _decode_content(payload)
        if text is None:
            return None
        return text, payload.get("html_url", file_url(owner, repo, branch, path))
    except httpx.HTTPError as e:
        logger.warning(
            f"[github_client] get_file failed for {owner}/{repo}/{path}: {e}")
        return None


def list_files(owner: str, repo: str, branch: str = "main") -> list[str]:
    """Full file-path listing via the Git Trees API, so the deep-dive stage
    can choose relevant files instead of guessing filenames blindly.
    """
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/git/trees/{branch}"
    try:
        response = httpx.get(url, headers=_headers(), params={
                             "recursive": "1"}, timeout=15.0)
        response.raise_for_status()
        tree = response.json().get("tree", [])
        return [item["path"] for item in tree if item.get("type") == "blob"]
    except httpx.HTTPError as e:
        logger.warning(
            f"[github_client] list_files failed for {owner}/{repo}: {e}")
        return []


def repo_url(owner: str, repo: str) -> str:
    return f"https://github.com/{owner}/{repo}"


def file_url(owner: str, repo: str, branch: str, path: str) -> str:
    return f"https://github.com/{owner}/{repo}/blob/{branch}/{path}"
