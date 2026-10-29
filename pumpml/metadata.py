"""Metadonnees des coins (JSON IPFS) et images."""
from __future__ import annotations

import asyncio
import io
import json
import logging
import re
import time
from pathlib import Path

import aiohttp

log = logging.getLogger(__name__)
_CID = re.compile(r"/ipfs/([A-Za-z0-9]+(?:/[^?#]*)?)")


# Passerelle -> instant (monotonic) jusqu'auquel on l'evite apres un 429 / une erreur.
_cooldown: dict[str, float] = {}
_rr = 0


def _gw(url: str) -> str | None:
    i = url.find("/ipfs/")
    return url[:i + 6] if i > 0 else None


def _penalize(url: str, seconds: float):
    g = _gw(url)
    if g:
        _cooldown[g] = time.monotonic() + seconds


def candidate_urls(uri: str, gateways: list[str]) -> list[str]:
    """URLs a essayer : passerelles disponibles en tourniquet (repartit la charge), puis l'URL d'origine."""
    global _rr
    if not uri:
        return []
    uri = uri.strip()
    cid = None
    if uri.startswith("ipfs://"):
        cid = uri[7:].removeprefix("ipfs/")
    elif m := _CID.search(uri):
        cid = m.group(1)
    if cid:
        now = time.monotonic()
        ok = [g for g in gateways if _cooldown.get(g, 0) <= now] or list(gateways)
        _rr += 1
        k = _rr % len(ok)
        return [g + cid for g in ok[k:] + ok[:k]] + ([uri] if uri.startswith("http") else [])
    return [uri] if uri.startswith("http") else []


def _str(v) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s[:4000] or None


def parse_metadata(j: dict) -> dict:
    ext = j.get("extensions") if isinstance(j.get("extensions"), dict) else {}
    pick = lambda *ks: next((_str(src.get(k)) for src in (j, ext) for k in ks if src.get(k)), None)
    return {
        "description": pick("description"),
        "twitter": pick("twitter", "x"),
        "telegram": pick("telegram"),
        "website": pick("website"),
        "image_uri": pick("image"),
    }


async def fetch_json(s: aiohttp.ClientSession, uri: str, gateways: list[str], timeout: float) -> dict | None:
    for url in candidate_urls(uri, gateways):
        try:
            async with s.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as r:
                if r.status != 200:
                    _penalize(url, 120 if r.status == 429 else 30)
                    continue
                data = await r.read()
                if len(data) > 200_000:
                    continue
                j = json.loads(data)
                if isinstance(j, dict):
                    return j
        except (aiohttp.ClientError, asyncio.TimeoutError):
            _penalize(url, 60)
            continue
        except (ValueError, UnicodeDecodeError):
            continue
    return None


def _save_thumb(data: bytes, path: Path) -> bool:
    from PIL import Image
    try:
        img = Image.open(io.BytesIO(data))
        img.seek(0)  # premiere image d'un GIF
        img = img.convert("RGB")
        img.thumbnail((256, 256))
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        img.save(tmp, "JPEG", quality=88)
        tmp.replace(path)
        return True
    except Exception:
        return False


async def fetch_image(s: aiohttp.ClientSession, uri: str, gateways: list[str], timeout: float,
                      max_bytes: int, path: Path) -> bool:
    for url in candidate_urls(uri, gateways):
        try:
            async with s.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as r:
                if r.status != 200:
                    _penalize(url, 120 if r.status == 429 else 30)
                    continue
                if int(r.headers.get("Content-Length") or 0) > max_bytes:
                    return False
                buf = bytearray()
                async for chunk in r.content.iter_chunked(65536):
                    buf += chunk
                    if len(buf) > max_bytes:
                        return False
                data = bytes(buf)
            if await asyncio.to_thread(_save_thumb, data, path):
                return True
        except (aiohttp.ClientError, asyncio.TimeoutError):
            _penalize(url, 60)
            continue
    return False
