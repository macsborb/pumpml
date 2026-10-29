"""Liens X/Twitter des coins : analyse du lien et recuperation du compte et du tweet.

Le fournisseur se choisit dans .env (TWITTER_PROVIDER). Sans fournisseur, on garde ce
qu'on deduit du lien seul (tweet, profil ou communaute, handle).
Toutes les reponses sont gardees brutes (colonne ``raw``) pour pouvoir re-extraire plus tard.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from datetime import datetime
from urllib.parse import urlparse

import aiohttp

log = logging.getLogger(__name__)

_TWEET = re.compile(r"^/([A-Za-z0-9_]{1,15})/status(?:es)?/(\d+)")
_I_TWEET = re.compile(r"^/i/(?:web/)?status/(\d+)")
_COMMUNITY = re.compile(r"^/i/communities/(\d+)")
_PROFILE = re.compile(r"^/([A-Za-z0-9_]{1,15})/?$")
_RESERVED = {"home", "search", "explore", "i", "intent", "share", "hashtag", "settings", "messages"}


TWITTER_EPOCH_MS = 1288834974657


def tweet_time_ms(tweet_id: str | None) -> int | None:
    """Date de creation d'un tweet, lue dans son identifiant (snowflake) : gratuit, sans appel API."""
    try:
        return (int(tweet_id) >> 22) + TWITTER_EPOCH_MS if tweet_id else None
    except (TypeError, ValueError):
        return None


def parse_link(url: str | None) -> dict:
    """Retourne kind (tweet/profile/community/other/none), handle, tweet_id, tweet_ms, community_id."""
    out = {"kind": "none", "handle": None, "tweet_id": None, "tweet_ms": None, "community_id": None}
    if not isinstance(url, str) or not url:
        return out  # absent (None, ou NaN venant de pandas)
    u = url.strip()
    if not re.match(r"^https?://", u, re.I):
        u = "https://" + u.lstrip("@/") if "." in u.split("/")[0] else "https://x.com/" + u.lstrip("@")
    try:
        p = urlparse(u)
    except ValueError:
        out["kind"] = "other"
        return out
    host = (p.hostname or "").lower().removeprefix("www.").removeprefix("mobile.")
    if host not in ("x.com", "twitter.com"):
        out["kind"] = "other"
        return out
    path = p.path
    if m := _TWEET.match(path):
        out.update(kind="tweet", handle=m.group(1).lower(), tweet_id=m.group(2))
    elif m := _I_TWEET.match(path):
        out.update(kind="tweet", tweet_id=m.group(1))
    elif m := _COMMUNITY.match(path):
        out.update(kind="community", community_id=m.group(1))
    elif (m := _PROFILE.match(path)) and m.group(1).lower() not in _RESERVED:
        out.update(kind="profile", handle=m.group(1).lower())
    else:
        out["kind"] = "other"
    out["tweet_ms"] = tweet_time_ms(out["tweet_id"])
    return out


def _first(*vals):
    """Premiere valeur non vide (0 est une vraie valeur : un compte a 0 abonne)."""
    return next((v for v in vals if v is not None and v != ""), None)


def _ms(s) -> int | None:
    if not s:
        return None
    for fmt in ("%a %b %d %H:%M:%S %z %Y", None):
        try:
            d = datetime.strptime(s, fmt) if fmt else datetime.fromisoformat(str(s).replace("Z", "+00:00"))
            return int(d.timestamp() * 1000)
        except (ValueError, TypeError):
            continue
    return None


def _int(v) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


class RateLimiter:
    def __init__(self, per_minute: int):
        self.gap = 60.0 / max(1, per_minute)
        self.next = 0.0
        self.lock = asyncio.Lock()

    async def wait(self):
        async with self.lock:
            now = time.monotonic()
            if self.next > now:
                await asyncio.sleep(self.next - now)
            self.next = max(now, self.next) + self.gap


class Provider:
    name = "none"

    def __init__(self, key: str, session: aiohttp.ClientSession):
        self.key, self.s = key, session

    async def fetch(self, link: dict) -> dict:
        return {}


class TwitterApiIo(Provider):
    """https://twitterapi.io — format de reponse lu de facon defensive."""
    name = "twitterapi"
    base = "https://api.twitterapi.io/twitter"

    async def _get(self, path: str, params: dict) -> dict:
        async with self.s.get(self.base + path, params=params, headers={"X-API-Key": self.key},
                              timeout=aiohttp.ClientTimeout(total=20)) as r:
            r.raise_for_status()
            return await r.json()

    @staticmethod
    def _user(u: dict) -> dict:
        return {
            "handle": (u.get("userName") or u.get("screen_name") or "").lower() or None,
            "followers": _int(_first(u.get("followers"), u.get("followers_count"))),
            "following": _int(_first(u.get("following"), u.get("friends_count"))),
            "verified": bool(u.get("isBlueVerified") or u.get("isVerified") or u.get("verified")),
            "verified_type": u.get("verifiedType") or u.get("verified_type"),
            "account_created_ms": _ms(_first(u.get("createdAt"), u.get("created_at"))),
            "statuses": _int(_first(u.get("statusesCount"), u.get("statuses_count"))),
            "bio": u.get("description"),
        }

    async def fetch(self, link: dict) -> dict:
        if link["kind"] == "tweet":
            j = await self._get("/tweets", {"tweet_ids": link["tweet_id"]})
            tw = (j.get("tweets") or [{}])[0]
            out = self._user(tw.get("author") or {})
            ents = (tw.get("entities") or {}).get("user_mentions") or []
            out.update(
                tweet_text=tw.get("text"), tweet_created_ms=_ms(tw.get("createdAt")),
                likes=_int(tw.get("likeCount")), retweets=_int(tw.get("retweetCount")),
                replies=_int(tw.get("replyCount")), quotes=_int(tw.get("quoteCount")),
                views=_int(tw.get("viewCount")), is_reply=bool(tw.get("isReply")),
                mentions=",".join(sorted({(m.get("screen_name") or "").lower() for m in ents if m})),
            )
            if tw.get("isReply") and tw.get("inReplyToId"):
                out.update(rel="reply", parent_id=str(tw["inReplyToId"]))
            elif tw.get("isQuote") and isinstance(tw.get("quoted_tweet"), dict):
                out.update(rel="quote", **parent_fields(tw["quoted_tweet"], self._user))
            out["raw"] = json.dumps(j)[:20000]
            return out
        if link["kind"] == "profile":
            j = await self._get("/user/info", {"userName": link["handle"]})
            out = self._user(j.get("data") or {})
            out["raw"] = json.dumps(j)[:20000]
            return out
        return {}


class XApiV2(Provider):
    """API officielle X v2 (Bearer token)."""
    name = "x"
    base = "https://api.x.com/2"
    ufields = "created_at,public_metrics,verified,verified_type,description"

    async def _get(self, path: str, params: dict) -> dict:
        async with self.s.get(self.base + path, params=params, headers={"Authorization": f"Bearer {self.key}"},
                              timeout=aiohttp.ClientTimeout(total=20)) as r:
            r.raise_for_status()
            return await r.json()

    def _user(self, u: dict) -> dict:
        pm = u.get("public_metrics") or {}
        return {
            "handle": (u.get("username") or "").lower() or None,
            "followers": _int(pm.get("followers_count")), "following": _int(pm.get("following_count")),
            "verified": bool(u.get("verified")), "verified_type": u.get("verified_type"),
            "account_created_ms": _ms(u.get("created_at")), "statuses": _int(pm.get("tweet_count")),
            "bio": u.get("description"),
        }

    async def fetch(self, link: dict) -> dict:
        if link["kind"] == "tweet":
            j = await self._get(f"/tweets/{link['tweet_id']}", {
                "tweet.fields": "created_at,public_metrics,entities,in_reply_to_user_id",
                "expansions": "author_id", "user.fields": self.ufields})
            tw = j.get("data") or {}
            users = (j.get("includes") or {}).get("users") or [{}]
            out = self._user(users[0])
            pm = tw.get("public_metrics") or {}
            ents = (tw.get("entities") or {}).get("mentions") or []
            out.update(
                tweet_text=tw.get("text"), tweet_created_ms=_ms(tw.get("created_at")),
                likes=_int(pm.get("like_count")), retweets=_int(pm.get("retweet_count")),
                replies=_int(pm.get("reply_count")), quotes=_int(pm.get("quote_count")),
                views=_int(pm.get("impression_count")), is_reply=bool(tw.get("in_reply_to_user_id")),
                mentions=",".join(sorted({(m.get("username") or "").lower() for m in ents})),
            )
            out["raw"] = json.dumps(j)[:20000]
            return out
        if link["kind"] == "profile":
            j = await self._get(f"/users/by/username/{link['handle']}", {"user.fields": self.ufields})
            out = self._user(j.get("data") or {})
            out["raw"] = json.dumps(j)[:20000]
            return out
        return {}


def parent_fields(tw: dict, user) -> dict:
    """Tweet d'origine (celui auquel le tweet du coin repond, ou qu'il cite)."""
    u = user(tw.get("author") or {})
    return {"parent_id": str(tw.get("id") or "") or None, "parent_handle": u["handle"],
            "parent_followers": u["followers"], "parent_verified": u["verified"],
            "parent_text": (tw.get("text") or "")[:1000], "parent_created_ms": _ms(tw.get("createdAt")),
            "parent_likes": _int(tw.get("likeCount")), "parent_views": _int(tw.get("viewCount"))}


async def fetch_parent(provider: Provider, parent_id: str) -> dict:
    """Lit le tweet d'origine d'une reponse (une requete de plus, seulement pour les reponses)."""
    if not isinstance(provider, TwitterApiIo):
        return {}
    j = await provider._get("/tweets", {"tweet_ids": parent_id})
    tw = (j.get("tweets") or [None])[0]
    return parent_fields(tw, provider._user) if tw else {}


def make_provider(name: str, key: str, session: aiohttp.ClientSession) -> Provider:
    if name == "twitterapi" and key:
        return TwitterApiIo(key, session)
    if name == "x" and key:
        return XApiV2(key, session)
    return Provider(key, session)
