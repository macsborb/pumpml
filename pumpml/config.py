"""Chargement de config.toml et des secrets de .env."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Secrets:
    pumpportal_key: str
    helius_key: str
    twitter_provider: str
    twitter_key: str
    jupiter_key: str


class Config:
    def __init__(self, path: Path | None = None):
        path = path or ROOT / "config.toml"
        with open(path, "rb") as f:
            self.raw = tomllib.load(f)
        load_dotenv(ROOT / ".env")
        self.secrets = Secrets(
            pumpportal_key=os.getenv("PUMPPORTAL_API_KEY", "").strip(),
            helius_key=os.getenv("HELIUS_API_KEY", "").strip(),
            twitter_provider=os.getenv("TWITTER_PROVIDER", "none").strip().lower() or "none",
            twitter_key=os.getenv("TWITTER_API_KEY", "").strip(),
            jupiter_key=os.getenv("JUPITER_API_KEY", "").strip(),
        )

    def __getitem__(self, section: str) -> dict:
        return self.raw[section]

    def path(self, name: str) -> Path:
        p = Path(self.raw["paths"][name])
        p = p if p.is_absolute() else ROOT / p
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def data(self) -> Path:
        return self.path("data")

    @property
    def models(self) -> Path:
        return self.path("models")

    @property
    def logs(self) -> Path:
        return self.path("logs")

    def pumpportal_url(self) -> str:
        url = self["collector"]["pumpportal_ws"]
        return f"{url}?api-key={self.secrets.pumpportal_key}" if self.secrets.pumpportal_key else url

    def chain_ws_url(self) -> str:
        c = self["collector"]
        if self.secrets.helius_key:
            return f"{c['helius_ws']}?api-key={self.secrets.helius_key}"
        return c["public_ws"]
