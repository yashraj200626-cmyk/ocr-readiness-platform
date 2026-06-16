"""
Config manager — reads and writes teammate IP addresses from config.json.
This way IPs are updated from the Settings page without touching any code.
"""

import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

PORTS = {
    "vivek":   "8001",
    "mansi":   "8000",
    "krish":   "8002",
    "tanusha": "9001",
}

ENDPOINTS = {
    "vivek":   "/analyze-image",
    "mansi":   "/analyze",
    "krish":   "/scores",
    "tanusha": "/analyze",
}

DEFAULT_CONFIG = {
    "vivek_ip":   "127.0.0.1",
    "mansi_ip":   "127.0.0.1",
    "krish_ip":   "127.0.0.1",
    "tanusha_ip": "127.0.0.1",
}


def load_config() -> dict:
    """Load IPs from config.json. Returns defaults if file missing."""
    if not os.path.exists(CONFIG_PATH):
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG
    try:
        with open(CONFIG_PATH, "r") as f:
            data = json.load(f)
        # Fill any missing keys with defaults
        for k, v in DEFAULT_CONFIG.items():
            if k not in data:
                data[k] = v
        return data
    except Exception:
        return DEFAULT_CONFIG


def save_config(cfg: dict) -> None:
    """Save IPs to config.json."""
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=4)


def build_urls(cfg: dict) -> dict:
    """Build full API URLs from IPs stored in config."""
    return {
        "vivek":   f"http://{cfg['vivek_ip']}:{PORTS['vivek']}{ENDPOINTS['vivek']}",
        "mansi":   f"http://{cfg['mansi_ip']}:{PORTS['mansi']}{ENDPOINTS['mansi']}",
        "krish":   f"http://{cfg['krish_ip']}:{PORTS['krish']}{ENDPOINTS['krish']}",
        "tanusha": f"http://{cfg['tanusha_ip']}:{PORTS['tanusha']}{ENDPOINTS['tanusha']}",
    }


def get_urls() -> dict:
    """One-call shortcut: load config and return full URLs."""
    return build_urls(load_config())
