import json
from pathlib import Path
from typing import Dict, Any


def load_config() -> Dict[str, Any]:
    cfg_path = Path(__file__).parent / 'config.json'
    if not cfg_path.exists():
        return {}

    try:
        with cfg_path.open('r', encoding='utf-8') as fh:
            return json.load(fh)
    except Exception:
        return {}
