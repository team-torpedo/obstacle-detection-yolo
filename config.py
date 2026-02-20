import yaml
from pathlib import Path
from typing import Dict, Any


def load_config() -> Dict[str, Any]:
    cfg_path = Path(__file__).parent / 'config.yaml'
    if not cfg_path.exists():
        # Fallback to json if yaml not found, for backward compatibility during migration
        json_path = Path(__file__).parent / 'config.json'
        if json_path.exists():
            import json
            with json_path.open('r', encoding='utf-8') as fh:
                return json.load(fh)
        return {}

    try:
        with cfg_path.open('r', encoding='utf-8') as fh:
            return yaml.safe_load(fh)
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}
