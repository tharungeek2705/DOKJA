import sys
import uvicorn
import yaml
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def main():
    config_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    app_cfg = config.get("app", {})
    host = app_cfg.get("host", "0.0.0.0")
    port = app_cfg.get("port", 8000)

    print("=" * 65)
    print("  DOKJA — AI Smart Traffic & Vehicle Intelligence Platform")
    print(f"  Starting server on http://localhost:{port}")
    print("=" * 65)

    uvicorn.run(
        "backend.main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    main()
