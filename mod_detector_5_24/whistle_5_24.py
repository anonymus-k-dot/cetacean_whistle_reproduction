"""Entry point for the 5-24 kHz detector variant:  python mod_detector_5_24/whistle_5_24.py <command> ...

Same commands as whistle.py, but every command reads mod_detector_5_24/pipeline.toml and mod_detector_5_24/detector.toml
instead of configs/. Nothing in src/ or configs/ is modified; the main detector, its caches and the dataset are untouched
(all paths in mod_detector_5_24/pipeline.toml point inside this folder)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

import whistlepipe.config as config  # noqa: E402

config.DEFAULT_PIPELINE_CFG = HERE / "pipeline.toml"
config.DEFAULT_DETECTOR_CFG = HERE / "detector.toml"

from whistlepipe.cli import main  # noqa: E402

if __name__ == "__main__":
    if "--config" in sys.argv:
        sys.exit("whistle_5_24.py always uses mod_detector_5_24/pipeline.toml; --config is not allowed here")
    sys.exit(main())
