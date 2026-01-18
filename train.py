import sys
from pathlib import Path

# Make src/ importable
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from clutch_ai.training.train import main

if __name__ == "__main__":
    # allow LiteGPT-style: python train.py config/train_fineweb.py
    if len(sys.argv) >= 2 and sys.argv[1].endswith(".py") and not sys.argv[1].startswith("--"):
        sys.argv = [sys.argv[0], "--config", sys.argv[1]] + sys.argv[2:]
    main()
