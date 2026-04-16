from __future__ import annotations

import argparse
import json

from ridrec.config import load_config
from ridrec.trainer import Trainer
from ridrec.utils import seed_everything



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs="+", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-path", default=None)
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    config = load_config(*args.config)
    seed_everything(config.seed)
    trainer = Trainer(config)
    metrics = trainer.evaluate_checkpoint(args.checkpoint, args.test_path)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
