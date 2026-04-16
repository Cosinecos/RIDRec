from __future__ import annotations

import argparse
from dataclasses import asdict

from ridrec.data import SessionPreprocessor



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--session-col", default="session_id")
    parser.add_argument("--item-col", default="item_id")
    parser.add_argument("--time-col", default="timestamp")
    parser.add_argument("--min-session-length", type=int, default=2)
    parser.add_argument("--min-item-support", type=int, default=5)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    processor = SessionPreprocessor(
        min_session_length=args.min_session_length,
        min_item_support=args.min_item_support,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
    )
    stats = processor.run(
        input_path=args.input,
        output_dir=args.output_dir,
        session_col=args.session_col,
        item_col=args.item_col,
        time_col=args.time_col,
    )
    print(asdict(stats))


if __name__ == "__main__":
    main()
