from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ridrec.utils import write_json, write_jsonl


@dataclass
class PreprocessStats:
    num_train: int
    num_valid: int
    num_test: int
    num_items: int
    num_sessions: int


class SessionPreprocessor:
    def __init__(
        self,
        min_session_length: int = 2,
        min_item_support: int = 5,
        train_ratio: float = 0.8,
        valid_ratio: float = 0.1,
    ) -> None:
        self.min_session_length = min_session_length
        self.min_item_support = min_item_support
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio

    def run(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        session_col: str,
        item_col: str,
        time_col: str,
    ) -> PreprocessStats:
        output_dir = Path(output_dir)
        df = pd.read_csv(input_path)
        df = df[[session_col, item_col, time_col]].copy()
        df.columns = ["session_id", "item_id", "timestamp"]
        df = df.sort_values(["session_id", "timestamp"]).reset_index(drop=True)

        session_lengths = df.groupby("session_id").size()
        valid_sessions = session_lengths[session_lengths >= self.min_session_length].index
        df = df[df["session_id"].isin(valid_sessions)].copy()

        item_support = df.groupby("item_id").size()
        valid_items = item_support[item_support >= self.min_item_support].index
        df = df[df["item_id"].isin(valid_items)].copy()

        session_lengths = df.groupby("session_id").size()
        valid_sessions = session_lengths[session_lengths >= self.min_session_length].index
        df = df[df["session_id"].isin(valid_sessions)].copy()

        session_end = df.groupby("session_id")["timestamp"].max().sort_values()
        ordered_sessions = session_end.index.tolist()
        n_sessions = len(ordered_sessions)
        train_cut = int(n_sessions * self.train_ratio)
        valid_cut = int(n_sessions * (self.train_ratio + self.valid_ratio))
        train_sessions = set(ordered_sessions[:train_cut])
        valid_sessions = set(ordered_sessions[train_cut:valid_cut])
        test_sessions = set(ordered_sessions[valid_cut:])

        train_df = df[df["session_id"].isin(train_sessions)].copy()
        train_items = set(train_df["item_id"].unique().tolist())
        valid_df = df[df["session_id"].isin(valid_sessions) & df["item_id"].isin(train_items)].copy()
        test_df = df[df["session_id"].isin(test_sessions) & df["item_id"].isin(train_items)].copy()

        item_counter = Counter(train_df["item_id"].tolist())
        item_map = {item: idx + 1 for idx, item in enumerate(sorted(item_counter))}

        train_rows = self._build_examples(train_df, item_map)
        valid_rows = self._build_examples(valid_df, item_map)
        test_rows = self._build_examples(test_df, item_map)

        write_jsonl(output_dir / "train.jsonl", train_rows)
        write_jsonl(output_dir / "valid.jsonl", valid_rows)
        write_jsonl(output_dir / "test.jsonl", test_rows)
        write_json(
            output_dir / "metadata.json",
            {
                "num_items": len(item_map),
                "num_sessions": n_sessions,
                "num_train_examples": len(train_rows),
                "num_valid_examples": len(valid_rows),
                "num_test_examples": len(test_rows),
                "session_col": session_col,
                "item_col": item_col,
                "time_col": time_col,
            },
        )
        return PreprocessStats(
            num_train=len(train_rows),
            num_valid=len(valid_rows),
            num_test=len(test_rows),
            num_items=len(item_map),
            num_sessions=n_sessions,
        )

    def _build_examples(self, df: pd.DataFrame, item_map: dict) -> list[dict]:
        rows: list[dict] = []
        grouped = df.sort_values(["session_id", "timestamp"]).groupby("session_id")
        for session_id, frame in grouped:
            mapped = [item_map[item] for item in frame["item_id"].tolist() if item in item_map]
            if len(mapped) < 2:
                continue
            for idx in range(1, len(mapped)):
                rows.append(
                    {
                        "session_id": int(session_id),
                        "items": mapped[:idx],
                        "target": mapped[idx],
                    }
                )
        return rows
