"""Export the model-ready dataset (same features, same time split) for external tools such as PyCaret.

Run:  python -m nrg.export_dataset   -> data/processed/train.csv, data/processed/test.csv
Columns: FEATURE_COLS + negative (0/1 target) + score (critic score, regression target).
"""
from __future__ import annotations

from . import config
from .train import load_dataset


def main() -> None:
    games, X, y, train_mask = load_dataset()
    df = X.assign(negative=y, score=games["metrics_review_score"].astype(float).values)
    out = config.DATA_DIR / "processed"
    out.mkdir(parents=True, exist_ok=True)
    df[train_mask].to_csv(out / "train.csv", index=False)
    df[~train_mask].to_csv(out / "test.csv", index=False)
    print(f"train {int(train_mask.sum())} rows, test {int((~train_mask).sum())} rows -> {out}")


if __name__ == "__main__":
    main()
