"""Episode 2: does a random train/test split leak?

    python scripts/split_leakage.py

FLEURS has the same sentence read by several people. A random split by clip puts the same
sentence (and often the same speaker) on both sides: a fine-tuned model is then tested on
text it has memorised. We split by group instead and write data/splits/{train,test}.csv.
FLEURS publishes no speaker id, so the sentence id is the best group we have here; with
your own data, group by speaker (and patient, facility, day) as well.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
P = yaml.safe_load((ROOT / "configs" / "params.yaml").read_text(encoding="utf-8"))


def leak(train: pd.DataFrame, test: pd.DataFrame) -> float:
    return float(test.sentence_id.isin(set(train.sentence_id)).mean())


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "manifest.csv", dtype={"clip_id": str})
    frac, seed = P["split"]["test_fraction"], P["seed"]
    test = df.sample(frac=frac, random_state=seed)
    train = df.drop(test.index)
    print(f"random split by clip:      {len(train)} train / {len(test)} test, "
          f"test clips whose sentence is also in train: {leak(train, test):.0%}")

    rng = np.random.default_rng(seed)
    groups = df.sentence_id.unique()
    test_groups = set(rng.choice(groups, size=int(round(frac * len(groups))), replace=False))
    test = df[df.sentence_id.isin(test_groups)]
    train = df[~df.sentence_id.isin(test_groups)]
    print(f"grouped split by sentence: {len(train)} train / {len(test)} test, "
          f"test clips whose sentence is also in train: {leak(train, test):.0%}")
    out = ROOT / "data" / "splits"
    out.mkdir(exist_ok=True)
    train.to_csv(out / "train.csv", index=False, encoding="utf-8")
    test.to_csv(out / "test.csv", index=False, encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}/train.csv and test.csv")
    print(f"gender column values: {df.gender.value_counts().to_dict()}  <- check before you trust it")


if __name__ == "__main__":
    main()
