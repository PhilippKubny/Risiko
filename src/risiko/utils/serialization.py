from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from risiko.az.self_play import SelfPlaySample


def save_samples(path: str | Path, samples: Iterable[SelfPlaySample]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.__dict__) + "\n")


def load_samples(path: str | Path) -> List[SelfPlaySample]:
    path = Path(path)
    samples: List[SelfPlaySample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            samples.append(
                SelfPlaySample(
                    state=record["state"],
                    policy=record["policy"],
                    value=record["value"],
                )
            )
    return samples
