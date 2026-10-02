"""Run every experiment and write the raw result tables to ``results/``.

Usage
-----
    python scripts/run_experiments.py              # full study (about 40 min on 2 cores)
    python scripts/run_experiments.py --quick      # 2 seeds, small grids (about 1 min)

Each ``(n_features, n_train, seed)`` configuration is independent. Its output is
cached under ``results/cache/``, so an interrupted run resumes where it stopped and
adding seeds only computes the new ones. Delete the cache to recompute from scratch.
"""

from __future__ import annotations

import argparse
import json
import pickle
import platform
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from qksvm.data import CLASS_NAMES, RawSplit, load_fashion_mnist
from qksvm.experiments import Config, Grids, Record, run_config

#: PCA dimensions compared at a fixed training-set size.
N_FEATURES = (4, 6, 8)
N_TRAIN = 512
#: Training-set sizes of the learning curve, at the largest PCA dimension.
LEARNING_CURVE_N_TRAIN = (64, 128, 256, 512, 1024)
TABLES = ("sweep", "geometry", "shots")


def build_configs(seeds: range, quick: bool) -> list[Config]:
    """Enumerate the configurations of the study, cheapest first."""
    if quick:
        grids = Grids(reps=(1, 2, 4), bandwidth=(0.03125, 0.25, 1.0))
        shapes = [(4, 64), (8, 64), (8, 128)]
    else:
        grids = Grids()
        shapes = [(d, N_TRAIN) for d in N_FEATURES]
        shapes += [(max(N_FEATURES), n) for n in LEARNING_CURVE_N_TRAIN if n != N_TRAIN]
    shapes.sort(key=lambda shape: (shape[1], shape[0]))
    return [Config(d, n, seed, grids=grids) for d, n in shapes for seed in seeds]


def run_cached(
    config: Config, train: RawSplit, test: RawSplit, cache: Path
) -> dict[str, list[Record]]:
    """Run one configuration, or load it from the cache if it was already computed."""
    path = cache / f"d{config.n_features}_n{config.n_train}_s{config.seed}.pkl"
    if path.exists():
        cached_config, tables = pickle.loads(path.read_bytes())
        if cached_config == config:
            return tables
    start = time.perf_counter()
    tables = run_config(config, train, test)
    path.write_bytes(pickle.dumps((config, tables)))
    print(f"  done {config.tags()} in {time.perf_counter() - start:.0f} s", flush=True)
    return tables


def main() -> None:
    """Parse arguments, run all configurations and save the result tables."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, default=10, help="number of training subsamples")
    parser.add_argument("--jobs", type=int, default=-1, help="parallel workers (-1: all cores)")
    parser.add_argument("--data", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--quick", action="store_true", help="tiny smoke-test version")
    args = parser.parse_args()

    out: Path = args.out / "quick" if args.quick else args.out
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    train, test = load_fashion_mnist(args.data)
    configs = build_configs(range(2 if args.quick else args.seeds), args.quick)
    print(f"Running {len(configs)} configurations -> {out}/", flush=True)

    start = time.perf_counter()
    outputs = Parallel(n_jobs=args.jobs)(
        delayed(run_cached)(config, train, test, cache) for config in configs
    )
    elapsed = time.perf_counter() - start

    for table in TABLES:
        frame = pd.DataFrame([record for output in outputs for record in output[table]])
        frame.to_csv(out / f"{table}.csv", index=False, float_format="%.6g")
        print(f"Wrote {out / f'{table}.csv'} ({len(frame)} rows)")

    reference = configs[0]
    metadata = {
        "dataset": "Fashion-MNIST",
        "classes": [CLASS_NAMES[c] for c in reference.classes],
        "n_test": int(sum((test.labels == c).sum() for c in reference.classes)),
        "seeds": sorted({config.seed for config in configs}),
        "configurations": sorted({(c.n_features, c.n_train) for c in configs}),
        "n_folds": reference.n_folds,
        "grids": asdict(reference.grids),
        "wall_time_seconds": round(elapsed, 1),
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("numpy", "scipy", "scikit-learn", "pandas")},
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Finished in {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
