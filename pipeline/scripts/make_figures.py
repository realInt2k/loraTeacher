"""Regenerate every table and figure of an experiment from its results store.

    python scripts/make_figures.py --config configs/main_teacher_x_student.yaml
"""

import argparse

import _path  # noqa: F401

from subliminal.analysis import make_all
from subliminal.config import load_config
from subliminal.runs import Store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--metric", choices=["p_substring", "p_word"], default="p_substring")
    args = parser.parse_args()
    cfg = load_config(args.config)
    for path in make_all(cfg, Store(cfg), args.metric):
        print(path)


if __name__ == "__main__":
    main()
