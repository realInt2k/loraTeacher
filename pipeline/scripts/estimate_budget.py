"""Print the estimated GPU-hour budget of a config.

    python scripts/estimate_budget.py --config configs/main_teacher_x_student.yaml

Edit the per-unit costs under ``budget.costs`` in the config.
"""

import argparse

import _path  # noqa: F401

from subliminal.budget import estimate_budget, format_budget
from subliminal.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    estimate = estimate_budget(load_config(args.config))
    print(format_budget(estimate))
    if estimate["over_cap"]:
        print("Over the cap: run_stage.py will refuse to launch without --confirm.")


if __name__ == "__main__":
    main()
