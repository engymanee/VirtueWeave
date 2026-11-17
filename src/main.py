"""
Command line runner for VirtueWeave.

    python -m src.main --ask "Teaching loops to my 6-year-olds, 15 minutes, they keep giving up."

Prints the top 3 practices and writes a JSON trace of the run to traces/.
For the website, run:  python -m uvicorn server:app --port 8000
"""

import argparse

from src.pipeline import format_result, run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VirtueWeave: coding practices that grow a virtue")
    parser.add_argument(
        "--ask",
        metavar="SENTENCE",
        required=True,
        help="describe your class in one sentence",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(format_result(run_pipeline(args.ask)))


if __name__ == "__main__":
    main()
