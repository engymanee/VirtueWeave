"""
Command line runner for VirtueWeave.

    python -m src.main --topic loops --grade 1 --minutes 15 --note "they keep giving up"

Prints the 3 virtues that fit best, with how each fits the topic and an
activity for that grade and time. A JSON trace of the run goes to traces/.
For the website, run:  python -m uvicorn server:app --port 8000
"""

import argparse

from src.catalog import GRADES, load_catalog
from src.pipeline import format_result, run_pipeline


def parse_args() -> argparse.Namespace:
    topics = load_catalog().topics
    parser = argparse.ArgumentParser(description="VirtueWeave: virtues to teach alongside computer science")
    parser.add_argument("--topic", required=True, choices=topics, help="the topic you are teaching")
    parser.add_argument("--grade", required=True, choices=GRADES, help="K or 1 to 12")
    parser.add_argument("--minutes", required=True, type=int, help="minutes you have")
    parser.add_argument("--note", default="", help="optional: what is happening in your class")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(format_result(run_pipeline(args.topic, args.grade, args.minutes, args.note)))


if __name__ == "__main__":
    main()
