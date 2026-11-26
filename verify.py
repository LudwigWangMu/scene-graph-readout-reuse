"""Verify the archived file hashes and recompute selected results from individual records."""
import argparse
import csv
import hashlib
import math
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def rows(name):
    with (ROOT / "data" / name).open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def check(same, label):
    if not same:
        raise ValueError(label)


def near(actual, expected, label):
    check(math.isclose(actual, float(expected), abs_tol=1e-9, rel_tol=1e-10), label)


def verify_results():
    packets = rows("shared_slot_packets.csv")
    for r in rows("shared_slot_summary.csv"):
        selected = [p for p in packets if p["rule"] == r["rule"]]
        near(len(selected), r["arrivals"], r["rule"])
        near(sum(int(p["wrong_winner"]) for p in selected), r["wrong_winner"], r["rule"])
        near(sum(float(p["output_distance"]) > 1e-10 for p in selected), r["numeric_mismatch"], r["rule"])
    directed = rows("directed_release.csv")
    stale = sum(int(p["accepted"]) and float(p["oracle_delta"]) > 1e-10 for p in directed)
    summary = json.loads((ROOT / "data/queued_summary.json").read_text())
    near(len(directed), summary["directed_tests"], "directed release count")
    near(stale, summary["accepted_stale"], "accepted stale count")
    print("graph-readouts: release and ordering summaries agree")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-only", action="store_true",
                        help="Check the numerical summaries without checking the original file digests.")
    args = parser.parse_args()
    if not args.results_only:
        for line in (ROOT / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
            expected, name = line.split("  ", 1)
            check(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name)
    verify_results()


if __name__ == "__main__":
    main()
