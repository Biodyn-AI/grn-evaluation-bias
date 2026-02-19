#!/usr/bin/env python3
"""Validate REPORT.pdf tables against CSV sources."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from PyPDF2 import PdfReader


def _load_csv(path: Path) -> List[dict]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _normalize_pdf_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = _insert_spaces_between_letters_and_digits(text)
    text = " ".join(text.split())
    return text


def _insert_spaces_between_letters_and_digits(text: str) -> str:
    out = []
    prev = ""
    for ch in text:
        if prev:
            if (prev.isalpha() or prev == "_") and ch.isdigit():
                out.append(" ")
            elif prev.isdigit() and (ch.isalpha() or ch == "_"):
                out.append(" ")
        out.append(ch)
        prev = ch
    return "".join(out)


def _concat_tokens(tokens: List[str], start: int, length: int) -> str:
    return "".join(tokens[start : start + length])


def _match_tokens(
    tokens: List[str], start: int, candidates: set, max_len: int = 6
) -> Tuple[str | None, int]:
    for length in range(max_len, 0, -1):
        if start + length <= len(tokens):
            candidate = _concat_tokens(tokens, start, length)
            if candidate in candidates:
                return candidate, length
    return None, 0


def _extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def _parse_timeseries_table(pdf_text: str, methods: set) -> Dict[str, dict]:
    header = (
        "method edges hpn_precision hpn_recall hpn_f1 hpn_aupr "
        "beeline_precision beeline_recall beeline_f1 beeline_aupr"
    )
    start = pdf_text.find(header)
    if start == -1:
        raise ValueError("Could not locate time-series table header in PDF text.")
    table_text = _normalize_pdf_text(pdf_text[start + len(header) :])
    tokens = table_text.split()

    rows = {}
    idx = 0
    while idx < len(tokens):
        method, consumed = _match_tokens(tokens, idx, methods)
        if not method:
            break
        idx += consumed
        if idx + 9 > len(tokens):
            break
        edges = tokens[idx]
        values = tokens[idx + 1 : idx + 9]
        idx += 9
        rows[method] = {
            "method": method,
            "edges": edges,
            "hpn_precision": values[0],
            "hpn_recall": values[1],
            "hpn_f1": values[2],
            "hpn_aupr": values[3],
            "beeline_precision": values[4],
            "beeline_recall": values[5],
            "beeline_f1": values[6],
            "beeline_aupr": values[7],
        }
    return rows


def _parse_grn_table(pdf_text: str, methods: set, references: set) -> Dict[Tuple[str, str], dict]:
    header = "method reference precision recall f1 aupr pred_edges true_edges candidate_edges"
    start = pdf_text.find(header)
    if start == -1:
        raise ValueError("Could not locate GRN baseline table header in PDF text.")
    table_text = _normalize_pdf_text(pdf_text[start + len(header) :])
    tokens = table_text.split()

    rows = {}
    idx = 0
    while idx < len(tokens):
        method, consumed = _match_tokens(tokens, idx, methods)
        if not method:
            break
        idx += consumed
        reference, consumed = _match_tokens(tokens, idx, references)
        if not reference:
            break
        idx += consumed
        if idx + 7 > len(tokens):
            break
        values = tokens[idx : idx + 7]
        idx += 7
        rows[(method, reference)] = {
            "method": method,
            "reference": reference,
            "precision": values[0],
            "recall": values[1],
            "f1": values[2],
            "aupr": values[3],
            "pred_edges": values[4],
            "true_edges": values[5],
            "candidate_edges": values[6],
        }
    return rows


def _format_float(value: str) -> str:
    return f"{float(value):.6f}"


def _compare_timeseries(csv_rows: List[dict], pdf_rows: Dict[str, dict]) -> List[str]:
    issues = []
    for row in csv_rows:
        method = row["method"]
        if method not in pdf_rows:
            issues.append(f"Missing time-series row in PDF: {method}")
            continue
        pdf_row = pdf_rows[method]
        if int(row["edges"]) != int(pdf_row["edges"]):
            issues.append(
                f"Time-series {method} edges mismatch: pdf={pdf_row['edges']} csv={row['edges']}"
            )
        for key in (
            "hpn_precision",
            "hpn_recall",
            "hpn_f1",
            "hpn_aupr",
            "beeline_precision",
            "beeline_recall",
            "beeline_f1",
            "beeline_aupr",
        ):
            if _format_float(row[key]) != _format_float(pdf_row[key]):
                issues.append(
                    f"Time-series {method} {key} mismatch: pdf={_format_float(pdf_row[key])} "
                    f"csv={_format_float(row[key])}"
                )
    for method in sorted(set(pdf_rows) - {row["method"] for row in csv_rows}):
        issues.append(f"Extra time-series row in PDF: {method}")
    return issues


def _compare_grn(csv_rows: List[dict], pdf_rows: Dict[Tuple[str, str], dict]) -> List[str]:
    issues = []
    for row in csv_rows:
        key = (row["method"], row["reference"])
        if key not in pdf_rows:
            issues.append(f"Missing GRN baseline row in PDF: {row['method']} {row['reference']}")
            continue
        pdf_row = pdf_rows[key]
        for key_name in ("pred_edges", "true_edges", "candidate_edges"):
            if int(row[key_name]) != int(pdf_row[key_name]):
                issues.append(
                    f"GRN {row['method']} {row['reference']} {key_name} mismatch: "
                    f"pdf={pdf_row[key_name]} csv={row[key_name]}"
                )
        for key_name in ("precision", "recall", "f1", "aupr"):
            if _format_float(row[key_name]) != _format_float(pdf_row[key_name]):
                issues.append(
                    f"GRN {row['method']} {row['reference']} {key_name} mismatch: "
                    f"pdf={_format_float(pdf_row[key_name])} csv={_format_float(row[key_name])}"
                )
    expected = {(row["method"], row["reference"]) for row in csv_rows}
    for method, reference in sorted(set(pdf_rows) - expected):
        issues.append(f"Extra GRN baseline row in PDF: {method} {reference}")
    return issues


def _write_log(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        if lines:
            handle.write("MISMATCHES FOUND\n")
            for line in lines:
                handle.write(f"- {line}\n")
        else:
            handle.write("No mismatches found.\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate REPORT.pdf tables against CSV sources.")
    parser.add_argument("--pdf", type=Path, required=True, help="Path to REPORT.pdf")
    parser.add_argument("--timeseries", type=Path, required=True, help="Path to summary_timeseries_metrics.csv")
    parser.add_argument("--grn", type=Path, required=True, help="Path to score_eval_grn_baselines_immune.csv")
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("network_inference/outputs/report_table_validation.log"),
        help="Output log path",
    )
    args = parser.parse_args()

    pdf_text = _extract_pdf_text(args.pdf)
    timeseries_csv = _load_csv(args.timeseries)
    grn_csv = _load_csv(args.grn)

    timeseries_methods = {row["method"] for row in timeseries_csv}
    grn_methods = {row["method"] for row in grn_csv}
    grn_references = {row["reference"] for row in grn_csv}

    timeseries_pdf = _parse_timeseries_table(pdf_text, timeseries_methods)
    grn_pdf = _parse_grn_table(pdf_text, grn_methods, grn_references)

    issues = []
    issues.extend(_compare_timeseries(timeseries_csv, timeseries_pdf))
    issues.extend(_compare_grn(grn_csv, grn_pdf))

    _write_log(args.log, issues)

    if issues:
        for line in issues:
            print(line)
        print(f"Wrote mismatch log to {args.log}")
        return 1
    print(f"All rows match. Log written to {args.log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
