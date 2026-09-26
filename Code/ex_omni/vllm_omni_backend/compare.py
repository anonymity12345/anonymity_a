"""Compare matched runs without treating stochastic output differences as parity."""

import argparse
import json
from pathlib import Path

from .benchmark import summarize


def compare(baseline, candidate):
    baseline, candidate = Path(baseline), Path(candidate)
    metadata = [json.loads((p / "metadata.json").read_text()) for p in (baseline, candidate)]
    required = ("sampling", "seed", "max_new_tokens", "include_render", "render_batch_size",
                "sample_rate", "video_fps", "image_size", "checkpoint_config_sha256", "hardware", "samples_per_mode",
                "devices", "post_device")
    mismatches = [key for key in required if metadata[0].get(key) != metadata[1].get(key)]
    # Local file paths may differ between environments, audio bytes must match.
    speech_hashes = [[f["sha256"] for f in m["speech_files"]] for m in metadata]
    if speech_hashes[0] != speech_hashes[1]:
        mismatches.append("speech_files")
    modes = []
    for mode in ("text", "speech"):
        paths = [p / f"raw_{mode}.json" for p in (baseline, candidate)]
        if any(p.exists() for p in paths):
            if not all(p.exists() for p in paths):
                mismatches.append(f"missing_{mode}")
                continue
            records = [json.loads(p.read_text()) for p in paths]
            if len(records[0]) != len(records[1]) or len(records[0]) != metadata[0]["samples_per_mode"]:
                mismatches.append(f"incomplete_{mode}")
                continue
            differences = []
            for index, (a, b) in enumerate(zip(*records)):
                if (a["sample_idx"], a["prompt"]) != (b["sample_idx"], b["prompt"]):
                    mismatches.append(f"prompt_{mode}_{index}")
                fields = [key for key in ("text_token_ids", "speech_units") if a[key] != b[key]]
                if fields:
                    differences.append(dict(record=index, sample_idx=a["sample_idx"], fields=fields))
            modes.append(dict(input_mode=mode, baseline=summarize(records[0]), candidate=summarize(records[1]), differences=differences))
    exact = bool(modes) and not any(m["differences"] for m in modes)
    return dict(
        metadata_mismatches=mismatches, exact_tokens_and_units=exact,
        comparable=not mismatches and bool(modes), sampling=metadata[0]["sampling"], modes=modes,
        quality_validated=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = compare(args.baseline, args.candidate)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    rows = ["# HF vs vLLM-Omni", "", f"Comparable settings: {report['comparable']}. Exact tokens/units: {report['exact_tokens_and_units']}.", "",
            "All completed samples are reported, including differing outputs. RAS runs require repeated trials and audio/video quality review.", "",
            "| Input | HF RTF | Omni RTF | Change | HF P50 s | Omni P50 s | Differing outputs |", "|:--|--:|--:|--:|--:|--:|--:|"]
    for mode in report["modes"]:
        a, b = mode["baseline"], mode["candidate"]
        rows.append(f"| {mode['input_mode']} | {a['avg_rtf']:.4f} | {b['avg_rtf']:.4f} | {(b['avg_rtf'] / a['avg_rtf'] - 1) * 100:+.2f}% | {a['p50_latency_s']:.3f} | {b['p50_latency_s']:.3f} | {len(mode['differences'])} |")
    if report["metadata_mismatches"]:
        rows.extend(["", "Settings mismatch: " + ", ".join(report["metadata_mismatches"])])
    rows.extend(["", "No acceleration or quality acceptance is implied by this report. First-chunk latency was not measured."])
    (args.output / "comparison.md").write_text("\n".join(rows) + "\n")
    if not report["comparable"] or (report["sampling"] == "greedy" and not report["exact_tokens_and_units"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
