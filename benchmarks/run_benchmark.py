#!/usr/bin/env python3
"""Benchmark runner for MinText. Measures MFU, step time, throughput."""

import argparse
import json
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="MinText benchmark runner")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--extra_args", nargs="*", default=[])
    args = parser.parse_args()

    # Build command
    cmd = [
        sys.executable, "-m", "mintext.train",
        "--config", args.config,
    ] + args.extra_args

    print(f"Running: {' '.join(cmd)}")
    print(f"Warmup steps: {args.warmup_steps}")
    print("=" * 60)

    # Run and capture output
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )

    step_times = []
    mfu_values = []
    tflops_values = []

    for line in proc.stdout:
        print(line, end="")
        # Parse log lines: "step=N ... step_time=X.XXXs ... MFU=Y.Y%"
        if "step=" in line and "step_time=" in line and "MFU=" in line:
            try:
                parts = line.split()
                step_num = None
                step_time = None
                mfu = None
                tflops = None
                for part in parts:
                    if part.startswith("step="):
                        step_num = int(part.split("=")[1])
                    elif part.startswith("step_time="):
                        step_time = float(part.split("=")[1].rstrip("s"))
                    elif part.startswith("MFU="):
                        mfu = float(part.split("=")[1].rstrip("%"))
                    elif part.startswith("TFLOP/s/device="):
                        tflops = float(part.split("=")[1])

                if step_num is not None and step_num >= args.warmup_steps:
                    if step_time is not None:
                        step_times.append(step_time)
                    if mfu is not None:
                        mfu_values.append(mfu)
                    if tflops is not None:
                        tflops_values.append(tflops)
            except (ValueError, IndexError):
                pass

    proc.wait()

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    if step_times:
        avg_step_time = sum(step_times) / len(step_times)
        avg_mfu = sum(mfu_values) / len(mfu_values) if mfu_values else 0
        avg_tflops = sum(tflops_values) / len(tflops_values) if tflops_values else 0

        results = {
            "avg_step_time_s": round(avg_step_time, 4),
            "avg_mfu_pct": round(avg_mfu, 2),
            "avg_tflops_per_device": round(avg_tflops, 2),
            "num_timed_steps": len(step_times),
            "warmup_steps": args.warmup_steps,
            "min_step_time_s": round(min(step_times), 4),
            "max_step_time_s": round(max(step_times), 4),
        }

        for k, v in results.items():
            print(f"  {k}: {v}")

        # Write results to JSON
        with open("benchmark_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to benchmark_results.json")
    else:
        print("  No timed steps captured!")
        sys.exit(1)


if __name__ == "__main__":
    main()
