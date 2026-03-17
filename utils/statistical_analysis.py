from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, shapiro


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Statistical analysis for probe vs zero-shot.")
    parser.add_argument(
        "--probe-csv",
        default="test_125_fixed/per_case_dice.csv",
        help="Path to probe per-case dice CSV.",
    )
    parser.add_argument(
        "--zeroshot-csv",
        default="results/zero shot/zeroshot_brats21_results_checkpoint_per_slice.csv",
        help="Path to zero-shot per-case dice CSV.",
    )
    parser.add_argument(
        "--out",
        default="results/statistical_analysis.txt",
        help="Path to write the summary output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load results
    probe = pd.read_csv(args.probe_csv)
    zeroshot = pd.read_csv(args.zeroshot_csv)

    def to_case_level(df: pd.DataFrame, dice_col: str = "dice") -> pd.DataFrame:
        if "case" not in df.columns:
            raise ValueError("Expected a 'case' column for matching.")
        if "slice" in df.columns:
            # Aggregate slice-level rows to per-case mean dice
            return df.groupby("case", as_index=False)[dice_col].mean()
        return df[["case", dice_col]].copy()

    probe = to_case_level(probe, dice_col="dice")
    zeroshot = to_case_level(zeroshot, dice_col="dice")

    probe_matched = probe.sort_values("case")
    zeroshot_matched = zeroshot.sort_values("case")

    # Align on case ID to guarantee pairing
    merged = probe_matched.merge(zeroshot_matched, on="case", suffixes=("_probe", "_zeroshot"))

    probe_dice = merged["dice_probe"].values
    zeroshot_dice = merged["dice_zeroshot"].values

    lines = []

    # 1. Wilcoxon signed-rank test (one-sided: probe > zero-shot)
    stat, p = wilcoxon(probe_dice, zeroshot_dice, alternative="greater")
    lines.append(f"Wilcoxon: W={stat:.1f}, p={p:.4e}")

    # 2. Summary stats
    lines.append(
        f"\nProbe    mean={probe_dice.mean():.3f}  median={np.median(probe_dice):.3f}  std={probe_dice.std():.3f}"
    )
    lines.append(
        f"ZeroShot mean={zeroshot_dice.mean():.3f}  median={np.median(zeroshot_dice):.3f}  std={zeroshot_dice.std():.3f}"
    )
    lines.append(f"n matched cases: {len(merged)}")

    # 3. Optional: check if normal (to justify Wilcoxon vs t-test)
    _, p_norm = shapiro(probe_dice - zeroshot_dice)
    lines.append(
        f"\nShapiro-Wilk on differences: p={p_norm:.4f} "
        f"({'normal' if p_norm > 0.05 else 'not normal -- Wilcoxon correct'})"
    )

    output = "\n".join(lines)
    print(output)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()