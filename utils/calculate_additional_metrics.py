"""
Calculate additional metrics (sensitivity and precision) from the test_125_fixed\per_case_dice.csv file and zeroshot.csv file,
and write summary statistics (mean, standard deviation, 95% confidence interval, and median) to a file.
"""
import pandas as pd
import numpy as np
from scipy import stats

def calculate_probed_metrics():
    # Load data
    df = pd.read_csv('test_125_fixed/per_case_dice.csv')

    # Compute metrics
    df['sensitivity'] = df['tp'] / (df['tp'] + df['fn'])
    df['precision']   = df['tp'] / (df['tp'] + df['fp'])

    def summarise(col):
        n    = len(col)
        mean = col.mean()
        std  = col.std(ddof=1)
        se   = std / np.sqrt(n)
        ci   = stats.t.interval(0.95, df=n-1, loc=mean, scale=se)
        med  = col.median()
        return mean, std, ci, med

    # Collect results
    output_lines = []

    for name, col in [('sensitivity', df['sensitivity']),
                    ('precision',   df['precision'])]:
        
        mean, std, ci, med = summarise(col)
        
        output_lines.append(f"\nPer-case {name}")
        output_lines.append(f"  mean:   {mean:.4f}")
        output_lines.append(f"  std:    {std:.4f}")
        output_lines.append(f"  95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
        output_lines.append(f"  median: {med:.4f}")

    # Write to file
    with open('test_125_fixed/additional_metrics_summary.txt', 'w') as f:
        f.write('\n'.join(output_lines))

    print("Results saved to additional_metrics_summary.txt")

def calculate_zeroshot_metrics():
    df = pd.read_csv('results/zero shot/zeroshot_brats21_results_checkpoint_per_slice.csv')

    # Aggregate to per-case by summing raw counts
    per_case = df.groupby('case')[['tp', 'fp', 'fn']].sum().reset_index()

    # Compute per-case precision and sensitivity from aggregated counts
    per_case['precision']   = per_case['tp'] / (per_case['tp'] + per_case['fp'])
    per_case['sensitivity'] = per_case['tp'] / (per_case['tp'] + per_case['fn'])
    per_case['iou']         = per_case['tp'] / (per_case['tp'] + per_case['fp'] + per_case['fn'])

    def summarise(col):
        n    = len(col)
        mean = col.mean()
        std  = col.std(ddof=1)
        ci   = stats.t.interval(0.95, df=n-1, loc=mean, scale=std/np.sqrt(n))
        med  = col.median()
        return mean, std, ci, med

    output_lines = []
    for name in ['precision', 'sensitivity', 'iou']:
        mean, std, ci, med = summarise(per_case[name])
        output_lines.append(f"\nZero-shot per-case {name}")
        output_lines.append(f"  mean:    {mean:.4f}")
        output_lines.append(f"  std:     {std:.4f}")
        output_lines.append(f"  95% CI:  [{ci[0]:.4f}, {ci[1]:.4f}]")
        output_lines.append(f"  median:  {med:.4f}")

    # Pixel-level precision (sum all slices before dividing)
    total_tp = df['tp'].sum()
    total_fp = df['fp'].sum()
    pixel_precision = total_tp / (total_tp + total_fp)
    output_lines.append("")
    output_lines.append(f"Zero-shot pixel-level precision: {pixel_precision:.4f}")

    per_case.to_csv('results/zero shot/additional_metrics_zeroshot.csv', index=False)
    with open('results/zero shot/additional_metrics_summary_zeroshot.txt', 'w') as f:
        f.write('\n'.join(output_lines))
    print("Zero-shot summary saved to results/zero shot/additional_metrics_summary_zeroshot.txt")

if __name__ == "__main__":
    # calculate_probed_metrics()
    calculate_zeroshot_metrics()
