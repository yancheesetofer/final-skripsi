# Cheating Ground Truth (Generator Configuration)

This file summarizes the configuration parameters used to generate cheating patterns.

## Cheating Groups Configuration

| Group ID             | Severity        | Member User IDs     | Nav Similarity | Nav Noise | Timing Start Delay (min/member) | Timing Variance (s) | Completion Speed | Answer Similarity | Wrong Answer Bias |
|----------------------|-----------------|---------------------|----------------|-----------|---------------------------------|---------------------|------------------|-------------------|-------------------|
| group_high_severity_1 | high_severity   | [1, 2, 3, 4]        | 0.92           | 0.08      | 2                               | 5                   | fast             | 0.90              | 0.85              |
| group_high_severity_2 | high_severity   | [5, 6, 7, 8]        | 0.92           | 0.08      | 2                               | 5                   | fast             | 0.90              | 0.85              |
| group_medium_severity_1 | medium_severity | [9, 10, 11, 12, 13, 14] | 0.75           | 0.25      | 5                               | 20                  | medium           | 0.70              | 0.60              |
| group_medium_severity_2 | medium_severity | [15, 16, 17, 18, 19, 20] | 0.75           | 0.25      | 5                               | 20                  | medium           | 0.70              | 0.60              |
| group_medium_severity_3 | medium_severity | [21, 22, 23, 24, 25, 26] | 0.75           | 0.25      | 5                               | 20                  | medium           | 0.70              | 0.60              |
| group_low_severity_1 | low_severity    | [27, 28, 29, 30, 31, 32, 33, 34] | 0.55           | 0.35      | 10                              | 40                  | varied           | 0.50              | 0.40              |
| group_low_severity_2 | low_severity    | [35, 36, 37, 38, 39, 40, 41, 42] | 0.55           | 0.35      | 10                              | 40                  | varied           | 0.50              | 0.40              |
| group_low_severity_3 | low_severity    | [43, 44, 45, 46, 47, 48, 49, 50] | 0.55           | 0.35      | 10                              | 40                  | varied           | 0.50              | 0.40              |


