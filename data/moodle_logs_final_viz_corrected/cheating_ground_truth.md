# Cheating Ground Truth (Generator Configuration)

This file summarizes the configuration parameters used to generate cheating patterns.

## Cheating Groups Configuration

| Group ID             | Severity        | Member User IDs     | Nav Similarity | Nav Noise | Timing Start Delay (min/member) | Timing Variance (s) | Completion Speed | Answer Similarity | Wrong Answer Bias |
|----------------------|-----------------|---------------------|----------------|-----------|---------------------------------|---------------------|------------------|-------------------|-------------------|
| group_high_severity_1 | high_severity   | [1, 2, 3]           | 0.96           | 0.07      | 1                               | 2                   | fast             | 0.94              | 0.87              |
| group_medium_severity_1 | medium_severity | [4, 5, 6, 7]        | 0.80           | 0.30      | 5                               | 15                  | medium           | 0.84              | 0.60              |
| group_low_severity_1 | low_severity    | [8, 9, 10, 11, 12]  | 0.55           | 0.45      | 10                              | 30                  | slow             | 0.55              | 0.40              |


