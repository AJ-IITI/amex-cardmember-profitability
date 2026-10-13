# Experiment history

The recorded investigation progressed from basic issuer revenue/cost hypotheses to a risk-adjusted scorecard with targeted changes to benefit and redemption penalties.

| Recorded experiment | Public score (participant reported) |
| --- | ---: |
| Early heuristic variants | approximately 0.42, then approximately 0.34 |
| Near-final benefit/reward treatment | 0.909514 |
| **File5 reference scorecard** | **0.910** |
| More relaxed benefit penalties | approximately 0.908 |
| Multiple risk/cost/engagement changes | approximately 0.907 |
| Additional blended financial rankers | approximately 0.901 |

The File5 numerical equation in `src/profitability.py` exactly reproduces the 500,000 archived File5 predictions when run on the original data; there is no independent official leaderboard export available. Alternate File4, File6, File7 and File8 submissions selected different top-100K customer cohorts. The File5_Comeback261 workbook reproduced the File5 predictions but contained a different textual methodology sheet, so its description should not be treated as separate verified model evidence.

The experiments included spend-based, revenue-based, risk-adjusted and blended financial ranking ideas. A proposed GMM/symbolic-regression method appears in the research notes, but no executed version was recovered. PCA/KMeans analysis here is an exploratory portfolio follow-up, **not** the method that earned 0.910.

Public score refers to the contest's top-20%-membership metric, not held-out accuracy of a supervised classifier.
