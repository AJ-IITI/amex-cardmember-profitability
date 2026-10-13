# Cardmember Profitability Analysis

**American Express Campus Challenge 2026 — Round 1**  
Python · financial modeling · feature engineering · ranking analysis

## Problem

Rank 500,000 anonymized premium cardmembers using 23 masked financial and behavioral attributes, and identify the most profitable 20%. The challenge did not provide a labeled profitability target, so this was an issuer-economics **ranking** problem rather than ordinary supervised classification.

**Best reported public leaderboard score: 0.910** (competition's top-20%-identification metric). The documented File5 scoring equation reproduces all 500,000 predictions in the archived File5 workbook; the official public leaderboard record is not part of this repository.

## Approach

1. **Financial reasoning.** Model a relative issuer contribution from revolving income and interchange-like spend, adjusted for risk, servicing, rewards and benefits.
2. **Data investigation.** Check missingness, distributions and correlations among the masked attributes. Examples from the original data: `f17` lending line has 58.45% missing values; `f23` email clicks has 87.79%; zero-filled `f17` and `f18` have correlation ≈0.909.
3. **Feature interactions.** Consider risk × revolving balance, risk × lending line and collections × balance, plus benefit-to-spend ratios and missingness flags. The reference scorecard includes the exposure interactions; other features are exploratory.
4. **Compare ranking hypotheses.** Examine spend-only, revenue-only, revenue-minus-risk and full P&L rankings, including sensitivity to changes in modeled costs.
5. **Select the top 20%.** Score all customers and select 100,000 of 500,000, without using the identifier as an input to the score.

The reference equation is implemented in [`src/profitability.py`](src/profitability.py). The score is a relative index; coefficients are modeling hypotheses and not official issuer rates.

## Repository

| Location | Content |
| --- | --- |
| [`src/profitability.py`](src/profitability.py) | Documented 0.910-reference profitability equation |
| [`src/score_customers.py`](src/score_customers.py) | Data loading, imputation, ranking and predictions |
| [`src/feature_engineering.py`](src/feature_engineering.py) | Exploratory financial interactions and ratios |
| [`src/research.py`](src/research.py) | EDA, baseline rankers, ablations and exploratory PCA/KMeans |
| [`scripts/make_synthetic_demo.py`](scripts/make_synthetic_demo.py) | Fabricated example data generator |
| [`docs/approach.md`](docs/approach.md) | Financial assumptions and feature choices |
| [`docs/experiments.md`](docs/experiments.md) | Historical competition experiments and results |
| [`tests/test_pipeline.py`](tests/test_pipeline.py) | Scoring, imputation, ranking and analysis checks |

## Reproduce the public demo

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m scripts.make_synthetic_demo --rows 1200
python -m src.score_customers --data sample_data/synthetic_cardmembers.csv --output outputs/predictions.csv
python -m src.research --data sample_data/synthetic_cardmembers.csv --out outputs/research --clusters
```

Outputs include the selected top-20% cohort, feature missingness plot, ranking-overlap analysis and PCA/KMeans exploratory segmentation. The example dataset is **synthetic**. To run on the challenge data, use an authorized local copy; don't publish individual cardmember records or submission workbooks.

## Interpretation

The recorded competition iterations moved from economic heuristics to the File5 scorecard by refining cost and risk assumptions; see [experiment history](docs/experiments.md). Offline ranker overlaps indicate **changes in selected customers, not accuracy against hidden labels**. PCA/KMeans and exploratory ratios were not responsible for the recorded 0.910 score. This distinction matters because the challenge withheld actual profitability labels.
