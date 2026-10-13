# Financial modeling approach

The challenge was to rank cardmembers by estimated *issuer net contribution*, not spending volume alone. Because actual profitability labels were withheld, the model was a scorecard calibrated using limited public leaderboard feedback rather than a supervised classifier.

**Revenue proxies:** category-specific spend and revolving balances. **Risk proxies:** average risk score interacted with revolving balance and lending-line exposure. These are not calibrated probabilities of default. **Cost proxies:** collection events, cancellation calls, lounge visits, premium credits and redeemed points. **Relationship proxies:** supplementary card accounts and website engagement.

The final equation includes 17 masked attributes. Explore their documented descriptions in the feature dictionary in the README and the reference scoring code in `src/profitability.py`. Missing spending/count fields were filled with zero, while missing risk scores were median-imputed. These are modeling assumptions; a missing credit line does not prove zero economic exposure.

The repository also investigates lending-line collinearity, spend-category reconciliation, risk-to-revolve interactions, cost-to-spend ratios and missingness flags. These exploratory features are **not all inputs to the final 0.910 equation**.

PCA/KMeans segment analysis in the research script is an offline portfolio extension. It explores attribute structure, not true profitable/unprofitable labels. Similarly, ranking overlap is sensitivity to assumptions, **not hidden-label prediction accuracy**. Coefficients are competition hypotheses, not American Express published revenue, risk or loss parameters.
