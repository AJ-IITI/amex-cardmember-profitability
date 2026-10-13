"""Exploratory financial features. Additional ratios are NOT part of the File5 scorecard."""
import numpy as np

def build_features(X: np.ndarray) -> dict[str, np.ndarray]:
    """Columns f1..f23 in order; missing values represented by NaN."""
    if X.ndim != 2 or X.shape[1] != 23:
        raise ValueError("Expected an n x 23 matrix")
    X = np.where(np.isfinite(X), X, np.nan)
    z = np.nan_to_num(X, nan=0., posinf=0., neginf=0.)
    f = lambda j: z[:, j-1]
    category_sum = sum(f(i) for i in range(6, 11))
    benefit_proxy = 45*f(13) + .75*f(14) + 12*f(15) + .0005*f(21)
    return {
        "category_spend_sum": category_sum,
        "total_minus_category_spend": f(5)-category_sum,
        "risk_x_revolving_balance": f(11)*f(1),
        "risk_x_lending_line": f(11)*f(17),
        "collections_x_revolving_balance": f(3)*f(1),
        "revolve_to_spend": f(1)/(np.abs(f(5))+1),
        "benefit_cost_proxy": benefit_proxy,
        "benefit_cost_to_spend": benefit_proxy/(np.abs(f(5))+1),
        "signed_log_spend": np.sign(f(5))*np.log1p(np.abs(f(5))),
        "signed_log_revolve": np.sign(f(1))*np.log1p(np.abs(f(1))),
        "missing_lending_line": np.isnan(X[:,16]).astype(float),
        "missing_rewards_balance": np.isnan(X[:,3]).astype(float),
    }
