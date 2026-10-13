import tempfile,unittest,csv
from pathlib import Path
import numpy as np
from src.profitability import ACTIVE_FEATURES,score_customer
from src.score_customers import rank_customers
from src.feature_engineering import build_features
from src.research import ablations,score_components

class PipelineTests(unittest.TestCase):
    def test_identifier_unused(self):
        d={k:"0" for k in ACTIVE_FEATURES};d["f11"]="0.1";d["f5"]="100"
        self.assertEqual(score_customer(dict(d,id="a"),risk_median=.1),score_customer(dict(d,id="b"),risk_median=.1))
    def test_median_imputation(self):
        d={k:"0" for k in ACTIVE_FEATURES}
        d["f11"]=""
        self.assertAlmostEqual(score_customer(d,risk_median=.1),score_customer(dict(d,f11=".1"),risk_median=.1))
    def test_ranking(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"input.csv"
            with path.open("w",newline="") as f:
                w=csv.DictWriter(f,fieldnames=["id",*ACTIVE_FEATURES]);w.writeheader()
                for i in range(10):
                    row={k:"0" for k in ACTIVE_FEATURES};row.update(id=str(i),f11=".1",f5=str(100*i));w.writerow(row)
            self.assertEqual([id for id,y in rank_customers(path) if y],["8","9"])
    def test_features_and_ablations(self):
        X=np.zeros((40,23));X[:,4]=np.arange(40)+1;X[:,10]=.1
        X[0,16]=np.nan
        self.assertEqual(build_features(X)["missing_lending_line"][0],1)
        _,results=ablations(X)
        self.assertEqual(results["documented_pnl"]["top20_overlap_with_reference_pct"],100.)
        c=score_components(X);self.assertEqual(len(c["revenue"]),40)

if __name__=="__main__":unittest.main()
