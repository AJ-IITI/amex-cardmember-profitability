"""Generate fabricated demonstration data. This is not Amex source data."""
import argparse,csv
from pathlib import Path
import numpy as np

def generate(path,rows=1200,seed=2026):
    rng=np.random.default_rng(seed)
    X=np.zeros((rows,23))
    spend=rng.lognormal(10,.8,rows);revolve=rng.lognormal(8.1,.95,rows)
    X[:,4]=spend;X[:,0]=revolve;X[:,10]=rng.beta(2,9,rows)
    for j,scale in [(5,.12),(6,.23),(7,.11),(8,.17),(9,.14)]:
        X[:,j]=spend*scale*rng.lognormal(0,.55,rows)
    for j,mu in [(1,.4),(2,.07),(11,23),(12,3),(14,4),(18,1),(19,1),(21,8),(22,2)]:
        X[:,j]=rng.poisson(mu,rows)
    for j,scale in [(3,9000),(13,150),(15,60),(16,18000),(17,14000),(20,25000)]:
        X[:,j]=rng.gamma(1.6,scale,rows)
    for j,prob in [(3,.4),(5,.2),(6,.2),(16,.5),(17,.5),(20,.5),(21,.5),(22,.8)]:
        X[rng.random(rows)<prob,j]=np.nan
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf8") as f:
        w=csv.writer(f);w.writerow(["id",*[f"f{i}" for i in range(1,24)]])
        for i,row in enumerate(X):
            w.writerow([f"SYNTH_{i:06}",*["" if np.isnan(x) else round(float(x),5) for x in row]])
    print(f"Wrote {rows} synthetic rows to {path}")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",default="sample_data/synthetic_cardmembers.csv")
    p.add_argument("--rows",type=int,default=1200)
    a=p.parse_args();generate(a.output,a.rows)
