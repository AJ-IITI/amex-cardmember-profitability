"""EDA, scorecard ablations and exploratory PCA/KMeans customer segmentation.

Outputs are descriptive. No actual profitability labels exist in the public dataset.
"""
import argparse, csv, json, math
from pathlib import Path
import numpy as np
from .feature_engineering import build_features
FEATURES=[f"f{i}" for i in range(1,24)]

def load_matrix(path):
    with open(path,encoding="utf-8-sig",newline="") as f:
        r=csv.DictReader(f)
        if not r.fieldnames or not set(FEATURES).issubset(r.fieldnames):
            raise ValueError("Input must contain f1 through f23")
        return np.array([[float(row[k]) if row[k] and row[k].strip() else np.nan
                          for k in FEATURES] for row in r],dtype=float)

def profile(X):
    output={}
    for j,f in enumerate(FEATURES):
        col=X[:,j]; good=col[np.isfinite(col)]
        output[f]={"missing_pct":round(float(np.mean(~np.isfinite(col))*100),2),
                   "median":float(np.median(good)) if len(good) else None,
                   "negative_pct_observed":round(float(np.mean(good<0)*100),2) if len(good) else None}
    return output

def score_components(X):
    a=np.nan_to_num(X,nan=0.).copy()
    risk=X[:,10]
    a[:,10]=np.where(np.isfinite(risk),risk,np.nanmedian(risk))
    f=lambda j:a[:,j-1]
    cat=sum(f(i) for i in range(6,11))
    return {
      "revenue":.025*f(6)+.025*f(7)+.020*f(8)-.025*f(9)+.010*f(10)+.015*f(5)+.300*f(1),
      "risk":.700*f(11)*(f(1)+cat/12)+.025*f(11)*f(17),
      "servicing":f(3)*(.2*f(1)+200)+200*f(2),
      "benefits":45*f(13)+.75*f(14)+12*f(15)+.0005*f(21),
      "relationship":20*f(19)+.05*f(12),
    }

def ablations(X):
    c=score_components(X)
    z=np.nan_to_num(X,nan=0.)
    candidates={
      "spend_only":z[:,4],
      "revenue_only":c["revenue"],
      "revenue_less_risk":c["revenue"]-c["risk"],
      "without_benefits":c["revenue"]-c["risk"]-c["servicing"]+c["relationship"],
      "documented_pnl":c["revenue"]-c["risk"]-c["servicing"]-c["benefits"]+c["relationship"],
    }
    n=math.ceil(.2*len(X))
    def top(s): return set(np.argsort(-s,kind="stable")[:n].tolist())
    ref=top(candidates["documented_pnl"])
    return candidates,{k:{"top20_overlap_with_reference_pct":round(len(top(s)&ref)*100/n,3),
                          "replacements":n-len(top(s)&ref)} for k,s in candidates.items()}

def clusters(X,reference,out):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f=build_features(X);z=np.nan_to_num(X,nan=0.)
    signed=lambda a:np.sign(a)*np.log1p(np.abs(a))
    A=np.column_stack([signed(z[:,4]),signed(z[:,0]),z[:,10],
                       np.clip(f["benefit_cost_to_spend"],0,10),
                       signed(z[:,16]),f["missing_lending_line"]])
    sample=np.random.default_rng(2026).choice(len(X),min(len(X),20000),replace=False)
    scaler=StandardScaler().fit(A[sample]);S=scaler.transform(A)
    km=KMeans(n_clusters=4,random_state=2026,n_init=5).fit(S[sample])
    group=km.predict(S)
    pca=PCA(n_components=2).fit(S[sample])
    xy=pca.transform(S[sample])
    fig,ax=plt.subplots(figsize=(7,5));n=min(2000,len(sample))
    ax.scatter(xy[:n,0],xy[:n,1],c=group[sample[:n]],s=4,alpha=.5)
    ax.set(xlabel="PC1",ylabel="PC2",title="Exploratory cardmember segments")
    fig.tight_layout();fig.savefig(out/"segments.png",dpi=150);plt.close(fig)
    return {"explained_variance_ratio":pca.explained_variance_ratio_.tolist(),
            "segments":[{"cluster":int(i),"count":int((group==i).sum()),
                         "mean_reference_score":float(reference[group==i].mean())} for i in range(4)]}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",required=True)
    p.add_argument("--out",default="outputs/research")
    p.add_argument("--clusters",action="store_true")
    args=p.parse_args()
    X=load_matrix(args.data);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    stats=profile(X);models,comparison=ablations(X)
    result={"records":len(X),"feature_profile":stats,"ranker_comparison":comparison,
            "metric_note":"Overlap with the reference ranking, not accuracy against hidden labels."}
    if args.clusters:result["exploratory_segments"]=clusters(X,models["documented_pnl"],out)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,4))
    ax.bar(FEATURES,[stats[k]["missing_pct"] for k in FEATURES])
    ax.set(ylabel="Missing (%)",xlabel="Masked attributes",ylim=(0,100),title="Feature missingness")
    fig.tight_layout();fig.savefig(out/"missingness.png",dpi=150);plt.close(fig)
    (out/"summary.json").write_text(json.dumps(result,indent=2))
    print(f"Analyzed {len(X):,} rows; results in {out}")

if __name__=="__main__":main()
