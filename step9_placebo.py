"""Placebo: same final forecasts, but each week's sign is randomly flipped per pair -> same activity, no information.
If the real gross Sharpe is far above the placebo range, the edge is not an artefact of the backtest."""
import step7_model as M, numpy as np, pandas as pd
from common import WORK
P=pd.read_pickle(f"{WORK}/p4.pkl"); pred=P["preds"]["6. + trading layer (smoothing, no-trade band)"]
D=M.F.loc[pred.index,["time","pair"]]
wk=D.time.dt.to_period("W").astype(str)+D.pair
rng=np.random.default_rng(0); out=[]
for k in range(20):
    flips=pd.Series(rng.choice([-1,1],size=wk.nunique()),index=wk.unique())
    bt,_=M.backtest(pred*wk.map(flips).values,smooth=4,band=0.5)
    out.append(bt["sharpe_gross"])
real,_=M.backtest(pred,smooth=4,band=0.5)
print("real gross Sharpe",round(real["sharpe_gross"],2),"placebo mean",round(np.mean(out),2),"placebo sd",round(np.std(out),2),"max",round(max(out),2))
pd.Series(out).to_csv(f"{WORK}/placebo.csv",index=False)
