import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import step7_model as M, numpy as np, pandas as pd
M.F["y"]=M.F["y1h"]
def wf(spec,train_days,method):
    need=list(set(spec["dir"])|set(spec.get("cond",[])))
    D=M.F.dropna(subset=need+["y","r_exec","beta"])
    refits=pd.date_range(D.time.min()+pd.Timedelta(days=90+31),D.time.max(),freq="7D")
    preds=[]
    for t0,t1 in zip(refits[:-1],refits[1:]):
        tr=D[(D.time<t0-pd.Timedelta("1h15min"))&(D.time>=t0-pd.Timedelta(days=train_days))]
        te=D[(D.time>=t0)&(D.time<t1)]
        Xtr,s=M.design(tr,spec); Xte,_=M.design(te,spec,s); y=tr.y.clip(-5,5).values
        mu=Xtr.mean().values
        if method=="ridge":
            A=Xtr.values-mu; w=np.linalg.solve(A.T@A+1e3*np.eye(A.shape[1]),A.T@(y-y.mean()))
        else:  # IC-weighted: each standardised feature weighted by its training-window correlation with the target
            sd=Xtr.std().values
            w=np.array([np.corrcoef(Xtr.iloc[:,k],y)[0,1] for k in range(Xtr.shape[1])])/sd
        preds.append(pd.Series((Xte.values-mu)@w,index=te.index))
    pr=pd.concat(preds); y=M.F.loc[pr.index,"y"].clip(-5,5).values
    return round(np.corrcoef(pr.values,y)[0,1],4)
S6=["z1h","z4h_1h","z24h_4h","last_bar","imb_diff","volsurp_diff"]
for td in (180,365):
    for spec in [{"dir":["last_bar"]},{"dir":["z1h","z4h_1h","z24h_4h","last_bar"]},{"dir":S6},
                 {"dir":S6,"cond":M.COND,"interact":["z1h","last_bar"]}]:
        print(td,spec["dir"][-1] if len(spec["dir"])==1 else len(spec["dir"]),"cond" if "cond" in spec else "", "ridge",wf(spec,td,"ridge"),"icw",wf(spec,td,"icw"))
