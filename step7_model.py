"""Walk-forward ridge, baseline comparison, improvement loop, cost-staged backtest.
Development period only: Jan 2023 – Jul 2025. Holdout Aug–Nov 2025 is NOT touched."""
from common import *
from scipy import stats
F=pd.read_pickle(f"{WORK}/features.pkl")
HOLDOUT=pd.Timestamp("2025-08-01",tz="UTC")
F=F[F.time<HOLDOUT].sort_values(["pair","time"]).reset_index(drop=True)
F["z4h_1h"]=F.z4h-F.z1h; F["z24h_4h"]=F.z24h-F.z4h
COND=["corr7d","volratio","adf","mktvol"]
TARGET="y1h"; GAP=pd.Timedelta("1h15min")   # 1h horizon + 1 bar execution delay
F["y"]=F[TARGET]
TRAIN_DAYS=365; REFIT="7D"; ALPHA=1e3        # rolling 1-year training window, refit weekly (90/180 days tested: noisier)
HALF_SPREAD_BPS={"BTC":0.01,"ETH":0.02,"SOL":0.23,"XRP":0.83,"DOGE":0.40}   # from step2 output (median spread / 2)
FEE_BPS=5.0                                                                  # Binance base taker, per side per leg

def design(D,spec,stats_=None):
    """Build X. Direction features: scaled (not centred) so the model stays sign-symmetric across pair order.
       Conditioning features only enter as interactions with direction features."""
    if stats_ is None:
        stats_={c:(D[c].mean(),D[c].std()) for c in set(spec["dir"])|set(spec.get("cond",[]))}
    X={}
    for c in spec["dir"]:
        X[c]=(D[c]/stats_[c][1]).clip(-5,5)
    for c in spec.get("cond",[]):
        cz=((D[c]-stats_[c][0])/stats_[c][1]).clip(-3,3)
        for d in spec.get("interact",[]):
            X[f"{d}*{c}"]=X[d]*cz
    return pd.DataFrame(X,index=D.index),stats_

def walk_forward(spec,alpha=ALPHA,train_days=TRAIN_DAYS,method="ridge"):
    need=list(set(spec["dir"])|set(spec.get("cond",[])))
    D=F.dropna(subset=need+["y","r_exec","beta"])
    refits=pd.date_range(pd.Timestamp("2023-06-01",tz="UTC"),D.time.max(),freq=REFIT)   # same OOS start for every model (first fits use the data available so far)
    preds=[];coefs=[]
    for t0,t1 in zip(refits[:-1],refits[1:]):
        tr=D[(D.time<t0-GAP)&(D.time>=t0-pd.Timedelta(days=train_days))]   # gap so every training target is fully known before t0 (no look-ahead)
        te=D[(D.time>=t0)&(D.time<t1)]
        if len(te)==0: continue
        Xtr,s=design(tr,spec); Xte,_=design(te,spec,s)
        ytr=tr.y.clip(-5,5).values
        mu=Xtr.mean().values; A=Xtr.values-mu; yc=ytr-ytr.mean()      # centre: the intercept (pair drift = beta, not signal) is fitted and then DROPPED
        if method=="ridge":
            w=np.linalg.solve(A.T@A+alpha*np.eye(A.shape[1]),A.T@yc)
        else:   # "icw": IC-weighted combination: weight_k = corr(feature_k, target) in the training window / std_k
            w=np.array([np.corrcoef(A[:,k],yc)[0,1] for k in range(A.shape[1])])/A.std(0)
        preds.append(pd.Series((Xte.values-mu)@w,index=te.index)); coefs.append(pd.Series(w,index=Xtr.columns,name=t0))
    pred=pd.concat(preds)
    return pred,pd.DataFrame(coefs)

def ic_report(pred):
    """Primary fit metric: Pearson IC between forecast and the (clipped) vol-scaled 4h target."""
    D=F.loc[pred.index]; p=pred.values; y=D.y.clip(-5,5).values
    ic=np.corrcoef(p,y)[0,1]
    # non-overlapping sample for an honest t-stat
    m=(D.time.dt.minute==0)            # one sample per hour = non-overlapping for a 1h target
    ic_no=np.corrcoef(p[m.values],y[m.values])[0,1]; n=m.sum()
    return ic,ic_no,ic_no*np.sqrt(n-2)/np.sqrt(1-ic_no**2),n

def backtest(pred,smooth=0,band=0.0,delay=True,costs=True,fee=FEE_BPS):
    """Per pair: position = prediction scaled by its trailing 30-day std, capped at ±1 unit
       (1 unit = $1 long/short coin A against $beta of coin B)."""
    out=[]
    for pid,idx in F.loc[pred.index].groupby("pair").groups.items():
        D=F.loc[idx].copy(); D["pred"]=pred.loc[idx]
        p=D.pred.ewm(halflife=smooth).mean() if smooth else D.pred
        scale=p.abs().rolling(30*96,min_periods=96).mean().shift(1)
        tgt=(p/(2*scale)).clip(-1,1).fillna(0).values
        pos=np.zeros(len(tgt)); cur=0.0
        for i,x in enumerate(tgt):
            if abs(x-cur)>band: cur=x
            pos[i]=cur
        a,b=pid.split("-")
        cost_unit=(fee+HALF_SPREAD_BPS[a]+D.beta.values*(fee+HALF_SPREAD_BPS[b]))/1e4
        ret=D.r_exec.values if delay else D.groupby("pair").r_exec.shift(1).values
        gross=pos*np.nan_to_num(ret)
        tc=np.abs(np.diff(pos,prepend=0))*cost_unit if costs else 0*pos
        out.append(pd.DataFrame({"time":D.time.values,"pair":pid,"gross":gross,"cost":tc,"net":gross-tc,"turn":np.abs(np.diff(pos,prepend=0))}))
    B=pd.concat(out)
    daily=B.groupby(pd.to_datetime(B.time).dt.floor("D"))[["gross","cost","net","turn"]].sum()/10   # equal weight across 10 pairs
    sh=lambda s: s.mean()/s.std()*np.sqrt(365) if s.std()>0 else np.nan
    return {"sharpe_gross":sh(daily.gross),"sharpe_net":sh(daily.net),"ann_net_%":daily.net.mean()*365*100,
            "ann_gross_%":daily.gross.mean()*365*100,"ann_cost_%":daily.cost.mean()*365*100,"daily_turnover":daily.turn.mean()},daily

def naive_rule(entry=2.0):
    """Textbook pairs rule (a rule, not a forecast): short the spread when z24h>2, long when < -2, exit at 0."""
    s=pd.Series(0.0,index=F.index)
    for pid,idx in F.groupby("pair").groups.items():
        z=F.loc[idx,"z24h"].values; pos=np.zeros(len(z)); cur=0
        for i,v in enumerate(z):
            if np.isnan(v): pos[i]=cur; continue
            if cur==0 and v>entry: cur=-1
            elif cur==0 and v<-entry: cur=1
            elif cur==-1 and v<=0: cur=0
            elif cur==1 and v>=0: cur=0
            pos[i]=cur
        s.loc[idx]=pos
    return s

if __name__=="__main__":
    results=[];daily_store={}
    DIR6=["z1h","z4h_1h","z24h_4h","last_bar","imb_diff","volsurp_diff"]
    singles={}
    for f in ["z1h","z4h","z24h","last_bar","imb_diff","volsurp_diff"]:
        pr,_=walk_forward({"dir":[f]})
        singles[f]=(pr,)+ic_report(pr)
        print(f"single {f:13s} OOS IC {singles[f][1]:+.4f}  non-overlap IC {singles[f][2]:+.4f} t={singles[f][3]:.1f}")
    best=max(singles,key=lambda k:singles[k][1]); print("best standalone feature:",best)
    ITER=[("0. Best single feature ("+best+")",{"dir":[best]},"ridge",{}),
          ("1. Spread features, ridge",{"dir":["z1h","z4h","z24h","last_bar"]},"ridge",{}),
          ("2. Non-overlapping spread gaps",{"dir":["z1h","z4h_1h","z24h_4h","last_bar"]},"ridge",{}),
          ("3. IC-weighted fit instead of ridge",{"dir":["z1h","z4h_1h","z24h_4h","last_bar"]},"icw",{}),
          ("4. + order-flow features (rejected: IC fell)",{"dir":DIR6},"icw",{}),
          ("5. Step 3 + conditioning (corr, vol, coint, mkt vol)",{"dir":["z1h","z4h_1h","z24h_4h","last_bar"],"cond":COND,"interact":["z1h","last_bar"]},"icw",{}),
          ("6. + trading layer (smoothing, no-trade band)",None,None,{"smooth":4,"band":0.5})]
    last_pred=None;coefs_final=None;preds_store={}
    for name,spec,meth,trade in ITER:
        if spec is not None:
            pred,coefs=walk_forward(spec,method=meth)
            if "rejected" not in name: last_pred,coefs_final=pred,coefs
        else:
            pred=last_pred
        preds_store[name]=pred
        ic,ic_no,t,n=ic_report(pred)
        bt,daily=backtest(pred,**trade); btm,dailym=backtest(pred,fee=2.0,**trade)
        daily_store[name]=daily; daily_store[name+" [maker]"]=dailym
        results.append({"iteration":name,"OOS IC":ic,"non-overlap IC":ic_no,"t-stat":t,**bt,"sharpe_net_maker":btm["sharpe_net"],"ann_net_maker_%":btm["ann_net_%"]})
        print(name,{k:round(v,3) for k,v in results[-1].items() if k!="iteration"})
    R=pd.DataFrame(results); R.to_csv(f"{WORK}/iterations.csv",index=False)
    # naive textbook rule
    tmp=naive_rule()
    out=[]
    for pid,idx in F.groupby("pair").groups.items():
        D=F.loc[idx]; a,b=pid.split("-")
        cu=(FEE_BPS+HALF_SPREAD_BPS[a]+D.beta.values*(FEE_BPS+HALF_SPREAD_BPS[b]))/1e4
        p=tmp.loc[idx].values; tr=np.abs(np.diff(p,prepend=0))
        out.append(pd.DataFrame({"time":D.time.values,"gross":p*np.nan_to_num(D.r_exec.values),"cost":tr*np.nan_to_num(cu),"turn":tr}))
    NB=pd.concat(out); NB["net"]=NB.gross-NB.cost
    nd=NB.groupby(pd.to_datetime(NB.time).dt.floor("D"))[["gross","cost","net","turn"]].sum()/10
    first=daily_store[ITER[0][0]].index.min(); nd=nd[nd.index>=first]
    daily_store["Naive rule"]=nd
    print("naive rule: sharpe net",round(nd.net.mean()/nd.net.std()*np.sqrt(365),2),"ann net %",round(nd.net.mean()*365*100,2),"ann gross %",round(nd.gross.mean()*365*100,2),"turnover",round(nd.turn.mean(),3))
    stages=[]
    for label,kw in [("1. Mid-to-mid, no delay, no costs",{"delay":False,"costs":False}),
                     ("2. + one-bar execution delay",{"delay":True,"costs":False}),
                     ("3. + maker fees (2 bps) and half-spread",{"delay":True,"costs":True,"fee":2.0}),
                     ("4. + taker fees (5 bps) and half-spread",{"delay":True,"costs":True})]:
        bt,_=backtest(last_pred,smooth=4,band=0.5,**kw); stages.append({"stage":label,**bt})
    S=pd.DataFrame(stages); print(S.round(3).to_string()); S.to_csv(f"{WORK}/cost_stages.csv",index=False)
    coefs_final.to_csv(f"{WORK}/coefs_final.csv")
    pd.to_pickle({"daily":daily_store,"singles":{k:v[1:] for k,v in singles.items()},"best":best,"preds":preds_store},f"{WORK}/p4.pkl")
