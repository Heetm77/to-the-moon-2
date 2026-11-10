"""Cointegration scan + hourly pair features (all trailing / past-only)."""
from common import *
import matplotlib.ticker
import itertools, warnings
from statsmodels.tsa.stattools import adfuller, coint
warnings.filterwarnings("ignore")

p=pd.read_pickle(PANEL)
def fld(f): return p.xs(f,axis=1,level=1).rename(columns=SHORT).astype("float64")
mid=fld("close_mid").ffill(limit=30)
H=pd.DataFrame({c:mid[c].resample("h").last() for c in mid})          # hourly close (bar labelled by its start; value = last mid in that hour)
H.index=H.index+pd.Timedelta("1h")                                     # label by bar END so time t means "known at t"
L=np.log(H)
buy=fld("buy_volume").resample("h").sum(); sell=fld("sell_volume").resample("h").sum(); dv=fld("dollar_volume").resample("h").sum()
for d in (buy,sell,dv): d.index=d.index+pd.Timedelta("1h")
names=list(L.columns)
pairs=list(itertools.combinations(names,2))
W=720  # 30 days of hours

# ---------- full-sample Engle-Granger + rolling ADF strength ----------
rows=[]
for a,b in pairs:
    x=L[[a,b]].dropna()
    t,pv,_=coint(x[a],x[b])
    rows.append((f"{a}-{b}",round(t,2),round(pv,3)))
eg=pd.DataFrame(rows,columns=["pair","EG t-stat","p-value"]); print(eg.to_string(index=False))
eg.to_csv(f"{WORK}/eg_fullsample.csv",index=False)

def rolling_adf(a,b,step=24,W=W):
    x=L[[a,b]].dropna()
    out={}
    for end in range(W,len(x),step):
        w=x.iloc[end-W:end]
        bb=np.polyfit(w[b],w[a],1)
        res=w[a]-np.polyval(bb,w[b])
        out[x.index[end-1]]=adfuller(res.values,maxlag=1,autolag=None)[0]
    return pd.Series(out)
adf={f"{a}-{b}":rolling_adf(a,b) for a,b in pairs}
adf=pd.DataFrame(adf)
adf90=pd.DataFrame({f"{a}-{b}":rolling_adf(a,b,step=24*7,W=2160) for a,b in pairs})
print("share of 90-day windows cointegrated at ~5%:\n",(adf90<-3.37).mean().round(3).to_dict())
crit=-3.37   # ~5% Engle-Granger critical value, 2 variables
frac=(adf<crit).mean().sort_values(ascending=False)
print("share of 30-day windows cointegrated at ~5%:\n",frac.round(3))

# plot: when is each pair cointegrated?
order=list(frac.index)
wk=(adf<crit).resample("W").mean()[order]
fig,(a1,a2)=plt.subplots(1,2,figsize=(13,4.2),gridspec_kw={"width_ratios":[3,1]})
im=a1.imshow(wk.T.values,aspect="auto",cmap="Greens",vmin=0,vmax=1,interpolation="nearest",
             extent=[mdates_lo:=0,len(wk),len(order)-0.5,-0.5])
ticks=np.arange(0,len(wk),13); a1.set_xticks(ticks,[d.strftime("%b %y") for d in wk.index[ticks]],rotation=0)
a1.set_yticks(range(len(order)),order); a1.set_title("Weeks where the trailing 30-day window passes a cointegration test (green)")
a2.barh(range(len(order)),frac.values,color="#2E8B57"); a2.set_yticks(range(len(order)),order); a2.invert_yaxis()
a2.set_xlabel("Share of windows cointegrated"); a2.set_title("Overall")
for i,v in enumerate(frac.values): a2.text(v+0.001,i,f"{v:.1%}",va="center",fontsize=8)
a2.axvline(0.05,color="#C0392B",ls="--",lw=1); a2.text(0.0505,len(order)-0.6,"5% = expected\nby pure chance",color="#C0392B",fontsize=7.5,va="bottom")
a2.set_xlim(0,0.065); a2.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1,decimals=0))
fig.suptitle("Price-level cointegration is rare: only 1.5–3% of 30-day windows pass a 5% test, below the 5% expected by chance",fontweight="bold")
fig.savefig(f"{FIG}/05_cointegration_over_time.png",bbox_inches="tight"); plt.close()

# ---------- 15-minute pair features ----------
# Exploration (explore.py) showed spread reversion lives at 1-4h, not 24h, so features run on 15-min bars.
BAR="15min"; NH=4                                   # bars per hour
M=mid.resample(BAR).last(); M.index=M.index+pd.Timedelta(BAR)
Lq=np.log(M); rq=Lq.diff()
bq=fld("buy_volume").resample(BAR).sum(); sq=fld("sell_volume").resample(BAR).sum(); dq=fld("dollar_volume").resample(BAR).sum().replace(0,np.nan)
for d in (bq,sq,dq): d.index=d.index+pd.Timedelta(BAR)
imb=((bq-sq)/(bq+sq)).ewm(halflife=NH,min_periods=2).mean()                         # 1h EMA of taker buy/sell imbalance
volsurp=np.log(dq.rolling(NH,min_periods=2).sum()/dq.rolling(720*NH,min_periods=240*NH).mean()/NH)   # last-hour $ volume vs 30-day norm
mktvol=np.log((rq["BTC"]**2).rolling(24*NH,min_periods=12*NH).mean()**0.5/((rq["BTC"]**2).rolling(720*NH,min_periods=240*NH).mean()**0.5))
Wq=720*NH
feats=[]
for a,b in pairs:
    pid=f"{a}-{b}"
    beta=(rq[a].rolling(Wq,min_periods=Wq//2).cov(rq[b])/rq[b].rolling(Wq,min_periods=Wq//2).var()).clip(0,3)
    rp=rq[a]-beta.shift(1)*rq[b]                      # 15-min hedged (pair) return, hedge set one bar earlier
    sig=rp.rolling(Wq,min_periods=Wq//2).std()        # trailing 30-day pair vol per bar
    f=pd.DataFrame(index=Lq.index)
    for hh in (1,4,24):                               # spread gap vs its EMA, half-life 1h / 4h / 24h, in units of bar vol
        h=hh*NH
        dA=Lq[a]-Lq[a].ewm(halflife=h,min_periods=h).mean(); dB=Lq[b]-Lq[b].ewm(halflife=h,min_periods=h).mean()
        f[f"z{hh}h"]=(dA-beta*dB)/(sig*np.sqrt(h))
    f["last_bar"]=rp/sig                              # most recent 15-min spread move
    f["imb_diff"]=imb[a]-imb[b]
    f["volsurp_diff"]=volsurp[a]-volsurp[b]
    f["volratio"]=np.log(rp.rolling(24*NH,min_periods=12*NH).std()/sig)   # spread vol today vs 30-day
    f["corr7d"]=rq[a].rolling(168*NH,min_periods=84*NH).corr(rq[b])
    f["adf"]=adf[pid].dropna().reindex(Lq.index,method="ffill",limit=48*NH)          # cointegration strength (more negative = stronger)
    f["mktvol"]=mktvol
    f["beta"]=beta; f["sig"]=sig
    # target: pair return over the next 4h, starting ONE bar after the signal (execution delay), in units of 4h pair vol
    for hh in (1,4):
        n=hh*NH
        f[f"y{hh}h"]=((Lq[a].shift(-n-1)-Lq[a].shift(-1))-beta*(Lq[b].shift(-n-1)-Lq[b].shift(-1)))/(sig*np.sqrt(n))
    f["y"]=f["y4h"]
    f["r_exec"]=rq[a].shift(-2)-beta*rq[b].shift(-2)  # return of a position decided at t, filled at t+1 bar, held for the bar after
    f["pair"]=pid
    feats.append(f)
F=pd.concat(feats).rename_axis("time").reset_index()
F.to_pickle(f"{WORK}/features.pkl")
print(F.shape); print(F.describe().T[["count","mean","std","min","max"]].round(3))
