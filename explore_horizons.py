import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *
import itertools
from scipy import stats
p=pd.read_pickle(PANEL)
mid=p.xs("close_mid",axis=1,level=1).rename(columns=SHORT).astype("float64").ffill(limit=30)
names=list(mid.columns)
for bar in ("15min","1h"):
    B=mid.resample(bar).last(); B.index=B.index+pd.Timedelta(bar); L=np.log(B); r=L.diff()
    n_per_h=4 if bar=="15min" else 1
    W=720*n_per_h
    res=[]
    for a,b in itertools.combinations(names,2):
        beta=(r[a].rolling(W,min_periods=W//2).cov(r[b])/r[b].rolling(W,min_periods=W//2).var()).clip(0,3)
        rp=r[a]-beta.shift(1)*r[b]; sig=rp.rolling(W,min_periods=W//2).std()
        d=pd.DataFrame(index=L.index)
        for hh in (1,2,4,8,24):
            h=hh*n_per_h
            dA=L[a]-L[a].ewm(halflife=h).mean(); dB=L[b]-L[b].ewm(halflife=h).mean()
            d[f"z{hh}h"]=(dA-beta*dB)/sig
        for th in (1,4,24):
            n=th*n_per_h
            # skip one bar before the target starts (one-bar delay)
            d[f"y{th}h"]=((L[a].shift(-n-1)-L[a].shift(-1))-beta*(L[b].shift(-n-1)-L[b].shift(-1)))/sig
        res.append(d[d.index<"2025-08-01"])
    D=pd.concat(res).dropna()
    print(bar, len(D))
    print(D.corr(method="spearman").loc[[c for c in D if c.startswith("z")],[c for c in D if c.startswith("y")]].round(4))
