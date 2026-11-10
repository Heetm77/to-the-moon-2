from common import *
r=logret_1min()
names=list(r.columns)
# ---------- 1. EMA-horizon correlation matrices ----------
horizons=[1,5,15,60,240,1440]
mats={}
for h in horizons:
    x=r if h==1 else r.ewm(halflife=h,min_periods=h).mean()
    mats[h]=x.corr()
fig,axes=plt.subplots(2,3,figsize=(11,7))
for ax,h in zip(axes.ravel(),horizons):
    m=mats[h].values
    im=ax.imshow(m,vmin=0.3,vmax=1,cmap="Blues")
    for i in range(5):
        for j in range(5):
            ax.text(j,i,f"{m[i,j]:.2f}",ha="center",va="center",fontsize=8,color="white" if m[i,j]>0.75 else "black")
    ax.set_xticks(range(5),names); ax.set_yticks(range(5),names)
    ax.set_title("Raw 1-min returns" if h==1 else f"EMA half-life {h} min" if h<1440 else "EMA half-life 1 day")
fig.colorbar(im,ax=axes,shrink=0.6,label="Correlation")
fig.suptitle("Correlation between the 5 coins is similar at every horizon (avg 0.64–0.66); BTC–ETH is the tightest pair",fontweight="bold")
fig.savefig(f"{FIG}/01_corr_by_ema_horizon.png",bbox_inches="tight"); plt.close()
offd=lambda m:m.values[np.triu_indices(5,1)].mean()
print("avg off-diagonal corr by horizon:",{h:round(offd(mats[h]),3) for h in horizons})

# ---------- 2. daily matrices for EMA 60, averaged ----------
H=60
x=r.ewm(halflife=H,min_periods=H).mean()
day=x.index.floor("D")
daily=x.groupby(day).corr()                      # (day, coin) x coin
daily=daily.dropna(how="all")
avg=daily.groupby(level=1).mean().loc[names,names]
# avg pairwise over time
iu=np.triu_indices(5,1)
ts=daily.groupby(level=0).apply(lambda d:d.loc[(d.index[0][0],names),names].values[iu].mean() if len(d)==5 else np.nan)
ts.index=pd.to_datetime(ts.index)
fig,(a1,a2)=plt.subplots(1,2,figsize=(12,4.3),gridspec_kw={"width_ratios":[1,2]})
m=avg.values; im=a1.imshow(m,vmin=0.3,vmax=1,cmap="Blues")
for i in range(5):
    for j in range(5): a1.text(j,i,f"{m[i,j]:.2f}",ha="center",va="center",fontsize=8,color="white" if m[i,j]>0.75 else "black")
a1.set_xticks(range(5),names); a1.set_yticks(range(5),names); a1.set_title(f"Average of daily matrices (EMA {H} min)")
a2.plot(ts.index,ts.values,color="#BBBBBB",lw=0.6,label="Daily")
a2.plot(ts.index,ts.rolling(30,min_periods=10).mean(),color="#1F4E9A",lw=1.6,label="30-day average")
a2.set_ylabel("Average pairwise correlation"); a2.set_title("How tightly the 5 coins move together, day by day"); a2.legend(frameon=False)
fig.savefig(f"{FIG}/02_daily_corr_avg.png",bbox_inches="tight"); plt.close()
avg.to_csv(f"{WORK}/avg_daily_corr.csv")
print("long-run avg pairwise:",round(m[iu].mean(),3), "daily ts min/max:",round(ts.min(),3),round(ts.max(),3))

# ---------- 3. high-vol day ----------
rv=(r["BTC"]**2).groupby(day).sum()**0.5*100
top=rv.sort_values(ascending=False).head(5); print("top BTC vol days (%):\n",top.round(2))
hd=top.index[0]
hm=daily.loc[hd].loc[names,names]
fig,axes=plt.subplots(1,3,figsize=(13,4))
for ax,mm,t,cm,lo,hi in [(axes[0],avg,"Long-run average",  "Blues",0.3,1),
                         (axes[1],hm,f"High-volatility day: {hd:%d %b %Y}","Blues",0.3,1),
                         (axes[2],hm-avg,"Difference (day − average)","RdBu_r",-0.5,0.5)]:
    v=mm.values; im=ax.imshow(v,vmin=lo,vmax=hi,cmap=cm)
    for i in range(5):
        for j in range(5): ax.text(j,i,f"{v[i,j]:+.2f}" if t.startswith("Diff") else f"{v[i,j]:.2f}",ha="center",va="center",fontsize=8,color="white" if (not t.startswith("Diff") and v[i,j]>0.75) else "black")
    ax.set_xticks(range(5),names); ax.set_yticks(range(5),names); ax.set_title(t)
fig.suptitle(f"On the most volatile day (BTC daily vol {top.iloc[0]:.1f}%), the coins move almost as one",fontweight="bold")
fig.savefig(f"{FIG}/03_high_vol_day.png",bbox_inches="tight"); plt.close()
print("high-vol day avg pairwise:",round(hm.values[iu].mean(),3))

# ---------- 4. volume-weighted index ----------
dv=load("dollar_volume").astype("float64")
w=dv.groupby(day).sum().rolling(30,min_periods=5).mean().shift(1)      # trailing 30-day $ volume, known before the day starts
w=w.div(w.sum(axis=1),axis=0)
wmin=w.reindex(r.index.floor("D")).set_axis(r.index)
idx=(r.fillna(0)*wmin).sum(axis=1,min_count=1)
idx[wmin.isna().any(axis=1)]=np.nan
print("avg index weights:",w.mean().round(3).to_dict())
r_all=r.copy(); r_all["INDEX"]=idx
h1=r_all.resample("h").sum(min_count=30)*100
fig,ax=plt.subplots(figsize=(9,5))
bins=np.linspace(-6,6,121)
for c in ["INDEX","BTC","ETH","SOL","XRP","DOGE"]:
    s=h1[c].dropna()
    ax.hist(s,bins=bins,density=True,histtype="step",lw=2.4 if c=="INDEX" else 1.3,color=COLORS[c],
            label=f"{c}  (std {s.std():.2f}%, kurt {s.kurt():.0f})")
ax.set_yscale("log"); ax.set_xlabel("1-hour return (%)  — shown between −6% and +6%; std and kurtosis use all hours"); ax.set_ylabel("Density (log scale)")
ax.set_title("1-hour return distributions: volume-weighted index vs each coin"); ax.legend(frameon=False)
fig.savefig(f"{FIG}/04_index_return_hist.png",bbox_inches="tight"); plt.close()
print(h1.describe().T[["std","min","max"]].round(3)); print("kurt",h1.kurt().round(1).to_dict())
pd.to_pickle({"rv":rv,"ts":ts,"top":top},f"{WORK}/p1.pkl")
