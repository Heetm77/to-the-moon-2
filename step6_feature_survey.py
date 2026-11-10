"""Feature quality + collinearity survey on the development period only (holdout Aug–Nov 2025 untouched)."""
from common import *
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
F=pd.read_pickle(f"{WORK}/features.pkl")
HOLDOUT=pd.Timestamp("2025-08-01",tz="UTC")
F=F[F.time<HOLDOUT].copy(); F["y"]=F["y1h"]          # same target as the model: next-1h spread return
SYM=["z1h","z4h","z24h","last_bar","imb_diff","volsurp_diff"]       # flip sign if the pair is flipped -> can predict direction
COND=["volratio","corr7d","adf","mktvol"]                             # same for both orderings -> used only to scale the others
FEATS=SYM+COND
LABEL={"z1h":"Spread gap vs 1h average","z4h":"Spread gap vs 4h average","z24h":"Spread gap vs 24h average",
       "last_bar":"Last 15-min spread move","imb_diff":"Buy/sell imbalance gap (A−B)","volsurp_diff":"Volume surprise gap (A−B)",
       "volratio":"Spread vol: 1 day vs 30 days","corr7d":"7-day return correlation","adf":"Cointegration strength (ADF t)","mktvol":"BTC vol: 1 day vs 30 days"}
D=F.dropna(subset=FEATS+["y"])
yrs=D.time.dt.year
st=pd.DataFrame({"missing %":F[FEATS].isna().mean()*100,"mean":D[FEATS].mean(),"std":D[FEATS].std(),
                 "skew":D[FEATS].skew(),"kurtosis":D[FEATS].kurt(),
                 "IC (next 1h)":[stats.spearmanr(D[f],D.y)[0] for f in FEATS]})
for y in sorted(yrs.unique()):
    st[f"IC {y}"]=[stats.spearmanr(D.loc[yrs==y,f],D.loc[yrs==y,"y"])[0] for f in FEATS]
st["type"]=["direction"]*len(SYM)+["conditioning"]*len(COND)
print(st.round(3).to_string()); st.to_csv(f"{WORK}/feature_stats.csv")
print("rows:",len(D),"pairs:",D.pair.nunique(),D.time.min(),D.time.max())
# ---- distributions
fig,axes=plt.subplots(2,5,figsize=(15,6))
for ax,f in zip(axes.ravel(),FEATS):
    s=D[f]; lo,hi=s.quantile([0.001,0.999])
    ax.hist(s.clip(lo,hi),bins=80,color="#4C72B0" if f in SYM else "#8E6BB0",alpha=0.85)
    ax.set_title(LABEL[f],fontsize=8.5); ax.set_xlabel(f,fontsize=7.5)
    ax.text(0.02,0.96,f"mean {s.mean():.2f}\nstd {s.std():.2f}\nskew {s.skew():.2f}\nkurt {s.kurt():.1f}",transform=ax.transAxes,va="top",fontsize=7)
    ax.set_yticks([])
fig.suptitle("Feature distributions — 10 pairs pooled, 15-min bars, Jan 2023 – Jul 2025 (blue = direction features, purple = conditioning; 0.1% tails trimmed for display)",fontweight="bold",fontsize=10)
fig.tight_layout(); fig.savefig(f"{FIG}/06_feature_distributions.png",bbox_inches="tight"); plt.close()
# ---- cross-correlation incl. target
cols=FEATS+["y"]
C=D[cols].sample(300000,random_state=0).corr(method="spearman")
fig,ax=plt.subplots(figsize=(8,6.8))
im=ax.imshow(C.values,cmap="RdBu_r",vmin=-1,vmax=1)
for i in range(len(cols)):
    for j in range(len(cols)):
        ax.text(j,i,f"{C.values[i,j]:.2f}",ha="center",va="center",fontsize=7,color="white" if abs(C.values[i,j])>0.6 else "black")
ax.set_xticks(range(len(cols)),cols,rotation=45,ha="right"); ax.set_yticks(range(len(cols)),cols)
ax.axhline(len(FEATS)-0.5,color="k",lw=1); ax.axvline(len(FEATS)-0.5,color="k",lw=1)
fig.colorbar(im,shrink=0.7,label="Spearman correlation")
ax.set_title("Feature cross-correlation (last row/col = target: next-1h spread return)")
fig.savefig(f"{FIG}/07_feature_crosscorr.png",bbox_inches="tight"); plt.close()
# ---- VIF, before and after making the z features non-overlapping
def vif(X):
    X=((X-X.mean())/X.std()).assign(const=1)
    return pd.Series([variance_inflation_factor(X.values,i) for i in range(X.shape[1]-1)],index=X.columns[:-1])
S=D.sample(300000,random_state=1)
v1=vif(S[FEATS])
S2=S.assign(z4h_1h=S.z4h-S.z1h,z24h_4h=S.z24h-S.z4h)
NONOV=["z1h","z4h_1h","z24h_4h","last_bar","imb_diff","volsurp_diff"]+COND
v2=vif(S2[NONOV])
print("VIF original:\n",v1.round(2)); print("VIF non-overlapping:\n",v2.round(2))
print("corr among non-overlapping z:\n",S2[["z1h","z4h_1h","z24h_4h","last_bar"]].corr().round(2))
pd.concat({"VIF (original)":v1,"VIF (non-overlapping)":v2},axis=1).to_csv(f"{WORK}/vif.csv")
# ---- IC by year chart
fig,ax=plt.subplots(figsize=(9,4))
xs=np.arange(len(FEATS)); yl=sorted(yrs.unique()); w=0.8/len(yl)
for k,(y,c) in enumerate(zip(yl,["#9DB4D9","#3D63A8","#1B2F5A"])):
    ax.bar(xs+(k-(len(yl)-1)/2)*w,st[f"IC {y}"],w,color=c,label=str(y) if y<2025 else "2025 (to Jul)")
ax.axhline(0,color="k",lw=0.8); ax.set_xticks(xs,FEATS,rotation=30,ha="right")
ax.set_ylabel("Rank correlation with next-1h spread return")
ax.legend(frameon=False,ncol=3); ax.set_title("Single-feature power by year: all four spread features revert (negative IC) in every year, weaker in 2025")
fig.savefig(f"{FIG}/08_feature_ic_by_year.png",bbox_inches="tight"); plt.close()
