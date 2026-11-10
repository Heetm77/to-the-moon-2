from common import *
R=pd.read_csv(f"{WORK}/iterations.csv")
P=pd.read_pickle(f"{WORK}/p4.pkl"); daily=P["daily"]
short=["0. Best single\nfeature","1. Spread\nfeatures (ridge)","2. Non-overlapping\ngaps","3. IC-weighted\nfit","4. + order flow\n(rejected)","5. + conditioning","6. + trading\nlayer"]
# ---- improvement loop
labels=["0. Best single feature","1. Spread features (ridge)","2. Non-overlapping gaps","3. IC-weighted fit",
        "4. + order flow (rejected)","5. + conditioning","6. + trading layer"]
fig,(a1,a2)=plt.subplots(1,2,figsize=(13,4.6),sharey=True)
y=np.arange(len(R))[::-1]
cols=["#9AA5B1" if "rejected" in n else ("#1F4E9A" if i in (5,6) else "#6C8FCB") for i,n in enumerate(R.iteration)]
a1.barh(y,R["OOS IC"]*100,color=cols)
for yy,v,t in zip(y,R["OOS IC"]*100,R["t-stat"]): a1.text(v+0.02,yy,f"{v:.2f}%  (t={t:.1f})",va="center",fontsize=8)
a1.set_yticks(y,labels); a1.set_xlabel("Out-of-sample IC with next-1h spread return (%)"); a1.set_xlim(0,2.4)
a1.set_title("Forecast quality: combined signal is ~2x the best single feature")
h=0.27
a2.barh(y+h,R["ann_gross_%"],h,color="#16A085",label="Gross return (before costs)")
a2.barh(y,R["ann_cost_%"]*(2+0.0)/(2+0.0)*0+R["ann_gross_%"]-R["ann_net_maker_%"],h,color="#E8901A",label="Cost with maker fees (2 bps)")
a2.barh(y-h,R["ann_cost_%"],h,color="#C0392B",label="Cost with taker fees (5 bps)")
a2.set_xscale("log"); a2.set_xlabel("% per year per unit of pair exposure (log scale)")
a2.set_title("Trading: smoothing + no-trade band cuts cost ~10x; cost still > gross")
a2.legend(frameon=False,fontsize=8,loc="lower right")
fig.suptitle("Improvement loop — walk-forward out-of-sample, Jun 2023 – Jul 2025, 10 pairs, 15-min bars",fontweight="bold")
fig.savefig(f"{FIG}/09_improvement_loop.png",bbox_inches="tight"); plt.close()
# ---- final slide plot v1
fin="6. + trading layer (smoothing, no-trade band)"
best=[k for k in daily if k.startswith("0.") and "maker" not in k][0]
fig,ax=plt.subplots(figsize=(10,5))
series=[(daily[fin].gross,"Final signal — before costs","#16A085","-",2.2),
        (daily[fin+" [maker]"].net,"Final signal — net, maker fees","#E8901A","-",2.0),
        (daily[fin].net,"Final signal — net, taker fees","#C0392B","-",2.0),
        (daily["Naive rule"].net,"Textbook pairs rule (|z|>2) — net, taker","#7F8C8D","--",1.5)]
for s,l,c,ls,lw in series:
    cs=s.cumsum()*100; ax.plot(cs.index,cs.values,color=c,ls=ls,lw=lw,label=f"{l}: {cs.values[-1]:+.0f}%")
ax.axhline(0,color="k",lw=0.8); ax.set_ylabel("Cumulative return per unit of pair exposure (%)")
ax.legend(frameon=False,fontsize=8.5,loc="lower left")
ax.set_title("FIRST VERSION of the final-slide plot: the spread signal earns before costs, but not after (out-of-sample)",fontsize=10)
fig.savefig(f"{FIG}/10_final_plot_v1.png",bbox_inches="tight"); plt.close()
print(R.round(4).to_string())
for s,l,*_ in series: print(l, round(s.cumsum().iloc[-1]*100,1))
