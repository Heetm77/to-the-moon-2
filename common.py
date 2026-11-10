import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
COINS=["BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT","DOGEUSDT"]
SHORT={c:c.replace("USDT","") for c in COINS}
import os
BASE=os.path.dirname(os.path.abspath(__file__))     # the repo folder, e.g. D:\BLOCKCHAIN PROJECT
DATA=os.path.join(BASE,"data")                      # step2 writes <COIN>_1min.pkl here
WORK=os.path.join(BASE,"work"); FIG=os.path.join(BASE,"figs")
os.makedirs(WORK,exist_ok=True); os.makedirs(FIG,exist_ok=True)
PANEL=os.path.join(WORK,"panel.pkl")
plt.rcParams.update({"figure.dpi":130,"font.size":9,"axes.spines.top":False,"axes.spines.right":False,
                     "axes.titlesize":10,"axes.titleweight":"bold"})
COLORS={"BTC":"#E8901A","ETH":"#5A6FD6","SOL":"#16A085","XRP":"#7F8C8D","DOGE":"#C0392B","INDEX":"#111111"}
def load(field="close_mid"):
    p=pd.read_pickle(PANEL)
    return p.xs(field,axis=1,level=1).rename(columns=SHORT)
def logret_1min():
    mid=load("close_mid").astype("float64")
    mid=mid.ffill(limit=30)          # bridge short gaps; returns across a gap land on the next valid minute
    return np.log(mid).diff()
