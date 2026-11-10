"""Step 3: combine the 5 per-coin files from step 2 into one table (work/panel.pkl)."""
from common import *
cols=["close_mid","dollar_volume","buy_volume","sell_volume","volume","spread_bps","carryover"]
panel=pd.concat({c:pd.read_pickle(os.path.join(DATA,f"{c}_1min.pkl"))[cols] for c in COINS},axis=1)
panel.to_pickle(PANEL)
print("Saved",PANEL,panel.shape,panel.index.min(),"->",panel.index.max())
