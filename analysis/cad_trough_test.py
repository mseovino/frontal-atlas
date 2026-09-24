"""The STNRY+TROF combination was cold-season while STNRY alone was not.
That points at the trough code carrying the wedge. Test it directly.
"""
import sys
import numpy as np, pandas as pd, pyarrow.dataset as pads
from common import CENTERS, MONTHS, read_fronts

YEARS = (2007, 2018)
COLD_M, WARM_M = [10,11,12,1,2,3,4], [5,6,7,8,9]
crest = lambda lat: -84.0 + (lat - 34.5) * (7.0/6.0)

raw = read_fronts(["STNRY", "TROF"])
t = pd.to_datetime(raw.valid_time, utc=True)
raw = raw.assign(atime=t, month=t.dt.month, year=t.dt.year)
raw = raw[(raw.year>=YEARS[0]) & (raw.year<=YEARS[1])].reset_index(drop=True)

def geom(lat_e2, lon_e2, east_max=2.5, lat_hi=38.5, min_km=300.0):
    lat = np.asarray(lat_e2,dtype=float)/100.0; lon = np.asarray(lon_e2,dtype=float)/100.0
    off = lon - crest(lat)
    m = (lat>=33.0)&(lat<=lat_hi)&(off>=-0.5)&(off<=east_max)
    if m.sum()<2: return False, np.nan
    la, lo = lat[m], lon[m]
    dy=(la[-1]-la[0])*111.32; dx=(lo[-1]-lo[0])*111.32*np.cos(np.radians(la.mean()))
    if np.hypot(dx,dy)<min_km: return False, np.nan
    return True, np.degrees(np.arctan2(dx,dy))%180.0

ok, bear = zip(*[geom(a,b) for a,b in zip(raw.lat_e2, raw.lon_e2)])
ok=np.array(ok); delta=np.abs((np.array(bear)-43.0+90)%180-90)

cds = pads.dataset(CENTERS, format="parquet", partitioning="hive")
hi = cds.to_table(columns=["valid_time","res","kind","lat","lon","pressure_hpa"],
                  filter=(pads.field("kind")=="H")).to_pandas()
hi = hi[hi.res=="HR"].copy(); hi["atime"]=pd.to_datetime(hi.valid_time, utc=True)
hi = hi[(hi.atime.dt.year>=YEARS[0])&(hi.atime.dt.year<=YEARS[1])]
shared = set(raw.atime) & set(hi.atime)
denom = pd.Series(sorted(shared)).dt.month.value_counts().reindex(range(1,13), fill_value=0)

def ratio(times,label,show=False):
    times=[x for x in times if x in shared]
    if not times: print(f"{label:56s} nothing"); return
    n=pd.Series(pd.DatetimeIndex(times)).dt.month.value_counts().reindex(range(1,13),fill_value=0)
    rate=100*n/denom; r=rate[COLD_M].mean()/rate[WARM_M].mean()
    print(f"{label:56s} {100*len(times)/len(shared):5.1f}%  cold/warm {r:5.2f}")
    if show: print("      "+"  ".join(f"{MONTHS[m-1]} {rate[m]:4.1f}" for m in range(1,13)))

sel = raw[ok & (delta<=25.0)]
for ft in ["TROF","STNRY"]:
    ratio(sel[sel.ftype==ft].atime.unique(), f"inner Piedmont 300km +/-25, {ft} only", show=True)

trof = sel[sel.ftype=="TROF"]
for box,name in [((41,50,-78,-63),"tight NE"),((42,50,-76,-64),"inland NE"),((40,48,-74,-60),"Gulf of Maine")]:
    la0,la1,lo0,lo1=box
    for p in [0,1018,1022,1026]:
        sub=hi[hi.lat.between(la0,la1)&hi.lon.between(lo0,lo1)&(hi.pressure_hpa>=p)]
        ratio(sorted(set(trof.atime)&set(sub.atime)), f"TROF + {name} high >= {p}", show=(p==1022))
