from src import * 
import numpy as np
u = 3. * np.ones(2)
zu = 10. * np.ones(2)
tair = 24. * np.ones(2) # deg C air temp 
ztair = 2. * np.ones(2) #2-meter air temperature 
rh = 0.2  * np.ones(2)
zq = 2. * np.ones(2) #RH height
ts = 31. * np.ones(2) #sea-surface temperature 
a = coare30vncoare30vn(u, zu, tair, ztair, rh, zq, ts = ts)
a["hlb"]