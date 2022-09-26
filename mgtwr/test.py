import numpy as np
import time
import libpysal as ps
import pandas as pd
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
data = ps.io.open(ps.examples.get_path('GData_utm.csv'))
coords = list(zip(data.by_col('X'), data.by_col('Y')))
longitude = np.array(coords)[:, 0].reshape(-1, 1)
latitude = np.array(coords)[:, 1].reshape(-1, 1)
y = np.array(data.by_col('PctBach')).reshape((-1, 1))
rural = np.array(data.by_col('PctRural')).reshape((-1, 1))
pov = np.array(data.by_col('PctPov')).reshape((-1, 1))
african_amer = np.array(data.by_col('PctBlack')).reshape((-1, 1))
X = np.hstack([rural, pov, african_amer])
bw = Sel_BW(coords, y, X).search(criterion='AICc', verbose=True, bw_min=0, bw_max=100)

model1 = GWR(coords, y, X, 94, kernel='bisquare', constant=False)
a = model1.fit()

a = np.ones((100, 5))
from gtwrtest.selt import SearchGWRParameter
model2 = SearchGWRParameter(coords, X, y, kernel='bisquare', fixed=False, thread=1)
bw0 = model2.search(verbose=True)
data = pd.DataFrame({
    'longitude': np.array(coords)[:, 0],
    'latitude': np.array(coords)[:, 1],
    'x1': rural.reshape(-1),
    'x2': pov.reshape(-1),
    'x3': african_amer.reshape(-1),
    'y': y
})
data.to_csv(r'D:\project\mgtwr\data\gwr.csv')
import numpy as np
from gtwrtest.selt import SearchGTWRParameter
np.random.seed(1)
u = np.array([(i-1)%12 for i in range(1,1729)]).reshape(-1,1)
v = np.array([((i-1)%144)//12 for i in range(1,1729)]).reshape(-1,1)
t = np.array([(i-1)//144 for i in range(1,1729)]).reshape(-1,1)
x1 = np.random.uniform(0,1,(1728,1))
x2 = np.random.uniform(0,1,(1728,1))
epsilon = np.random.randn(1728,1)
beta0 = 5
beta1 = 3 + (u + v + t)/6
beta2 = 3+((36-(6-u)**2)*(36-(6-v)**2)*(36-(6-t)**2))/128
y = beta0 + beta1 * x1 + beta2 * x2 + epsilon
coords = np.hstack([u,v])
X = np.hstack([x1,x2])
sel = SearchGTWRParameter(coords, t, X, y, kernel='gaussian', fixed=True, thread=12)
bw, tau = sel.search(bw_max=5, tau_max=20, verbose=True)
import time
from mgtwr.sel import SelBws
s = time.time()
sel = SelBws(coords, t, X, y, kernel='gaussian', fixed=True)
bw, tau = sel.search(bw_max=5, tau_max=20, verbose=True)
e = time.time()
print(e - s)

import numpy as np
from gtwrtest.selt import SearchMGTWRParameter
from gtwrtest.modelt import MGTWR
np.random.seed(1)
u = np.array([(i-1) % 12 for i in range(1, 1729)]).reshape(-1, 1)
v = np.array([((i-1) % 144)//12 for i in range(1, 1729)]).reshape(-1, 1)
t = np.array([(i-1) // 144 for i in range(1, 1729)]).reshape(-1, 1)
x1 = np.random.uniform(0, 1, (1728, 1))
x2 = np.random.uniform(0, 1, (1728, 1))
epsilon = np.random.randn(1728, 1)
beta0 = 5
beta1 = 3 + (u + v + t)/6
beta2 = 3 + ((36-(6-u)**2)*(36-(6-v)**2)*(36-(6-t)**2)) / 128
y = beta0 + beta1 * x1 + beta2 * x2 + epsilon
coords = np.hstack([u, v])
X = np.hstack([x1, x2])
sel_multi = SearchMGTWRParameter(coords, t, X, y, kernel = 'gaussian', fixed=True, thread=12)
sel_multi.search(multi_bw_min=[0.1], multi_bw_max=[2], multi_tau_min=[0.1], multi_tau_max=[4], verbose=True, tol_multi=1.0e-2)
mgtwr = MGTWR(coords, t, X, y, sel_multi, kernel='gaussian', fixed=True)
mgtwr.fit()

from mgtwr.sel import SelBws
from mgtwr.model import MGTWR
s = time.time()
sel_multi = SelBws(coords, t, X, y, kernel = 'gaussian', fixed = True, multi = True)
sel_multi.search(multi_bw_min=[0.1], multi_bw_max=[2], multi_tau_min=[0.1], multi_tau_max=[4], verbose=True, tol_multi=1.0e-2)
mgtwr = MGTWR(coords, t, X, y, sel_multi, kernel = 'gaussian', fixed = True)
mgtwr.fit()
e = time.time()
print(e - s)

import pandas as pd
from gtwrtest.selt import SearchGWRParameter
from gtwrtest.modelt import GWR
from gtwrtest.function_ import _compute_betas_gwr

data = pd.read_csv(r'D:\project\mgtwr\data\gwr.csv')
coords = data[['longitude', 'latitude']]
X = data[['x1', 'x2', 'x3']]
y = data['y']

coords = coords.values
X = X.values
y = y.values

sel = GWR(coords, y, X, 94, kernel='bisquare', fixed=False, thread=1)
a = sel.fit()
bw0 = sel.search(verbose=True)

import numpy as np
from gtwr.sel import SearchGWRParameter
np.random.seed(1)
u = np.array([(i-1) % 12 for i in range(1, 1729)]).reshape(-1, 1)
v = np.array([((i-1) % 144)//12 for i in range(1, 1729)]).reshape(-1, 1)
t = np.array([(i-1) // 144 for i in range(1, 1729)]).reshape(-1, 1)
x1 = np.random.uniform(0, 1, (1728, 1))
x2 = np.random.uniform(0, 1, (1728, 1))
epsilon = np.random.randn(1728, 1)
beta0 = 5
beta1 = 3 + (u + v + t)/6
beta2 = 3 + ((36-(6-u)**2)*(36-(6-v)**2)*(36-(6-t)**2)) / 128
y = beta0 + beta1 * x1 + beta2 * x2 + epsilon
coords = np.hstack([u, v])
X = np.hstack([x1, x2])
sel = SearchGWRParameter(coords, y, X, kernel='gaussian', fixed=True)
bw = sel.search(bw_max=40, verbose=True)
