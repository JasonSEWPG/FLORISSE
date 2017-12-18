import numpy as np
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

start1 = time.time()

num = 50
x = np.linspace(500.,10000.,num)
y = np.linspace(25.,160.,num)

X,Y = np.meshgrid(x,y)

filename = 'src/florisse3D/optRotor/rotor/OPTIMIZED.txt'
opedRatedT = open(filename)
ratedTdata = np.loadtxt(opedRatedT)
 = ratedTdata[:]

interp_spline = RectBivariateSpline(x, y, Z)
print interp_spline


print interp_spline(6000.,74.)
print time.time() - start1
start2 = time.time()

print interp_spline(2133.,122.)
print time.time()-start2

x2 = np.linspace(500.,10000.,100)
y2 = np.linspace(25.,160.,100)



X2, Y2 = np.meshgrid(x2,y2)
Z2 = interp_spline(y2, x2)

fig, ax = plt.subplots(nrows=1, ncols=2, subplot_kw={'projection': '3d'})
ax[0].plot_wireframe(X, Y, Z, color='k')

ax[1].plot_wireframe(X2, Y2, Z2, color='k')
for axes in ax:
    # axes.set_zlim(-0.2,1)
    axes.set_axis_off()

fig.tight_layout()
plt.xlabel('Turbine Rating')
plt.ylabel('Rotor Diameter')
plt.show()
