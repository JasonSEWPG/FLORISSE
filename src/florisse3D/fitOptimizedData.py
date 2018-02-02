import numpy as np
from scipy.interpolate import LinearNDInterpolator,SmoothBivariateSpline
from scipy import ndimage
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

filename = 'src/florisse3D/optRotor/rotor/OPTIMIZED.txt'
opedRatedT = open(filename)
ratedTdata = np.loadtxt(opedRatedT)
"ratedPower, rotorDiameter, ratedQ, blade_mass, Vrated, I1, I2, I3, ratedT, extremeT"
ratedPower = ratedTdata[:,0]
rotorDiameter = ratedTdata[:,1]
ratedQ = ratedTdata[:,2]
blade_mass = ratedTdata[:,3]
Vrated = ratedTdata[:,4]
I1 = ratedTdata[:,5]
I2 = ratedTdata[:,6]
I3 = ratedTdata[:,7]
ratedT = ratedTdata[:,8]
extremeT = ratedTdata[:,9]

num_fit = 30
ratedPower_fit = np.zeros(num_fit)
rotorDiameter_fit = np.zeros(num_fit)
ratedT_fit = np.zeros(num_fit)
for i in range(num_fit):
    index = int(len(ratedPower)/num_fit)*i
    ratedPower_fit[i] = ratedPower[index]
    rotorDiameter_fit[i] = rotorDiameter[index]
    ratedT_fit[i] = blade_mass[index]

print 'mean: ', np.mean(blade_mass)
print 'min: ', np.min(blade_mass)
print 'max: ', np.max(blade_mass)

# kernel = np.array([.2,.5])
# kernel = np.ones(len(ratedPower))
# ratedPower, rotorDiameter, blade_mass = ndimage.filters.gaussian_filter(np.array([ratedPower,rotorDiameter,blade_mass]),kernel)
print 'finished convolve'
cartcoord = list(zip(ratedPower,rotorDiameter))
interp_spline = LinearNDInterpolator(cartcoord,blade_mass)

cartcoord_fit = list(zip(ratedPower_fit,rotorDiameter_fit))
interp_spline_fit = LinearNDInterpolator(cartcoord_fit,ratedT_fit)
# w = np.ones(len(ratedPower_fit))
# interp_spline_fit = SmoothBivariateSpline(ratedPower_fit,rotorDiameter_fit,ratedT_fit,w)

print '500, 40: ', interp_spline(500.,40.)
print '1000, 60: ', interp_spline(1000.,60.)
print '3000, 95: ', interp_spline(3000.,95.)
print '8000, 120: ', interp_spline(8000.,120.)
print '6000, 160: ', interp_spline(6000.,160.)

print '500, 40: ', interp_spline_fit(500.,40.)
print '1000, 60: ', interp_spline_fit(1000.,60.)
print '3000, 95: ', interp_spline_fit(3000.,95.)
print '8000, 120: ', interp_spline_fit(8000.,120.)
print '6000, 160: ', interp_spline_fit(6000.,160.)

num = 100
x = np.linspace(500.,10000.,num)
y = np.linspace(25.,160.,num)
X,Y = np.meshgrid(x,y)
Z = np.zeros((num,num))
Z_fit = np.zeros((num,num))
for i in range(num):
    for j in range(num):
        Z[i][j] = interp_spline(x[i],y[j])
        Z_fit[i][j] = interp_spline_fit(x[i],y[j])

fig, ax = plt.subplots(nrows=1, ncols=2, subplot_kw={'projection': '3d'})
ax[0].plot_wireframe(X, Y, Z, color='k')
ax[0].plot(ratedPower, rotorDiameter, blade_mass, 'or')

ax[1].plot_wireframe(X, Y, Z_fit, color='k')
ax[1].plot(ratedPower_fit, rotorDiameter_fit, ratedT_fit, 'or')
for axes in ax:
    # axes.set_zlim(-0.2,1)
    axes.set_axis_off()

fig.tight_layout()
plt.xlabel('Turbine Rating')
plt.ylabel('Rotor Diameter')
plt.show()
