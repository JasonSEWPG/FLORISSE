#plot wakes

import numpy as np
import math

import matplotlib as mpl
from matplotlib import pyplot as plt

from scipy.io import loadmat
import pickle

from florisse.floris import AEPGroup
from florisse import config

from openmdao.api import Problem

# set temp option to use unified floris
config.floris_single_component = True

ICOWESdata = loadmat('YawPosResults.mat')

# visualization: define resolution
resolution = 2000

# Define turbine characteristics
rotorDiameter = 126.4
rotor_diameter = rotorDiameter
rotorArea = np.pi*rotorDiameter*rotorDiameter/4.0
axialInduction = 1.0/3.0 # used only for initialization
generatorEfficiency = 0.944
hub_height = 90.0
NREL5MWCPCT = pickle.load(open('../../../doc/tune/NREL5MWCPCT_dict.p'))
datasize = NREL5MWCPCT['CP'].size

nRows = 2
nTurbs = nRows**2
spacing = 5.  # turbine grid spacing in diameters
points = np.linspace(start=spacing*rotor_diameter, stop=nRows*spacing*rotor_diameter, num=nRows)
xpoints, ypoints = np.meshgrid(points, points)
turbineXinit = np.ndarray.flatten(xpoints)
turbineYinit = np.ndarray.flatten(ypoints)

# opt_filenameXYZ = 'Z_XYZ_XYZdt_5.0.txt'
#
# optXYZ = open(opt_filenameXYZ)
# optimizedXYZ = np.loadtxt(optXYZ)
# turbineXinit = optimizedXYZ[:,0]
# turbineYinit = optimizedXYZ[:,1]

# turbineXinit = np.array([1118.1, 1881.9])
# turbineYinit = np.array([1279.5, 1720.5])

prob = Problem(root=AEPGroup(nTurbines=nTurbs, nDirections=1, differentiable=True, use_rotor_components=True,
                                   datasize=datasize, nSamples=resolution*resolution))

prob.setup()

# load turbine properties into FLORIS
prob['gen_params:windSpeedToCPCT_CP'] = NREL5MWCPCT['CP']
prob['gen_params:windSpeedToCPCT_CT'] = NREL5MWCPCT['CT']
prob['gen_params:windSpeedToCPCT_wind_speed'] = NREL5MWCPCT['wind_speed']
prob['floris_params:cos_spread'] = 1E12
prob['axialInduction'] = np.ones(nTurbs)*axialInduction
prob['rotorDiameter'] = np.ones(nTurbs)*rotorDiameter
prob['hubHeight'] = np.ones(nTurbs)*hub_height
prob['generatorEfficiency'] = np.ones(nTurbs)*generatorEfficiency

# opt_filenameXYZ = 'AAoneDirXYZ2.txt'
# optXYZ = open(opt_filenameXYZ)
# optimizedXYZ = np.loadtxt(optXYZ)
# turbineXinit = optimizedXYZ[:,0]
# turbineYinit = optimizedXYZ[:,1]

prob['turbineX'] = turbineXinit
prob['turbineY'] = turbineYinit


# Define site measurements
windDirection = 270.
prob['windDirections'] = np.array([windDirection])
wind_speed = 10.    # m/s
prob['windSpeeds'] = np.array([wind_speed])
prob['air_density'] = 1.1716
wind_frequency = 1.0
prob['windFrequencies'] = np.array([wind_frequency])

# prob.initVelocitiesTurbines = np.ones_like(prob.windrose_directions)*wind_speed

windDirectionPlot = 270. - windDirection
# visualization:
#    generate points downstream slice
y_cut = np.linspace(-rotorDiameter, rotorDiameter, resolution)
z_cut = np.linspace(-hub_height, rotorDiameter, resolution)
yy, zz = np.meshgrid(y_cut, z_cut)
xx = np.ones(yy.shape) * 3.5*rotorDiameter
position = np.array([xx.flatten(), yy.flatten(), zz.flatten()])
rotationMatrix = np.array([(np.cos(windDirectionPlot*np.pi/180.), -np.sin(windDirectionPlot*np.pi/180.), 0.),
                                   (np.sin(windDirectionPlot*np.pi/180.), np.cos(windDirectionPlot*np.pi/180.), 0.),
                                   (0., 0., 1.)])
positionF = np.dot(rotationMatrix, position) + np.dot(np.array([(prob['turbineX'][0], prob['turbineY'][0], hub_height)]).transpose(), np.ones((1, np.size(position, 1))))

#      generate points hub-height
xlow = np.min(turbineXinit)
xhigh = np.max(turbineXinit)
ylow = np.min(turbineYinit)
yhigh = np.max(turbineYinit)

x = np.linspace(xlow-200.,xhigh+500., resolution)
y = np.linspace(ylow-200.,yhigh+200., resolution)
xx, yy = np.meshgrid(x, y)
wsPositionX = xx.flatten()
wsPositionY = yy.flatten()
wsPositionZ = np.ones(wsPositionX.shape)*hub_height

velocities = list()
velocities_cut = list()

prob['yaw0'] = np.array([20.,0.,-20.,0.])

# Call FLORIS horizontal slice
prob['wsPositionX'] = np.copy(wsPositionX)
prob['wsPositionY'] = np.copy(wsPositionY)
prob['wsPositionZ'] = np.copy(wsPositionZ)
prob.run()
velocities.append(np.array(prob['wsArray0']))

# plot slices
velocities = np.array(velocities)
vmin = 3.193
vmax = 10.0
velocities_cut = np.array(velocities_cut)

fig = plt.figure(frameon=False)
ax = fig.add_subplot(111)
# ax.axis('off')

vel = velocities.flatten()
vel = vel.reshape(len(y), len(x))
im = plt.pcolormesh(x, y, vel, cmap='Blues_r', vmin=vmin, vmax=vmax)
R = rotor_diameter/2.
for i in range(nTurbs):
    x = turbineXinit[i]
    y = turbineYinit[i]
    theta = math.radians(prob['yaw0'][i])
    # theta = math.radians(225.)
    # theta = 0.
    xt = x-R*np.sin(theta)
    xb = x+R*np.sin(theta)
    yt = y+R*np.cos(theta)
    yb = y-R*np.cos(theta)
    plt.plot([xt,xb],[yt,yb],'k',linewidth=4)
    # plt.plot(np.array([x,x]),np.array([y-R,y+R]),'k',linewidth=4)
ax.set_aspect('equal')
# ax.set_xticks(np.arange(500, 1751, 250))
# ax.set_yticks(np.arange(500, 1501, 250))
ax.autoscale(tight=True)

cbar = plt.colorbar(im,fraction=0.046, pad=0.04, orientation = 'horizontal', ticks=[vmin,(vmin+vmax)/2,vmax])
cbar.set_label('wind speed (m/s)')
mpl.rcParams.update({'font.size': 20})
plt.tight_layout()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

# plt.title('Original Layout')




if __name__ == "__main__":
    plt.show()
else:
    plt.show(block=False)
