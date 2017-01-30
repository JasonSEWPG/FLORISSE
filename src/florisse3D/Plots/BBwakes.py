# This example script compares FLORIS predictions with steady-state SOWFA data as obtained
# throught the simulations described in:
#

import numpy as np

import matplotlib as mpl
from matplotlib import pyplot as plt

from scipy.io import loadmat
import pickle

from florisse.floris import AEPGroup
from florisse import config

from openmdao.api import Problem

# set temp option to use unified floris
config.floris_single_component = True

# Load steady-state power data from SOWFA
ICOWESdata = loadmat('YawPosResults.mat')

# visualization: define resolution
resolution = 2000

# Define turbine characteristics
rotorDiameter = 126.4
rotorArea = np.pi*rotorDiameter*rotorDiameter/4.0
axialInduction = 1.0/3.0 # used only for initialization
generatorEfficiency = 0.944
hub_height = 90.
NREL5MWCPCT = pickle.load(open('../../../doc/tune/NREL5MWCPCT_dict.p'))
datasize = NREL5MWCPCT['CP'].size
turbineXinit = np.array([0., 3.4999*rotorDiameter])
turbineYinit = np.array([0., 20.])
turbineXinit = np.array([0.])
turbineYinit = np.array([0.])

nTurbs = len(turbineXinit)

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
# prob['hubHeight'] = np.array([hub_height, hub_height+160.])
prob['hubHeight'] = np.array([hub_height])
prob['generatorEfficiency'] = np.ones(nTurbs)*generatorEfficiency
prob['turbineX'] = turbineXinit
prob['turbineY'] = turbineYinit


# Define site measurements
# windDirection = 270.-0.523599*180./np.pi
windDirection = 270.
prob['windDirections'] = np.array([windDirection])
wind_speed = 10.0    # m/s
prob['windSpeeds'] = np.array([wind_speed])
prob['air_density'] = 1.1716
wind_frequency = 1.0
prob['windFrequencies'] = np.array([wind_frequency])

# prob.initVelocitiesTurbines = np.ones_like(prob.windrose_directions)*wind_speed

# windDirectionPlot = 270. - windDirection
windDirectionPlot = 0.
# visualization:
#    generate points downstream slice
y_cut = np.linspace(-2.*rotorDiameter, 2.*rotorDiameter, resolution)
z_cut = np.linspace(-hub_height, 1.3*rotorDiameter, resolution)
yy, zz = np.meshgrid(y_cut, z_cut)
xx = np.ones(yy.shape) * 3.5*rotorDiameter
position = np.array([xx.flatten(), yy.flatten(), zz.flatten()])
rotationMatrix = np.array([(np.cos(windDirectionPlot*np.pi/180.), -np.sin(windDirectionPlot*np.pi/180.), 0.),
                                   (np.sin(windDirectionPlot*np.pi/180.), np.cos(windDirectionPlot*np.pi/180.), 0.),
                                   (0., 0., 1.)])
positionF = np.dot(rotationMatrix, position) + np.dot(np.array([(prob['turbineX'][0], prob['turbineY'][0], hub_height)]).transpose(), np.ones((1, np.size(position, 1))))

#      generate points hub-height
x = np.linspace(750, 2400, resolution)
y = np.linspace(750, 2400, resolution)
xx, yy = np.meshgrid(x, y)
wsPositionX = xx.flatten()
wsPositionY = yy.flatten()
wsPositionZ = np.ones(wsPositionX.shape)*hub_height

# SWEEP TURBINE YAW
yawrange = ICOWESdata['yaw'][0]

velocities = list()
velocities_cut = list()

prob['yaw0'] = np.zeros(nTurbs)

# Call FLORIS cut-through slice
prob['wsPositionX'] = np.copy(positionF[0])
prob['wsPositionY'] = np.copy(positionF[1])
prob['wsPositionZ'] = np.copy(positionF[2])
prob.run()
velocities_cut.append(np.array(prob['wsArray0']))

# plot slices
velocities = np.array(velocities)
vmin = 3.193
vmax = 10.
velocities_cut = np.array(velocities_cut)

# fig, axes = plt.subplots(ncols=int(np.ceil(len(yawrange)/2.)), nrows=4, figsize=(23, 12))
# fig.suptitle("FLORIS flow-field prediction at hub-height and wake cut-through at 3.5D, for yaw sweep")

fig = plt.figure()
ax = fig.add_subplot(111)

# axes1 = list(axes[0])+list(axes[2])
# axes2 = list(axes[1])+list(axes[3])

vel = velocities_cut.flatten()
vel = vel.reshape(len(z_cut), len(y_cut))
im = ax.pcolormesh(y_cut, z_cut, vel, cmap='Blues_r', vmin=vmin, vmax=vmax)
ax.set_aspect('equal')
ax.autoscale(tight=True)
ax.invert_xaxis()
ax.set_xlim([-100,85])
# ax.set_ylim([-hub_height,250])


# cbar = plt.colorbar(im, orientation = 'horizontal', ticks=[vmin,(vmin+vmax)/2,vmax])
# cbar.set_label('wind speed (m/s)')
ax.axis('off')
ax.axis('off')
mpl.rcParams.update({'font.size': 12})
plt.tight_layout()
fig.subplots_adjust(top=0.95)


if __name__ == "__main__":
    plt.show()
else:
    plt.show(block=False)
