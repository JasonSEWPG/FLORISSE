import numpy as np
from FLORISSE3D.COE import *
from towerse.tower import TowerSE
from FLORISSE3D.GeneralWindFarmComponents import calculate_boundary, SpacingComp,\
            BoundaryComp, get_z, organizeWindSpeeds, getTurbineZ, AEPobj, speedFreq, actualSpeeds
from FLORISSE3D.simpleTower import calcMass
import matplotlib.pyplot as plt
from FLORISSE3D.floris import AEPGroup
from openmdao.api import Problem, Group, IndepVarComp, pyOptSparseDriver, ExecComp, ScipyOptimizer
import time
# from setupOptimization import *
import cPickle as pickle
from setupOptimization import *

if __name__=="__main__":

    use_rotor_components = True

    if use_rotor_components:
        NREL5MWCPCT = pickle.load(open('../../../doc/tune/NREL5MWCPCT_smooth_dict.p'))
        # print(NREL5MWCPCT)
        # NREL5MWCPCT = pickle.Unpickler(open('./input_files/NREL5MWCPCT.p')).load()
        datasize = NREL5MWCPCT['CP'].size
    else:
        datasize = 0

    rotor_diameter = 126.4

    # nRows = 1
    # nTurbs = nRows**2
    # spacing = 3   # turbine grid spacing in diameters
    # points = np.linspace(start=spacing*rotor_diameter, stop=nRows*spacing*rotor_diameter, num=nRows)
    # xpoints, ypoints = np.meshgrid(points, points)
    # turbineX = np.ndarray.flatten(xpoints)
    # turbineY = np.ndarray.flatten(ypoints)
    nTurbs = 4
    num = 50.
    turbineX = np.array([0.,1.*num,2.*num,3.*num])
    turbineY = np.array([0.,200.,400.,600.])


    turbineH1 = 72.5
    turbineH2 = 125.

    rotorDiameter = np.zeros(nTurbs)
    axialInduction = np.zeros(nTurbs)
    Ct = np.zeros(nTurbs)
    Cp = np.zeros(nTurbs)
    generatorEfficiency = np.zeros(nTurbs)
    yaw = np.zeros(nTurbs)


    # define initial values
    for turbI in range(0, nTurbs):
        rotorDiameter[turbI] = rotor_diameter            # m
        axialInduction[turbI] = 1.0/3.0
        Ct[turbI] = 4.0*axialInduction[turbI]*(1.0-axialInduction[turbI])
        # Cp[turbI] = 0.7737/0.944 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
        Cp[turbI] = 0.7737 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
        generatorEfficiency[turbI] = 1.0#0.944
        yaw[turbI] = 0.     # deg.

    """Define wind flow"""
    air_density = 1.1716    # kg/m^3

    windSpeeds, windFrequencies, windDirections, nDirections = amaliaWind()
    windSpeeds = np.array([10.])
    windFrequencies = np.array([.5])
    windDirections = np.array([0.])
    nDirections = len(windSpeeds)

    shearExp = 0.1

    """set up 3D aspects of wind farm"""
    H1_H2 = np.array([])
    for i in range(nTurbs/2):
        H1_H2 = np.append(H1_H2, 0)
        H1_H2 = np.append(H1_H2, 1)
    if len(H1_H2) < nTurbs:
        H1_H2 = np.append(H1_H2, 0)

    AEP = np.zeros(500)
    for i in range(len(AEP)):
        turbineX[0] = turbineX[0]+0.5
        """set up the problem"""
        prob = Problem()
        root = prob.root = Group()

        root.add('getTurbineZ', getTurbineZ(nTurbs), promotes=['*'])
        root.add('AEPGroup', AEPGroup(nTurbs, nDirections=nDirections,
                    use_rotor_components=use_rotor_components, datasize=datasize, differentiable=True,
                    optimizingLayout=False, nSamples=0), promotes=['*'])

        prob.setup()

        prob['turbineH1'] = turbineH1
        prob['turbineH2'] = turbineH2
        prob['H1_H2'] = H1_H2

        prob['turbineX'] = turbineX
        prob['turbineY'] = turbineY
        prob['yaw0'] = yaw
        prob['ratedPower'] = np.ones_like(turbineX)*5000 # in kw

        # assign values to constant inputs (not design variables)
        prob['rotorDiameter'] = rotorDiameter
        prob['axialInduction'] = axialInduction
        prob['generatorEfficiency'] = generatorEfficiency
        prob['air_density'] = air_density
        prob['windDirections'] = np.array([windDirections])
        prob['windFrequencies'] = np.array([windFrequencies])
        if use_rotor_components == True:
            prob['gen_params:windSpeedToCPCT_CP'] = NREL5MWCPCT['CP']
            prob['gen_params:windSpeedToCPCT_CT'] = NREL5MWCPCT['CT']
            prob['gen_params:windSpeedToCPCT_wind_speed'] = NREL5MWCPCT['wind_speed']
        else:
            prob['Ct_in'] = Ct
            prob['Cp_in'] = Cp
        prob['floris_params:cos_spread'] = 1E12
        prob['shearExp'] = shearExp
        prob['Uref'] = windSpeeds
        prob['zref'] = 90.
        prob['z0'] = 0.

        prob.run()

        AEP[i] = prob['AEP']

    x = np.linspace(0,1000,500)
    plt.plot(x,AEP)
    plt.show()
