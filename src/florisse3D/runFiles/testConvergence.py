from FLORISSE3D.COE import COEGroup
from FLORISSE3D.GeneralWindFarmComponents import calculate_boundary, SpacingComp,\
            BoundaryComp, get_z, getTurbineZ, AEPobj, speedFreq, actualSpeeds
import numpy as np
import matplotlib.pyplot as plt
from FLORISSE3D.floris import AEPGroup
from openmdao.api import Problem, Group, IndepVarComp, pyOptSparseDriver, ExecComp, ScipyOptimizer
import time
import cPickle as pickle
from setupOptimization import *
from scipy.interpolate import interp1d
from sys import argv

"""An example optimization of COE with tower structural constraints"""
"""Many gradients are FD"""
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
    # rotor_diameter = 106.6

    nRows = 2
    nTurbs = nRows**2

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
        Cp[turbI] = 0.7737 * 4.0 * 1.0/3.0 * np.power((1. - 1.0/3.0), 2)
        generatorEfficiency[turbI] = 1.0#0.944
        yaw[turbI] = 0.     # deg.

    minSpacing = 2.0

    """set up 3D aspects of wind farm"""
    H1_H2 = np.array([])
    for i in range(nTurbs/2):
        H1_H2 = np.append(H1_H2, 0)
        H1_H2 = np.append(H1_H2, 1)
    if len(H1_H2) < nTurbs:
        H1_H2 = np.append(H1_H2, 0)
    wind_zref = 90.
    wind_z0 = 0.

    """Define wind flow"""
    air_density = 1.1716    # kg/m^3

    windSpeeds = np.array([10.])
    windDirections = np.array([0.])
    windFrequencies = np.array([1.])
    nDirections = len(windSpeeds)

    nIntegrationPoints = 1 #Number of points in wind effective wind speed integral

    wind = 'PowerWind'

    shearExp = 0.15

    # opt_filenameXYZ = 'XYZ_super_6.3D_0.19.txt'
    #
    # optXYZ = open(opt_filenameXYZ)
    # optimizedXYZ = np.loadtxt(optXYZ)
    # turbineX = optimizedXYZ[:,0]
    # turbineY = optimizedXYZ[:,1]
    # turbineZ = optimizedXYZ[:,2]
    # turbineH1 = turbineZ[0]
    # turbineH2 = turbineZ[1]

    turbineX = np.array([ 500.17547272,  758.4,         379.2,         636.81568672])
    turbineY = np.array([462.38056137,  453.42929727,  684.36812836,  675.07119031])
    turbineH1 = 89.9356883984
    turbineH2 = 100.0

    num = 200
    turbine = np.zeros(nTurbs)
    turbine[1] = 1.
    turbineH1new = np.linspace(turbineH1-2.,turbineH1+2.,num)
    turbineH2new = np.linspace(turbineH2-2.,turbineH2+2.,num)

    turbineXnew = np.zeros((num, nTurbs))
    turbineYnew = np.zeros((num, nTurbs))

    for i in range(num):
        turbineXnew[i] = turbineX+(-5.+i*10./num)*turbine
        turbineYnew[i] = turbineY+(-5.+i*10./num)*turbine

    AEP = np.zeros(num)

    for i in range(num):
        # turbineH1 = turbineH1new[i]
        # turbineH2 = turbineH2new[i]
        # turbineX = turbineXnew[i]
        turbineY = turbineYnew[i]

        """set up the problem"""
        prob = Problem()
        root = prob.root = Group()

        #This component takes turbineH1, turbineH2, and H1_H2 and gives back an array
        #of turbineZ
        root.add('getTurbineZ', getTurbineZ(nTurbs), promotes=['*'])
        #These components adjust the parameterized z locations for TowerSE calculations
        #with respect to turbineZ
        root.add('AEPGroup', AEPGroup(nTurbs, nDirections=nDirections,
                    use_rotor_components=use_rotor_components, datasize=datasize, differentiable=True,
                    optimizingLayout=False, nSamples=0), promotes=['*'])

        prob.root.ln_solver.options['single_voi_relevance_reduction'] = True

        start = time.time()
        prob.setup()

        """run the problem"""

        if wind == "PowerWind":
            prob['shearExp'] = shearExp
        prob['turbineH1'] = turbineH1
        prob['turbineH2'] = turbineH2
        prob['H1_H2'] = H1_H2

        prob['turbineX'] = turbineX
        prob['turbineY'] = turbineY
        prob['yaw0'] = yaw
        prob['ratedPower'] = np.ones_like(turbineX)*5000. # in kw

        # assign values to constant inputs (not design variables)
        prob['nIntegrationPoints'] = nIntegrationPoints
        prob['rotorDiameter'] = rotorDiameter
        prob['axialInduction'] = axialInduction
        prob['generatorEfficiency'] = generatorEfficiency
        prob['air_density'] = air_density
        prob['windDirections'] = np.array([windDirections])
        prob['windFrequencies'] = np.array([windFrequencies])
        prob['Uref'] = windSpeeds
        if use_rotor_components == True:
            prob['gen_params:windSpeedToCPCT_CP'] = NREL5MWCPCT['CP']
            prob['gen_params:windSpeedToCPCT_CT'] = NREL5MWCPCT['CT']
            prob['gen_params:windSpeedToCPCT_wind_speed'] = NREL5MWCPCT['wind_speed']
        else:
            prob['Ct_in'] = Ct
            prob['Cp_in'] = Cp
        prob['floris_params:cos_spread'] = 1E12 # turns off cosine spread (just needs to be very large)
        prob['zref'] = wind_zref
        prob['z0'] = wind_z0

        prob.run()

        AEP[i] = prob['AEP']

    x = np.linspace(0,1,num)
    plt.plot(x,AEP,'or')

    # plt.figure(2)
    # for i in range(4):
    #     if turbine[i] == 1:
    #         plt.plot(turbineX[i], turbineY[i],'or')
    #     else:
    #         plt.plot(turbineX[i], turbineY[i],'ob')
    plt.show()
