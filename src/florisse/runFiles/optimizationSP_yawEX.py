from florisse.COE import COEGroup
from towerse.tower import TowerSE
from florisse.GeneralWindFarmComponents import calculate_boundary, SpacingComp,\
            BoundaryComp, get_z, get_z_DEL, getTurbineZ, AEPobj, speedFreq, actualSpeeds
import numpy as np
import matplotlib.pyplot as plt
from florisse.floris import AEPGroup
from commonse.environment import PowerWind, LogWind
import matplotlib.pyplot as plt
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

    nTurbs = 25

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

    """set up 3D aspects of wind farm"""
    H1_H2 = np.array([])
    for i in range(nTurbs/2):
        H1_H2 = np.append(H1_H2, 0)
        H1_H2 = np.append(H1_H2, 1)
    if len(H1_H2) < nTurbs:
        H1_H2 = np.append(H1_H2, 0)

    """Define wind flow"""
    air_density = 1.1716    # kg/m^3

    windData = "Amalia"

    """Amalia Wind Arrays"""
    if windData == "Amalia":
        windSpeeds, windFrequencies, windDirections, nDirections = amaliaWind()
    # windDirections = np.array([225.])
    # nDirections = len(windDirections)
    # windSpeeds = np.ones(nDirections)*10.
    # windFrequencies = np.ones(nDirections)/nDirections


    nIntegrationPoints = 1 #Number of points in wind effective wind speed integral

    wind = 'PowerWind'

    shearExp = 0.1

    space = float(argv[1])

    opt_filenameXYZ = 'XYZ_3.1_5_%s_SP_2_6.3.txt'%space
    shearExp = space
    optXYZ = open(opt_filenameXYZ)
    optimizedXYZ = np.loadtxt(optXYZ)
    turbineX = optimizedXYZ[:,0]
    turbineY = optimizedXYZ[:,1]
    turbineZ = optimizedXYZ[:,2]
    turbineH1 = np.float(turbineZ[0])
    turbineH2 = np.float(turbineZ[1])

    yawTotal = np.zeros((len(windDirections), nTurbs))


    for direction in range(nDirections):
        """set up the problem"""
        prob = Problem()
        root = prob.root = Group()

        root.deriv_options['type'] = 'fd'
        root.deriv_options['form'] = 'central'
        root.deriv_options['step_size'] = 1.E-4
        root.deriv_options['step_type'] = 'relative'

        root.add('turbineH1', IndepVarComp('turbineH1', turbineH1), promotes=['*'])
        root.add('turbineH2', IndepVarComp('turbineH2', turbineH2), promotes=['*'])
        #This component takes turbineH1, turbineH2, and H1_H2 and gives back an array
        #of turbineZ
        root.add('getTurbineZ', getTurbineZ(nTurbs), promotes=['*'])
        #These components adjust the parameterized z locations for TowerSE calculations
        #with respect to turbineZ

        root.add('AEPGroup', AEPGroup(nTurbs, nDirections=1,
                    use_rotor_components=use_rotor_components, datasize=datasize, differentiable=True,
                    optimizingLayout=False, nSamples=0), promotes=['*'])
        root.add('maxAEP', AEPobj(), promotes=['*'])

        prob.driver = ScipyOptimizer()
        prob.driver.options['optimizer'] = 'SLSQP'

        # --- Objective ---
        prob.driver.add_objective('maxAEP', scaler=1.0E-5)

        # # --- Design Variables ---
        # for direction_id in range(0, windDirections.size):
        prob.driver.add_desvar('yaw0', lower=-30.0, upper=30.0, scaler=1)

        prob.root.ln_solver.options['single_voi_relevance_reduction'] = True

        start = time.time()
        prob.setup()

        """run the problem"""

        if wind == "PowerWind":
            prob['shearExp'] = shearExp
        prob['turbineH1'] = turbineH1
        prob['turbineH2'] = turbineH2
        # for direction_id in range(0, windDirections.size):
        prob['yaw0'] = yaw
        prob['H1_H2'] = H1_H2

        prob['turbineX'] = turbineX
        prob['turbineY'] = turbineY
        prob['ratedPower'] = np.ones_like(turbineX)*5000 # in kw

        # assign values to constant inputs (not design variables)
        prob['nIntegrationPoints'] = nIntegrationPoints
        prob['rotorDiameter'] = rotorDiameter
        prob['axialInduction'] = axialInduction
        prob['generatorEfficiency'] = generatorEfficiency
        prob['air_density'] = air_density
        # prob['windDirections'] = np.array([windDirections])
        # prob['windFrequencies'] = np.array([windFrequencies])
        prob['windDirections'] = np.array([windDirections[direction]])
        prob['windFrequencies'] = np.array([windFrequencies[direction]])
        prob['windSpeeds'] = np.array([windSpeeds[direction]])

        prob['Uref'] = np.array([windSpeeds[direction]])
        if use_rotor_components == True:
            prob['gen_params:windSpeedToCPCT_CP'] = NREL5MWCPCT['CP']
            prob['gen_params:windSpeedToCPCT_CT'] = NREL5MWCPCT['CT']
            prob['gen_params:windSpeedToCPCT_wind_speed'] = NREL5MWCPCT['wind_speed']
        else:
            prob['Ct_in'] = Ct
            prob['Cp_in'] = Cp
        prob['floris_params:cos_spread'] = 1E12   # turns off cosine spread (just needs to be very large)
        prob['zref'] = 90.
        prob['z0'] = 0.


        prob.run()

        """print the results"""

        print 'TurbineX: ', prob['turbineX']
        print 'TurbineY: ', prob['turbineY']
        print 'TurbineZ: ', prob['turbineZ']

        print 'AEP: ', prob['AEP']

        print 'nDirections: ', nDirections
        print 'Time to run: ', time.time() - start

        # for direction_id in range(0, windDirections.size):
        print prob['yaw0']

        # for direction_id in range(0, len(windDirections)):
        yawTotal[direction] = prob['yaw0']

    print yawTotal

    np.savetxt('YAW_%s.txt'%space, np.c_[yawTotal], header="yaw")
