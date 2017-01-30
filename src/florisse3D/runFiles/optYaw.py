from FLORISSE3D.COE import COEGroup
from towerse.tower import TowerSE
from FLORISSE3D.GeneralWindFarmComponents import calculate_boundary, SpacingComp,\
            BoundaryComp, get_z, get_z_DEL, getTurbineZ, AEPobj, speedFreq, actualSpeeds
import numpy as np
import matplotlib.pyplot as plt
from FLORISSE3D.floris import AEPGroup
from commonse.environment import PowerWind, LogWind
import matplotlib.pyplot as plt
from openmdao.api import Problem, Group, IndepVarComp, ScipyOptimizer, pyOptSparseDriver
import time
import cPickle as pickle
from FLORISSE3D.setupOptimization import *
from scipy.interpolate import interp1d
from sys import argv


def frequ(bins, frequencies, speeds):
    f, size = wind_frequency_funcion(frequencies)
    g = wind_speeds_funcion(speeds)
    L = size/1.
    print "L: ", L
    bin_size = L/bins
    dx = 0.01
    x1 = 0.
    x2 = x1+dx
    bin_location = bin_size
    frequency = np.zeros(bins)
    windSpeeds = np.zeros(bins)
    for i in range(0, bins):
        sum_freq = 0.
        while x1 <= bin_location:
            dfrequency = dx*(f(x1)+f(x2))/2.
            dspeeds = (f(x1)*g(x1)+f(x2)*g(x2))/2.
            frequency[i] += dfrequency
            windSpeeds[i] += dspeeds
            sum_freq += f(x1)/2.+f(x2)/2.
            x1 = x2
            x2 += dx
        bin_location += bin_size
        windSpeeds[i] = windSpeeds[i]/sum_freq
    total = np.sum(frequency)
    frequency = frequency/total
    return frequency, windSpeeds

def wind_speeds_funcion(speeds):

    probability = speeds
    length_data = np.linspace(0,len(probability)+0.01,len(probability))
    f = interp1d(length_data, probability)
    return f

def wind_frequency_funcion(frequencies):

    probability = frequencies
    length_data = np.linspace(0,len(probability)+0.01,len(probability))
    f = interp1d(length_data, probability)
    return f, len(probability)

"""An example optimization of COE with tower structural constraints"""
"""Many gradients are FD"""
if __name__=="__main__":

    use_rotor_components = True

    if use_rotor_components:
        NREL5MWCPCT = pickle.load(open('doc/tune/NREL5MWCPCT_smooth_dict.p'))
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
    diff = 0.
    turbineH1 = 90.
    turbineH2 = 90.
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
        windSpeedsT, windFrequenciesT, windDirectionsT, nDirectionsT = amaliaWind()

    wind = 'PowerWind'

    shearExp = 0.1

    space = 3.0

    opt_filenameXYZ = 'src/florisse3D/Plots/Results_w_yaw/XYZ_XYZdt_analytic_%s.txt'%space
    shearExp = 0.1
    optXYZ = open(opt_filenameXYZ)
    optimizedXYZ = np.loadtxt(optXYZ)
    turbineX = optimizedXYZ[:,0]
    turbineY = optimizedXYZ[:,1]
    turbineZ = optimizedXYZ[:,2]
    turbineH1 = np.float(turbineZ[0])
    turbineH2 = np.float(turbineZ[1])

    # turbineH1 = 72.4
    # turbineH2 = 125.0
    # turbineX = np.array([0.,200.,400.,600.])
    # turbineY = np.array([0.,0.,0.,0.])

    yawAngles = np.zeros((nDirectionsT, nTurbs))

    """set up the problem"""
    for i in range(nDirectionsT):

        windDirections = np.array([windDirectionsT[i]])
        windSpeeds = np.array([windSpeedsT[i]])
        windFrequencies = np.array([1.0])
        nDirections = 1

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

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'SNOPT'
        prob.driver.opt_settings['Summary file'] = 'SNOPT_XYZdt_%s.out'%space
        prob.driver.opt_settings['Major iterations limit'] = 1000
        prob.driver.opt_settings['Major optimality tolerance'] = 1.0E-4
        prob.driver.opt_settings['Function precision'] = 1.0E-8

        # --- Objective ---
        prob.driver.add_objective('maxAEP', scaler=1.0E-8)

        # # --- Design Variables ---
        # for direction_id in range(0, windDirections.size):
        #     prob.driver.add_desvar('yaw%i' % direction_id, lower=-30.0, upper=30.0, scaler=1)
        prob.driver.add_desvar('yaw0', lower=-30.0, upper=30.0, scaler=1)

        prob.root.ln_solver.options['single_voi_relevance_reduction'] = True

        start = time.time()
        prob.setup()

        """run the problem"""


        prob['shearExp'] = shearExp
        prob['turbineH1'] = turbineH1
        prob['turbineH2'] = turbineH2
        prob['H1_H2'] = H1_H2

        prob['turbineX'] = turbineX
        prob['turbineY'] = turbineY
        # prob['yaw0'] = yaw
        prob['ratedPower'] = np.ones_like(turbineX)*5000 # in kw

        # assign values to constant inputs (not design variables)
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
        prob['floris_params:cos_spread'] = 1E12   # turns off cosine spread (just needs to be very large)
        prob['zref'] = 90.
        prob['z0'] = 0.


        prob.run()

        """print the results"""
        yawAngles[i] = prob['yaw0']

        print 'TurbineX: ', prob['turbineX']
        print 'TurbineY: ', prob['turbineY']
        print 'TurbineZ: ', prob['turbineZ']

        print 'AEP: ', prob['AEP']

        print 'nDirections: ', nDirections
        print 'Time to run: ', time.time() - start

        print 'yaw0',i,': ', prob['yaw0']

    for i in range(nDirectionsT):
        print 'Direction ', i
        print yawAngles[i]

    print 'nDirectionsT: ', nDirectionsT

    # for direction_id in range(windDirectionsT):
    #     print prob['yaw%i'%direction_id]

    # yaw = np.zeros((len(windDirections), nTurbs))
    # for direction_id in range(0, windDirections.size):
    #     yaw[i] = prob['yaw%i'%direction_id]

    # np.savetxt('YAW_%s.txt'%space, np.c_[yawAngles], header="yaw")
