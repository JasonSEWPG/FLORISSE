from florisse.GeneralWindFarmComponents import getTurbineZ, speedFreq, actualSpeeds
import numpy as np
import matplotlib.pyplot as plt
from florisse.floris import AEPGroup
from commonse.environment import PowerWind, LogWind
import matplotlib.pyplot as plt
from openmdao.api import Problem, Group, IndepVarComp
import time
import cPickle as pickle


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

    opt_filename = "practiceAEP165XY_MSP1.txt"

    nRows = 5
    nTurbs = nRows**2
    spacing = 3   # turbine grid spacing in diameters
    points = np.linspace(start=spacing*rotor_diameter, stop=nRows*spacing*rotor_diameter, num=nRows)
    xpoints, ypoints = np.meshgrid(points, points)
    turbineXstart = np.ndarray.flatten(xpoints)
    turbineYstart = np.ndarray.flatten(ypoints)

    opt = open(opt_filename)
    optimized = np.loadtxt(opt)
    turbineX = optimized[:,0]
    turbineY = optimized[:,1]
    turbineZ = optimized[:,2]
    nTurbs = len(turbineX)

    turbineH1 = float(turbineZ[0])
    turbineH2 = float(turbineZ[1])


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

    minSpacing = 2



    """Define wind flow"""
    air_density = 1.1716    # kg/m^3

    windData = "Amalia"

    """Amalia Wind Arrays"""
    if windData == "Amalia":
        windSpeeds = np.array([6.53163342, 6.11908394, 6.13415514, 6.0614625,  6.21344602,
                                    5.87000793, 5.62161519, 5.96779107, 6.33589422, 6.4668016,
                                    7.9854581,  7.6894432,  7.5089221,  7.48638098, 7.65764618,
                                    6.82414044, 6.36728201, 5.95982999, 6.05942132, 6.1176321,
                                    5.50987893, 4.18461796, 4.82863115, 0.,         0.,         0.,
                                    5.94115843, 5.94914252, 5.59386528, 6.42332524, 7.67904937,
                                    7.89618066, 8.84560463, 8.51601497, 8.40826823, 7.89479475,
                                    7.86194762, 7.9242645,  8.56269962, 8.94563889, 9.82636368,
                                   10.11153102, 9.71402212, 9.95233636,  10.35446959, 9.67156182,
                                    9.62462527, 8.83545158, 8.18011771, 7.9372492,  7.68726143,
                                    7.88134508, 7.31394723, 7.01839896, 6.82858346, 7.06213432,
                                    7.01949894, 7.00575122, 7.78735165, 7.52836352, 7.21392201,
                                    7.4356621,  7.54099962, 7.61335262, 7.90293531, 7.16021596,
                                    7.19617087, 7.5593657,  7.03278586, 6.76105501, 6.48004694,
                                    6.94716392])

        windFrequencies = np.array([1.17812570e-02, 1.09958570e-02, 9.60626600e-03, 1.21236860e-02,
                                   1.04722450e-02, 1.00695140e-02, 9.68687400e-03, 1.00090550e-02,
                                   1.03715390e-02, 1.12172280e-02, 1.52249700e-02, 1.56279300e-02,
                                   1.57488780e-02, 1.70577560e-02, 1.93535770e-02, 1.41980570e-02,
                                   1.20632100e-02, 1.20229000e-02, 1.32111160e-02, 1.74605400e-02,
                                   1.72994400e-02, 1.43993790e-02, 7.87436000e-03, 0.00000000e+00,
                                   2.01390000e-05, 0.00000000e+00, 3.42360000e-04, 3.56458900e-03,
                                   7.18957000e-03, 8.80068000e-03, 1.13583200e-02, 1.41576700e-02,
                                   1.66951900e-02, 1.63125500e-02, 1.31709000e-02, 1.09153300e-02,
                                   9.48553000e-03, 1.01097900e-02, 1.18819700e-02, 1.26069900e-02,
                                   1.58895900e-02, 1.77021600e-02, 2.04208100e-02, 2.27972500e-02,
                                   2.95438600e-02, 3.02891700e-02, 2.69861000e-02, 2.21527500e-02,
                                   2.12465500e-02, 1.82861400e-02, 1.66147400e-02, 1.90111800e-02,
                                   1.90514500e-02, 1.63932050e-02, 1.76215200e-02, 1.65341460e-02,
                                   1.44597600e-02, 1.40370300e-02, 1.65745000e-02, 1.56278200e-02,
                                   1.53459200e-02, 1.75210100e-02, 1.59702700e-02, 1.51041500e-02,
                                   1.45201100e-02, 1.34527800e-02, 1.47819600e-02, 1.33923300e-02,
                                   1.10562900e-02, 1.04521380e-02, 1.16201970e-02, 1.10562700e-02])

        # windSpeeds = windSpeeds* 1.714285714 #to get the average speed higher, close to 12 m/s
        nDirections = len(windSpeeds)
        windDirections = np.linspace(0,360-360/nDirections, nDirections)

    """Manual Wind Arrays"""
    if windData == "Manual":
        nDirections = 1
        windSpeeds = np.ones(nDirections)*10.
        # print nDirections
        # windDirections = np.linspace(0,360-360/nDirections, nDirections)
        # windFrequencies = np.ones(len(windSpeeds))/len(windSpeeds)
        windDirections = np.array([0.])
        windFrequencies = np.array([1.])


    index = np.where(windSpeeds==0.0)
    windSpeeds = np.delete(windSpeeds, index[0])
    windFrequencies = np.delete(windFrequencies, index[0])
    windDirections = np.delete(windDirections, index[0])
    nDirections = len(windSpeeds)

    nIntegrationPoints = 1 #Number of points in wind effective wind speed integral

    shearExp = 0.1



    """set up 3D aspects of wind farm"""
    H1_H2 = np.array([])
    for i in range(nTurbs/2):
        H1_H2 = np.append(H1_H2, 0)
        H1_H2 = np.append(H1_H2, 1)
    if len(H1_H2) < nTurbs:
        H1_H2 = np.append(H1_H2, 0)

    """set up the problem"""
    prob = Problem()
    root = prob.root = Group()

    root.add('turbineH1', IndepVarComp('turbineH1', turbineH1), promotes=['*'])
    root.add('turbineH2', IndepVarComp('turbineH2', turbineH2), promotes=['*'])
    root.add('getTurbineZ', getTurbineZ(nTurbs), promotes=['*'])
    root.add('AEPGroup', AEPGroup(nTurbs, nDirections=nDirections,
                use_rotor_components=use_rotor_components, datasize=datasize, differentiable=True,
                optimizingLayout=False, nSamples=0), promotes=['*'])

    prob.setup()

    prob['shearExp'] = shearExp
    prob['turbineH1'] = turbineH1
    prob['turbineH2'] = turbineH2
    prob['H1_H2'] = H1_H2

    prob['turbineX'] = turbineXstart
    prob['turbineY'] = turbineYstart
    prob['yaw0'] = yaw
    prob['ratedPower'] = np.ones_like(turbineX)*5000 # in kw

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
    prob['floris_params:cos_spread'] = 1E12

    prob.run()

    """print the results"""

    print 'Turbine H1: ', prob['turbineH1']
    print 'Turbine H2: ', prob['turbineH2']
    print 'Rotor Diameter: ', rotor_diameter
    print 'H1_H2: ', prob['H1_H2']
    print 'TurbineX: ', prob['turbineX']
    print 'TurbineY: ', prob['turbineY']

    print 'AEP: ', prob['AEP']

    AEPstart = prob['AEP']

    prob = Problem()
    root = prob.root = Group()

    root.add('turbineH1', IndepVarComp('turbineH1', turbineH1), promotes=['*'])
    root.add('turbineH2', IndepVarComp('turbineH2', turbineH2), promotes=['*'])
    root.add('getTurbineZ', getTurbineZ(nTurbs), promotes=['*'])
    root.add('AEPGroup', AEPGroup(nTurbs, nDirections=nDirections,
                use_rotor_components=use_rotor_components, datasize=datasize, differentiable=True,
                optimizingLayout=False, nSamples=0), promotes=['*'])

    prob.setup()

    prob['shearExp'] = shearExp
    prob['turbineH1'] = turbineH1
    prob['turbineH2'] = turbineH2
    prob['H1_H2'] = H1_H2

    prob['turbineX'] = turbineX
    prob['turbineY'] = turbineY
    prob['yaw0'] = yaw
    prob['ratedPower'] = np.ones_like(turbineX)*5000 # in kw

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
    prob['floris_params:cos_spread'] = 1E12

    prob.run()

    AEPopt = prob['AEP']


    print 'Start AEP: ', AEPstart
    print 'Optimal AEP: ', AEPopt
