import numpy as np
from math import pi
import time
from openmdao.api import Group, Component, Problem
from florisse.floris import AEPGroup
from COE import *

if __name__=="__main__":

    numRows = 6

    filename = "XYZ%stest.txt"%(numRows)

    file = open(filename)
    xin = np.loadtxt(file)
    n = len(xin)
    turbineX = np.zeros(n)
    turbineY = np.zeros(n)
    turbineZ = np.zeros(n)
    for i in range(n):
        turbineX[i] = xin[i][0]
        turbineY[i] = xin[i][1]
        turbineZ[i] = xin[i][2]

    z1 = turbineZ[0]
    nTurbs = len(turbineX)
    z2 = turbineZ[nTurbs-1]
    nTurbsH1 = 0
    for i in range(nTurbs):
        if turbineZ[i] == z1:
            nTurbsH1 += 1
    nTurbsH2 = nTurbs-nTurbsH1

    # initialize input variable arrays
    size = 36
    rotorDiameter = np.zeros(nTurbs)
    axialInduction = np.zeros(nTurbs)
    Ct = np.zeros(nTurbs)
    Cp = np.zeros(nTurbs)
    generatorEfficiency = np.zeros(nTurbs)
    yaw = np.zeros(nTurbs)

    # define initial values
    for turbI in range(0, nTurbs):
        rotorDiameter[turbI] = 126.4            # m
        axialInduction[turbI] = 1.0/3.0
        Ct[turbI] = 4.0*axialInduction[turbI]*(1.0-axialInduction[turbI])
        Cp[turbI] = 0.7737/0.944 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
        # Cp[turbI] = 0.7737 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
        generatorEfficiency[turbI] = 0.944
        yaw[turbI] = 0.     # deg.

    # Define flow properties
    wind_speed = 8.0        # m/s #TODO
    air_density = 1.1716    # kg/m^3
    # wind_direction = 240    # deg (N = 0 deg., using direction FROM, as in met-mast data)
    
    nDirections = size
    windDirections = np.linspace(0, 360.-360/nDirections, nDirections)
    windFrequencies = np.ones_like(windDirections)*1.0/size
    # set up problem
    prob = Problem()
    root = prob.root = Group()
    
    #root.add('getTurbineZ', getTurbineZ(nTurbs), promotes=['*'])
    root.add('AEPGroup', AEPGroup(nTurbs, nDirections=nDirections,
                use_rotor_components=False, datasize=0, differentiable=True,
                optimizingLayout=False, nSamples=0), promotes=['*'])
    root.add('COEComponent', COEComponent(nTurbs), promotes=['*'])
    
    #root.ln_solver = ScipyGMRES()

    # initialize problem
    prob.setup()

    prob['turbineX'] = turbineX
    prob['turbineY'] = turbineY
    #prob['turbineH1'] = z1
    #prob['turbineH2'] = z2
    #prob['nTurbsH1'] = nTurbsH1
    #prob['nTurbsH2'] = nTurbsH2
    prob['turbineZ'] = turbineZ
    prob['yaw0'] = yaw

    # assign values to constant inputs (not design variables)
    prob['rotorDiameter'] = rotorDiameter
    prob['axialInduction'] = axialInduction
    prob['generatorEfficiency'] = generatorEfficiency
    prob['windSpeeds'] = np.array([wind_speed])
    prob['air_density'] = air_density
    #prob['windDirections'] = np.array([wind_direction])
    prob['windFrequencies'] = windFrequencies
    prob['Ct_in'] = Ct
    prob['Cp_in'] = Cp
    prob['floris_params:cos_spread'] = 1E12         # turns off cosine spread (just needs to be very large)

    # run the problem
    print 'start run'
    tic = time.time()
    prob.run()
    toc = time.time()

    first = prob['COE']

    print 'turbineZ: ', prob['turbineZ']
    print 'AEP: ', prob['AEP']
    print 'COE: ', prob['COE']    
    optCOE = prob['COE']







    filename = "XY%stest.txt"%(numRows)

    file = open(filename)
    xin = np.loadtxt(file)
    n = len(xin)
    turbineX = np.zeros(n)
    turbineY = np.zeros(n)
    turbineZ = np.zeros(n)
    for i in range(n):
        turbineX[i] = xin[i][0]
        turbineY[i] = xin[i][1]
        turbineZ[i] = xin[i][2]

    # set up problem
    prob = Problem()
    root = prob.root = Group()
    
    root.add('AEPGroup', AEPGroup(nTurbs, nDirections=nDirections,
                use_rotor_components=False, datasize=0, differentiable=True,
                optimizingLayout=False, nSamples=0), promotes=['*'])
    root.add('COEComponent', COEComponent(nTurbs), promotes=['*'])
    
    #root.ln_solver = ScipyGMRES()

    # initialize problem
    prob.setup()

    prob['turbineX'] = turbineX
    prob['turbineY'] = turbineY
    prob['turbineZ'] = turbineZ
    prob['yaw0'] = yaw

    # assign values to constant inputs (not design variables)
    prob['rotorDiameter'] = rotorDiameter
    prob['axialInduction'] = axialInduction
    prob['generatorEfficiency'] = generatorEfficiency
    prob['windSpeeds'] = np.array([wind_speed])
    prob['air_density'] = air_density
    #prob['windDirections'] = np.array([wind_direction])
    prob['windFrequencies'] = windFrequencies
    prob['Ct_in'] = Ct
    prob['Cp_in'] = Cp
    prob['floris_params:cos_spread'] = 1E12         # turns off cosine spread (just needs to be very large)

    # run the problem
    print 'start run'
    tic = time.time()
    prob.run()
    toc = time.time()

    first = prob['COE']

    print 'turbineZ: ', prob['turbineZ']
    print 'AEP: ', prob['AEP']
    print 'COE: ', prob['COE']
    tdCOE = prob['COE']  

    perDec = (tdCOE-optCOE)/tdCOE*100
    print('Percent Decrease: ', perDec)
