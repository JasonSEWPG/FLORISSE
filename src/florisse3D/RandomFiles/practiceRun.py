import numpy as np
import matplotlib.pyplot as plt
from FLORISSE3D.floris import AEPGroup
import matplotlib.pyplot as plt
from openmdao.api import Problem, Group, IndepVarComp, pyOptSparseDriver
import cPickle as pickle
from FLORISSE3D.setupOptimization import *
from FLORISSE3D.GeneralWindFarmComponents import get_z, getTurbineZ, AEPobj

if __name__=="__main__":

    rotor_diameter1 = 126.4
    rotor_diameter2 = 100.
    rotor_diameter = np.max(np.array([rotor_diameter1, rotor_diameter2]))

    """Grid Wind Farm"""
    nRows = 5
    nTurbs = nRows**2
    spacing = 2.  # turbine grid spacing in diameters
    points = np.linspace(start=spacing*rotor_diameter, stop=nRows*spacing*rotor_diameter, num=nRows)
    xpoints, ypoints = np.meshgrid(points, points)
    turbineX = np.ndarray.flatten(xpoints)
    turbineY = np.ndarray.flatten(ypoints)

    minSpacing = 2.

    rotorDiameter = np.zeros(nTurbs)
    axialInduction = np.zeros(nTurbs)
    Ct = np.zeros(nTurbs)
    Cp = np.zeros(nTurbs)
    generatorEfficiency = np.zeros(nTurbs)
    yaw = np.zeros(nTurbs)

    H1_H2 = np.array([])
    for i in range(nTurbs/2):
        H1_H2 = np.append(H1_H2, 0)
        H1_H2 = np.append(H1_H2, 1)
    if len(H1_H2) < nTurbs:
        H1_H2 = np.append(H1_H2, 0)

    for i in range(nTurbs):
        if H1_H2[i] == 0:
            rotorDiameter[i] = rotor_diameter1
        else:
            rotorDiameter[i] = rotor_diameter2

    # define initial values
    for turbI in range(0, nTurbs):

        # rotorDiameter[turbI] = rotor_diameter            # m
        axialInduction[turbI] = 1.0/3.0
        Ct[turbI] = 4.0*axialInduction[turbI]*(1.0-axialInduction[turbI])
        # Cp[turbI] = 0.7737/0.944 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
        Cp[turbI] = 0.7737 * 4.0 * 1.0/3.0 * np.power((1. - 1.0/3.0), 2)
        generatorEfficiency[turbI] = 1.0#0.944
        yaw[turbI] = 0.     # deg.

    """set up 3D aspects of wind farm"""
    """H1_H2 is an array telling which turbines are in each height group"""
    turbineH1 = 130.
    turbineH2 = 73.

    zref = 90.
    z0 = 0.

    """Define wind flow"""
    air_density = 1.1716    # kg/m^3


    """Amalia Wind Arrays: I have a separate file called setupOptimization that
    these are pulled from"""
    # windSpeeds, windFrequencies, windDirections, nDirections = amaliaWind()
    nDirections = 5
    speeds = np.array([12.,5.,10.,8.,12.])
    random = np.random.rand(5)*4. -2.
    windSpeeds = speeds + random*0
    print 'windSpeeds: ', windSpeeds
    windDirections = np.array([0.,72.,144.,216.,288.])
    windFrequencies = np.array([.1,.3,.2,.3,.1])

    """The wind shear exponent"""
    shearExp = 0.1



    """set up the problem"""
    prob = Problem()
    root = prob.root = Group()

    #This component takes turbineH1, turbineH2, and H1_H2 and gives back an array
    #of turbineZ
    # root.add('turbineX', IndepVarComp('turbineX', turbineX), promotes=['*'])
    # root.add('turbineY', IndepVarComp('turbineY', turbineY), promotes=['*'])
    root.add('getTurbineZ', getTurbineZ(nTurbs), promotes=['*'])

    root.add('AEPGroup', AEPGroup(nTurbs, nDirections=nDirections,
                use_rotor_components=False, datasize=0, differentiable=True,
                optimizingLayout=False, nSamples=0), promotes=['*'])
    root.add('AEPobj', AEPobj(), promotes=['*'])
    # add constraint definitions
    # root.add('spacing_con', ExecComp('sc = wtSeparationSquared-(minSpacing*rotorDiameter[0])**2',
    #                              minSpacing=minSpacing, rotorDiameter=np.zeros(nTurbs),
    #                              sc=np.zeros(((nTurbs-1.)*nTurbs/2.)),
    #                              wtSeparationSquared=np.zeros(((nTurbs-1.)*nTurbs/2.))),
    #                              promotes=['*'])
    #
    # prob.driver = pyOptSparseDriver()
    # prob.driver.options['optimizer'] = 'SNOPT'
    # prob.driver.opt_settings['Summary file'] = 'practice.out'
    # prob.driver.opt_settings['Major iterations limit'] = 1000
    # prob.driver.opt_settings['Major optimality tolerance'] = 1.0E-4
    # prob.driver.opt_settings['Function precision'] = 1.0E-8

    # --- Objective ---
    # prob.driver.add_objective('maxAEP', scaler=1.0E-9)
    #
    # # # --- Design Variables ---
    # prob.driver.add_desvar('turbineX', lower=np.ones(nTurbs)*min(turbineX), upper=np.ones(nTurbs)*max(turbineX), scaler=1.0E-3)
    # prob.driver.add_desvar('turbineY', lower=np.ones(nTurbs)*min(turbineY), upper=np.ones(nTurbs)*max(turbineY), scaler=1.0E-3)

    # --- Constraints ---
    # boundary constraint (convex hull)
    # prob.driver.add_constraint('boundaryDistances', lower=np.zeros(nVertices*nTurbs), scaler=1.0)
    # spacing constraint
    # prob.driver.add_constraint('sc', lower=np.zeros(((nTurbs-1.)*nTurbs/2.)), scaler=1.0/rotor_diameter)

    #Not sure what this is but it's in Jared's code
    prob.root.ln_solver.options['single_voi_relevance_reduction'] = True

    prob.setup()

    """run the problem"""
    prob['shearExp'] = shearExp
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
    prob['Uref'] = windSpeeds
    prob['Ct_in'] = Ct
    prob['Cp_in'] = Cp
    prob['floris_params:cos_spread'] = 1E12 # turns off cosine spread (just needs to be very large)
    prob['zref'] = zref
    prob['z0'] = z0

    prob.run()

    """print the results"""

    print 'AEP: ', prob['AEP']/1.E6, 'GWh'
    x = prob['turbineX']
    y = prob['turbineY']

    print 'rotor diameters: ', prob['rotorDiameter']

    # plt.plot(x,y,'ob')
    # plt.show()
