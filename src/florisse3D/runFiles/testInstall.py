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
    spacing = 3.  # turbine grid spacing in diameters
    points = np.linspace(start=spacing*rotor_diameter, stop=nRows*spacing*rotor_diameter, num=nRows)
    xpoints, ypoints = np.meshgrid(points, points)
    turbineX = np.ndarray.flatten(xpoints)
    turbineY = np.ndarray.flatten(ypoints)

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

    # generate boundary constraint
    locations = np.zeros((len(turbineX),2))
    for i in range(len(turbineX)):
        locations[i][0] = turbineX[i]
        locations[i][1] = turbineY[i]
    print locations
    boundaryVertices, boundaryNormals = calculate_boundary(locations)
    nVertices = boundaryVertices.shape[0]

    """set up 3D aspects of wind farm"""
    H1_H2 = np.array([])
    for i in range(nTurbs/2):
        H1_H2 = np.append(H1_H2, 0)
        H1_H2 = np.append(H1_H2, 1)
    if len(H1_H2) < nTurbs:
        H1_H2 = np.append(H1_H2, 0)
    turbineH1 = 90.
    turbineH2 = 120.
    wind_zref = 90.
    wind_z0 = 0.

    """Define wind flow"""
    air_density = 1.1716    # kg/m^3

    # windData = "Amalia"
    #
    # """Amalia Wind Arrays"""
    # if windData == "Amalia":
    #     windSpeeds, windFrequencies, windDirections, nDirections = amaliaWind()

    windSpeeds = np.array([10.])
    windDirections = np.array([0.])
    windFrequencies = np.array([1.])
    nDirections = len(windSpeeds)

    nIntegrationPoints = 1 #Number of points in wind effective wind speed integral

    wind = 'PowerWind'

    shearExp = float(argv[1])

    """set up the problem"""
    prob = Problem()
    root = prob.root = Group()

    # root.deriv_options['type'] = 'fd'
    # root.deriv_options['form'] = 'central'
    # root.deriv_options['step_size'] = 1.E-4
    # root.deriv_options['step_type'] = 'relative'

    root.add('turbineH1', IndepVarComp('turbineH1', turbineH1), promotes=['*'])
    root.add('turbineH2', IndepVarComp('turbineH2', turbineH2), promotes=['*'])
    #This component takes turbineH1, turbineH2, and H1_H2 and gives back an array
    #of turbineZ
    root.add('getTurbineZ', getTurbineZ(nTurbs), promotes=['*'])
    #These components adjust the parameterized z locations for TowerSE calculations
    #with respect to turbineZ
    root.add('AEPGroup', AEPGroup(nTurbs, nDirections=nDirections,
                use_rotor_components=use_rotor_components, datasize=datasize, differentiable=True,
                optimizingLayout=False, nSamples=0), promotes=['*'])
    root.add('maxAEP', AEPobj(), promotes=['*'])

    root.add('spacing_comp', SpacingComp(nTurbines=nTurbs), promotes=['*'])

    # add constraint definitions
    root.add('spacing_con', ExecComp('sc = wtSeparationSquared-(minSpacing*rotorDiameter[0])**2',
                                 minSpacing=minSpacing, rotorDiameter=np.zeros(nTurbs),
                                 sc=np.zeros(((nTurbs-1.)*nTurbs/2.)),
                                 wtSeparationSquared=np.zeros(((nTurbs-1.)*nTurbs/2.))),
                                 promotes=['*'])

    if nVertices > 0:
        # add component that enforces a convex hull wind farm boundary
        root.add('boundary_con', BoundaryComp(nVertices=nVertices, nTurbines=nTurbs), promotes=['*'])


    prob.driver = pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'SNOPT'
    prob.driver.opt_settings['Summary file'] = 'test_%s.out'%shearExp
    prob.driver.opt_settings['Major iterations limit'] = 1000
    prob.driver.opt_settings['Major optimality tolerance'] = 1.E-4
    prob.driver.opt_settings['Function precision'] = 1.E-8

    # --- Objective ---
    prob.driver.add_objective('maxAEP', scaler=1.0E-5)

    # # --- Design Variables ---
    prob.driver.add_desvar('turbineH1', lower=rotor_diameter/2.+10., upper=100., scaler=1.0E-1)
    prob.driver.add_desvar('turbineH2', lower=rotor_diameter/2.+10., upper=100., scaler=1.0E-1)
    prob.driver.add_desvar('turbineX', lower=np.ones(nTurbs)*min(turbineX), upper=np.ones(nTurbs)*max(turbineX), scaler=1.0E-3)
    prob.driver.add_desvar('turbineY', lower=np.ones(nTurbs)*min(turbineY), upper=np.ones(nTurbs)*max(turbineY), scaler=1.0E-3)

    # --- Constraints ---
    # boundary constraint (convex hull)
    prob.driver.add_constraint('boundaryDistances', lower=np.zeros(nVertices*nTurbs), scaler=1.0)
    # spacing constraint
    prob.driver.add_constraint('sc', lower=np.zeros(((nTurbs-1.)*nTurbs/2.)), scaler=1.0/rotor_diameter)

    # ----------------------

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

    prob['boundaryVertices'] = boundaryVertices
    prob['boundaryNormals'] = boundaryNormals

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

    """print the results"""

    print 'Turbine H1: ', prob['turbineH1']
    print 'Turbine H2: ', prob['turbineH2']
    print 'Rotor Diameter: ', rotor_diameter
    print 'H1_H2: ', prob['H1_H2']
    print 'TurbineX: ', prob['turbineX']
    print 'TurbineY: ', prob['turbineY']

    print 'AEP: ', prob['AEP']

    print 'nDirections: ', nDirections

    x = prob['turbineX']
    y = prob['turbineY']

    plt.plot(x,y,'ob')
    plt.show()
