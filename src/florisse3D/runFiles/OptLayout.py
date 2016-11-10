from florisse.COE import COEGroup
from towerse.tower import TowerSE
from florisse.GeneralWindFarmComponents import calculate_boundary, SpacingComp,\
            BoundaryComp, get_z, getTurbineZ, AEPobj, speedFreq, actualSpeeds
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

    """set up the wind farm"""
    farm = "Grid"

    """Amalia Wind Farm"""
    if farm == "Amalia":
        filename = "layout_amalia.txt"
        amalia = open(filename)
        x_y = np.loadtxt(amalia)
        turbineX = x_y[:,0]
        turbineY = x_y[:,1]
        nTurbs = len(turbineX)

    """Manual Wind Farm"""
    if farm == "Manual":
        turbineX = np.array([50,50,50,50,50,50,50])
        turbineY = np.array([50,1050,2050,3050,4050,5050,6050])
        nTurbs = len(turbineX)

    """Grid Wind Farm"""
    if farm == "Grid":
        nRows = 5
        nTurbs = nRows**2
        spacing = 3.0  # turbine grid spacing in diameters
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
        Cp[turbI] = 0.7737 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
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
    diff = 0.
    turbineH1 = 90.
    turbineH2 = 90.
    H1_H2 = np.array([])
    for i in range(nTurbs/2):
        H1_H2 = np.append(H1_H2, 0)
        H1_H2 = np.append(H1_H2, 1)
    if len(H1_H2) < nTurbs:
        H1_H2 = np.append(H1_H2, 0)
    H1_H2 = np.zeros(nTurbs)

    """Define wind flow"""
    air_density = 1.1716    # kg/m^3

    windData = "Amalia"

    """Amalia Wind Arrays"""
    if windData == "Amalia":
        windSpeeds1, windFrequencies1, windDirections1, nDirections1 = amaliaWind()

    """!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""
    bins = 13
    windFrequencies2, windSpeeds2 = frequ(bins, windFrequencies1, windSpeeds1)
    nDirections = len(windSpeeds2)
    windDirections2 = np.linspace(0,360-360/nDirections, nDirections)

    "Multiple wind speeds"""
    num = 10
    speedFreqs = speedFreq(num)
    speeds = np.zeros((len(windSpeeds2)*num))
    newWindDirections = np.zeros(num*len(windSpeeds2))
    newWindFrequencies = np.zeros(num*len(windSpeeds2))

    for i in range(len(windSpeeds2)):
        speeds[i*num:i*num+num] = actualSpeeds(num, windSpeeds2[i])
        newWindDirections[i*num:i*num+num] = windDirections2[i]
        newWindFrequencies[i*num:i*num+num] = windFrequencies2[i]*speedFreqs

    windDirections = newWindDirections
    if num == 1:
        speeds = windSpeeds
    windSpeeds = speeds
    windFrequencies = newWindFrequencies
    nDirections = len(windSpeeds)
    """!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""

    print 'Wind Speeds: ', windSpeeds
    print 'Wind Frequencies: ', windFrequencies

    nIntegrationPoints = 1 #Number of points in wind effective wind speed integral

    """Define tower structural properties"""
    # --- geometry ----
    d_param = np.array([6.0, 4.935, 3.87])
    # d_param = np.array([4.6, 4.5, 3.87])
    t_param = np.array([0.027*1.3, 0.023*1.3, 0.019*1.3])
    d_paramH1 = np.array([6.0, 4.935, 3.87])
    d_paramH2 = np.array([6.0, 4.935, 3.87])
    t_paramH1 = np.array([0.027*1.3, 0.023*1.3, 0.019*1.3])
    t_paramH2 = np.array([0.027*1.3, 0.023*1.3, 0.019*1.3])

    n = 15

    L_reinforced, Toweryaw, E, G, rho, sigma_y, kidx, kx, ky, kz, ktx, kty, ktz, nK, \
                midx, m, mIxx, mIyy, mIzz, mIxy, mIxz, mIyz, mrhox, mrhoy, mrhoz, nMass, \
                addGravityLoadForExtraMass, wind_zref, wind_z0, shearExp, wind_Uref1, \
                plidx1, Fx1, Fy1, Fz1, Mxx1, Myy1, Mzz1, nPL, wind_Uref2, \
                plidx2, Fx2, Fy2, Fz2, Mxx2, Myy2, Mzz2, gamma_f, gamma_m, gamma_n, \
                gamma_b, z_DEL, M_DEL, nDEL, gamma_fatigue, life, m_SN, min_d_to_t, min_taper \
                = setupTower(n)

    nPoints = len(d_param)
    nFull = n
    wind = 'PowerWind'

    nRows = 5
    nTurbs = nRows**2
    spacing = 3.0
    points = np.linspace(start=spacing*rotor_diameter, stop=nRows*spacing*rotor_diameter, num=nRows)
    xpoints, ypoints = np.meshgrid(points, points)
    turbineX = np.ndarray.flatten(xpoints)
    turbineY = np.ndarray.flatten(ypoints)

    # generate boundary constraint
    locations = np.zeros((len(turbineX),2))
    for i in range(len(turbineX)):
        locations[i][0] = turbineX[i]
        locations[i][1] = turbineY[i]
    print locations
    boundaryVertices, boundaryNormals = calculate_boundary(locations)
    nVertices = boundaryVertices.shape[0]

    for i in range(1):
        """set up the problem"""
        shearExp = 0.1
        prob = Problem()
        root = prob.root = Group()

        root.deriv_options['step_size'] = 1.E-4
        root.deriv_options['step_type'] = 'relative'
        root.deriv_options['type'] = 'fd'
        root.deriv_options['form'] = 'central'

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

        #For Constraints
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
        prob.driver.opt_settings['Summary file'] = 'SNOPT_LAYOUT.out'
        prob.driver.opt_settings['Major iterations limit'] = 1000
        prob.driver.opt_settings['Minor iterations limit'] = 1000
        prob.driver.opt_settings['Major optimality tolerance'] = 1E-4
	prob.driver.opt_settings['Function precision'] = 1E-12

        # --- Objective ---
        prob.driver.add_objective('maxAEP', scaler=1.0E1)

        # # --- Design Variables ---

        prob.driver.add_desvar('turbineX', lower=np.ones(nTurbs)*min(turbineX), upper=np.ones(nTurbs)*max(turbineX), scaler=1.0E-3)
        prob.driver.add_desvar('turbineY', lower=np.ones(nTurbs)*min(turbineY), upper=np.ones(nTurbs)*max(turbineY), scaler=1.0E-3)


        # # --- Constraints ---

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
        prob['ratedPower'] = np.ones_like(turbineX)*5000 # in kw

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
        prob['floris_params:cos_spread'] = 1E12
        prob['zref'] = wind_zref
        prob['z0'] = wind_z0       # turns off cosine spread (just needs to be very large)

        """tower structural properties"""

        prob.run()

        """print the results"""

        print 'Turbine H1: ', prob['turbineH1']
        print 'Turbine H2: ', prob['turbineH2']
        print 'Rotor Diameter: ', rotor_diameter
        print 'H1_H2: ', prob['H1_H2']
        print 'TurbineX: ', prob['turbineX']
        print 'TurbineY: ', prob['turbineY']


        print 'Tower Mass H1: ', prob['TowerH1.tower1.mass']
        print 'Tower Mass H2: ', prob['TowerH2.tower1.mass']
        print 'd_param H1: ', prob['d_paramH1']
        print 't_param H1: ', prob['t_paramH1']
        print 'd_param H2: ', prob['d_paramH2']
        print 't_param H2: ', prob['t_paramH2']

        print 'COE: ', prob['COE']
        print 'AEP: ', prob['AEP']

        print 'z_param H1: ', prob['TowerH1.z_param']
        print 'z_full H1: ', prob['TowerH1.z_full']
        print 'nDirections: ', nDirections
        print 'Time to run: ', time.time() - start

        #spacing_nRows_shearCoeff_optimizer_minspacing_maxDiameter.txt
        np.savetxt('XYZ_LAYOUT.txt', np.c_[prob['turbineX'], prob['turbineY'], prob['turbineZ']], header="turbineX, turbineY, turbineZ")
        np.savetxt('dt_LAYOUT.txt', np.c_[prob['d_paramH1'], prob['d_paramH2'], prob['t_paramH1'], prob['t_paramH2']], header="d_paramH1, d_paramH2, t_paramH1, t_paramH2")
        #
        # turbineX = prob['turbineX']
        # turbineY = prob['turbineY']
        # d_paramH1 = prob['d_paramH1']
        # d_paramH2 = prob['d_paramH2']
        # t_paramH1 = prob['t_paramH1']
        # t_paramH2 = prob['t_paramH2']
        # turbineH1 = prob['turbineH1']
        # turbineH2 = prob['turbineH2']
