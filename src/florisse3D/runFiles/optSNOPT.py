from FLORISSE3D.COE import COEGroup
from towerse.tower import TowerSE
from FLORISSE3D.GeneralWindFarmComponents import calculate_boundary, SpacingComp,\
            BoundaryComp, get_z, get_z_DEL, getTurbineZ, AEPobj, speedFreq, actualSpeeds
import numpy as np
import matplotlib.pyplot as plt
from FLORISSE3D.floris import AEPGroup
from commonse.environment import PowerWind, LogWind
import matplotlib.pyplot as plt
from openmdao.api import Problem, Group, IndepVarComp, pyOptSparseDriver, ExecComp, ScipyOptimizer
import time
import cPickle as pickle
from setupOptimization import *
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
    length_data = np.linspace(0.,len(probability)+0.01,len(probability))
    f = interp1d(length_data, probability)
    return f

def wind_frequency_funcion(frequencies):

    probability = frequencies
    length_data = np.linspace(0.,len(probability)+0.01,len(probability))
    f = interp1d(length_data, probability)
    return f, len(probability)

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
    diff = 0.
    turbineH1 = 75.
    turbineH2 = 120.
    H1_H2 = np.array([])
    for i in range(nTurbs/2):
        H1_H2 = np.append(H1_H2, 0)
        H1_H2 = np.append(H1_H2, 1)
    if len(H1_H2) < nTurbs:
        H1_H2 = np.append(H1_H2, 0)
    #H1_H2 = np.zeros(nTurbs)

    """Define wind flow"""
    air_density = 1.1716    # kg/m^3

    windData = "Amalia"

    """Amalia Wind Arrays"""
    if windData == "Amalia":
        windSpeeds, windFrequencies, windDirections, nDirections = amaliaWind()

    """!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""
    # bins = 13
    # windFrequencies2, windSpeeds2 = frequ(bins, windFrequencies1, windSpeeds1)
    # nDirections = len(windSpeeds2)
    # windDirections2 = np.linspace(0.,360.-360./nDirections, nDirections)
    #
    # "Multiple wind speeds"""
    # num = 10
    # speedFreqs = speedFreq(num)
    # speeds = np.zeros((len(windSpeeds2)*num))
    # newWindDirections = np.zeros(num*len(windSpeeds2))
    # newWindFrequencies = np.zeros(num*len(windSpeeds2))
    #
    # for i in range(len(windSpeeds2)):
    #     speeds[i*num:i*num+num] = actualSpeeds(num, windSpeeds2[i])
    #     newWindDirections[i*num:i*num+num] = windDirections2[i]
    #     newWindFrequencies[i*num:i*num+num] = windFrequencies2[i]*speedFreqs
    #
    # windDirections = newWindDirections
    # if num == 1:
    #     speeds = windSpeeds
    # windSpeeds = speeds
    # windFrequencies = newWindFrequencies
    # nDirections = len(windSpeeds)
    """!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""

    """Define tower structural properties"""
    # --- geometry ----
    d_param = np.array([6.0, 4.935, 3.87])
    t_param = np.array([0.027*1.3, 0.023*1.3, 0.019*1.3])
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

    shearExp = float(argv[1])

    nRows = 5
    nTurbs = nRows**2
    # spacing = space[i]   # turbine grid spacing in diameters
    spacing = 3.
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

    # opt_filenameXYZ = 'SP3XYZ_same_%s.txt'%shearExp
    # opt_filename_dt = 'SP3dt_same_%s.txt'%shearExp

    # optXYZ = open(opt_filenameXYZ)
    # optimizedXYZ = np.loadtxt(optXYZ)
    # turbineX = optimizedXYZ[:,0]
    # turbineY = optimizedXYZ[:,1]
    # turbineZ = optimizedXYZ[:,2]
    # turbineH1 = turbineZ[0]
    # turbineH2 = turbineZ[1]
    # opt_dt = open(opt_filename_dt)
    # optimized_dt = np.loadtxt(opt_dt)
    # d_paramH1 = optimized_dt[:,0]
    # d_paramH2 = optimized_dt[:,1]
    # t_paramH1 = optimized_dt[:,2]
    # t_paramH2 = optimized_dt[:,3]

    """set up the problem"""
    prob = Problem()
    root = prob.root = Group()

    root.deriv_options['type'] = 'fd'
    root.deriv_options['form'] = 'central'
    root.deriv_options['step_size'] = 1.E-4
    root.deriv_options['step_type'] = 'relative'

    root.add('turbineH1', IndepVarComp('turbineH1', turbineH1), promotes=['*'])
    root.add('turbineH2', IndepVarComp('turbineH2', turbineH2), promotes=['*'])
    root.add('d_paramH1', IndepVarComp('d_paramH1', d_param), promotes=['*'])
    root.add('t_paramH1', IndepVarComp('t_paramH1', t_param), promotes=['*'])
    root.add('d_paramH2', IndepVarComp('d_paramH2', d_param), promotes=['*'])
    root.add('t_paramH2', IndepVarComp('t_paramH2', t_param), promotes=['*'])
    #This component takes turbineH1, turbineH2, and H1_H2 and gives back an array
    #of turbineZ
    root.add('getTurbineZ', getTurbineZ(nTurbs), promotes=['*'])
    #These components adjust the parameterized z locations for TowerSE calculations
    #with respect to turbineZ
    root.add('get_z_paramH1', get_z(nPoints))
    root.add('get_z_paramH2', get_z(nPoints))
    root.add('get_z_fullH1', get_z(n))
    root.add('get_z_fullH2', get_z(n))
    root.add('AEPGroup', AEPGroup(nTurbs, nDirections=nDirections,
                use_rotor_components=use_rotor_components, datasize=datasize, differentiable=True,
                optimizingLayout=False, nSamples=0), promotes=['*'])
    root.add('COEGroup', COEGroup(nTurbs), promotes=['*'])
    root.add('maxAEP', AEPobj(), promotes=['*'])

    #For Constraints
    root.add('TowerH1', TowerSE(nPoints, nFull, nK, nMass, nPL, nDEL, wind=wind), promotes=['L_reinforced',
                        'E','G','sigma_y','kidx','kx','ky','kz','ktx','kty',
                        'ktz','midx','m','mIxx','mIyy','mIzz','mIxy','mIxz','mIyz',
                        'mrhox','mrhoy','mrhoz','addGravityLoadForExtraMass','g',
                        'gamma_f','gamma_m','gamma_n','gamma_b','life','m_SN',
                        'gc.min_d_to_t','gc.min_taper','M_DEL','gamma_fatigue'])
    root.add('TowerH2', TowerSE(nPoints, nFull, nK, nMass, nPL, nDEL, wind=wind), promotes=['L_reinforced',
                        'E','G','sigma_y','kidx','kx','ky','kz','ktx','kty',
                        'ktz','midx','m','mIxx','mIyy','mIzz','mIxy','mIxz','mIyz',
                        'mrhox','mrhoy','mrhoz','addGravityLoadForExtraMass','g',
                        'gamma_f','gamma_m','gamma_n','gamma_b','life','m_SN',
                        'gc.min_d_to_t','gc.min_taper','M_DEL','gamma_fatigue'])
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

    root.connect('turbineH1', 'get_z_paramH1.turbineZ')
    root.connect('turbineH2', 'get_z_paramH2.turbineZ')
    root.connect('turbineH1', 'get_z_fullH1.turbineZ')
    root.connect('turbineH2', 'get_z_fullH2.turbineZ')
    root.connect('get_z_paramH1.z_param', 'TowerH1.z_param')
    root.connect('get_z_fullH1.z_param', 'TowerH1.z_full')
    root.connect('get_z_paramH2.z_param', 'TowerH2.z_param')
    root.connect('get_z_fullH2.z_param', 'TowerH2.z_full')
    root.connect('TowerH1.tower1.mass', 'mass1')
    root.connect('TowerH2.tower1.mass', 'mass2')
    root.connect('d_paramH1', 'TowerH1.d_param')
    root.connect('d_paramH2', 'TowerH2.d_param')
    root.connect('t_paramH1', 'TowerH1.t_param')
    root.connect('t_paramH2', 'TowerH2.t_param')

    prob.driver = pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'SNOPT'
    prob.driver.opt_settings['Summary file'] = 'SNOPT_%s.out'%shearExp
    prob.driver.opt_settings['Major iterations limit'] = 1000
    prob.driver.opt_settings['Major optimality tolerance'] = 1.0E-4
    prob.driver.opt_settings['Function precision'] = 1.0E-8
    # prob.driver = ScipyOptimizer()
    # prob.driver.options['optimizer'] = 'SLSQP'

    # --- Objective ---
    prob.driver.add_objective('COE', scaler=1.0E-1)

    # # --- Design Variables ---
    prob.driver.add_desvar('turbineH1', lower=rotor_diameter/2.+10, upper=160., scaler=1.0E-1)
    prob.driver.add_desvar('turbineH2', lower=rotor_diameter/2.+10, upper=160., scaler=1.0E-1)
    prob.driver.add_desvar('turbineX', lower=np.ones(nTurbs)*min(turbineX), upper=np.ones(nTurbs)*max(turbineX), scaler=1.0E-3)
    prob.driver.add_desvar('turbineY', lower=np.ones(nTurbs)*min(turbineY), upper=np.ones(nTurbs)*max(turbineY), scaler=1.0E-3)
    prob.driver.add_desvar('d_paramH1', lower=np.array([1.0, 1.0, d_param[nPoints-1]]), upper=np.ones(nPoints)*6.3, scaler=1.0E-1)
    prob.driver.add_desvar('t_paramH1', lower=np.ones(nPoints)*.001, upper=np.ones(nPoints)*0.05, scaler=1.0E1)
    prob.driver.add_desvar('d_paramH2', lower=np.array([1.0, 1.0, d_param[nPoints-1]]), upper=np.ones(nPoints)*6.3, scaler=1.0E-1)
    prob.driver.add_desvar('t_paramH2', lower=np.ones(nPoints)*.001, upper=np.ones(nPoints)*0.05, scaler=1.0E1)

    # --- Constraints ---
    #TowerH1 structure
    prob.driver.add_constraint('TowerH1.tower1.stress', upper=1.0)
    prob.driver.add_constraint('TowerH1.tower2.stress', upper=1.0)
    prob.driver.add_constraint('TowerH1.tower1.global_buckling', upper=1.0)
    prob.driver.add_constraint('TowerH1.tower2.global_buckling', upper=1.0)
    prob.driver.add_constraint('TowerH1.tower1.shell_buckling', upper=1.0)
    prob.driver.add_constraint('TowerH1.tower2.shell_buckling', upper=1.0)
    prob.driver.add_constraint('TowerH1.tower1.damage', upper=1.0)
    prob.driver.add_constraint('TowerH1.gc.weldability', upper=0.0)
    prob.driver.add_constraint('TowerH1.gc.manufacturability', upper=0.0)
    freq1p = 0.2  # 1P freq in Hz
    prob.driver.add_constraint('TowerH1.tower1.f1', lower=1.1*freq1p)

    #TowerH2 structure
    prob.driver.add_constraint('TowerH2.tower1.stress', upper=1.0)
    prob.driver.add_constraint('TowerH2.tower2.stress', upper=1.0)
    prob.driver.add_constraint('TowerH2.tower1.global_buckling', upper=1.0)
    prob.driver.add_constraint('TowerH2.tower2.global_buckling', upper=1.0)
    prob.driver.add_constraint('TowerH2.tower1.shell_buckling', upper=1.0)
    prob.driver.add_constraint('TowerH2.tower2.shell_buckling', upper=1.0)
    prob.driver.add_constraint('TowerH2.gc.weldability', upper=0.0)
    prob.driver.add_constraint('TowerH2.gc.manufacturability', upper=0.0)
    freq1p = 0.2  # 1P freq in Hz
    prob.driver.add_constraint('TowerH2.tower1.f1', lower=1.1*freq1p)

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
        prob['TowerH1.wind1.shearExp'] = shearExp
        prob['TowerH1.wind2.shearExp'] = shearExp
        prob['TowerH2.wind1.shearExp'] = shearExp
        prob['TowerH2.wind2.shearExp'] = shearExp
        prob['shearExp'] = shearExp
    prob['turbineH1'] = turbineH1
    prob['turbineH2'] = turbineH2
    prob['H1_H2'] = H1_H2
    # prob['diameter'] = rotor_diameter

    prob['turbineX'] = turbineX
    prob['turbineY'] = turbineY
    prob['yaw0'] = yaw
    prob['ratedPower'] = np.ones_like(turbineX)*5000 # in kw

    prob['boundaryVertices'] = boundaryVertices
    prob['boundaryNormals'] = boundaryNormals

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
    prob['floris_params:cos_spread'] = 1E12 # turns off cosine spread (just needs to be very large)
    prob['zref'] = wind_zref
    prob['z0'] = wind_z0

    """tower structural properties"""
    # --- geometry ----
    # prob['d_param'] = d_param
    # prob['t_param'] = t_param

    prob['L_reinforced'] = L_reinforced
    prob['TowerH1.yaw'] = Toweryaw
    prob['TowerH2.yaw'] = Toweryaw

    # --- material props ---
    prob['E'] = E
    prob['G'] = G
    prob['TowerH1.tower1.rho'] = rho
    prob['TowerH2.tower1.rho'] = rho
    prob['sigma_y'] = sigma_y

    # --- spring reaction data.  Use float('inf') for rigid constraints. ---
    prob['kidx'] = kidx
    prob['kx'] = kx
    prob['ky'] = ky
    prob['kz'] = kz
    prob['ktx'] = ktx
    prob['kty'] = kty
    prob['ktz'] = ktz

    # --- extra mass ----
    prob['midx'] = midx
    prob['m'] = m
    prob['mIxx'] = mIxx
    prob['mIyy'] = mIyy
    prob['mIzz'] = mIzz
    prob['mIxy'] = mIxy
    prob['mIxz'] = mIxz
    prob['mIyz'] = mIyz
    prob['mrhox'] = mrhox
    prob['mrhoy'] = mrhoy
    prob['mrhoz'] = mrhoz
    prob['addGravityLoadForExtraMass'] = addGravityLoadForExtraMass
    # -----------

    # --- wind ---
    prob['TowerH1.zref'] = wind_zref
    prob['TowerH2.zref'] = wind_zref
    prob['TowerH1.z0'] = wind_z0
    prob['TowerH2.z0'] = wind_z0
    # ---------------

    # # --- loading case 1: max Thrust ---
    prob['TowerH1.wind1.Uref'] = wind_Uref1
    prob['TowerH1.tower1.plidx'] = plidx1
    prob['TowerH1.tower1.Fx'] = Fx1
    prob['TowerH1.tower1.Fy'] = Fy1
    prob['TowerH1.tower1.Fz'] = Fz1
    prob['TowerH1.tower1.Mxx'] = Mxx1
    prob['TowerH1.tower1.Myy'] = Myy1
    prob['TowerH1.tower1.Mzz'] = Mzz1
    #
    prob['TowerH2.wind1.Uref'] = wind_Uref1
    prob['TowerH2.tower1.plidx'] = plidx1
    prob['TowerH2.tower1.Fx'] = Fx1
    prob['TowerH2.tower1.Fy'] = Fy1
    prob['TowerH2.tower1.Fz'] = Fz1
    prob['TowerH2.tower1.Mxx'] = Mxx1
    prob['TowerH2.tower1.Myy'] = Myy1
    prob['TowerH2.tower1.Mzz'] = Mzz1
    # # ---------------

    # # --- loading case 2: max Wind Speed ---
    prob['TowerH1.wind2.Uref'] = wind_Uref2
    prob['TowerH1.tower2.plidx'] = plidx2
    prob['TowerH1.tower2.Fx'] = Fx2
    prob['TowerH1.tower2.Fy'] = Fy2
    prob['TowerH1.tower2.Fz'] = Fz2
    prob['TowerH1.tower2.Mxx'] = Mxx2
    prob['TowerH1.tower2.Myy'] = Myy2
    prob['TowerH1.tower2.Mzz'] = Mzz2
    #
    prob['TowerH2.wind2.Uref'] = wind_Uref2
    prob['TowerH2.tower2.plidx'] = plidx2
    prob['TowerH2.tower2.Fx'] = Fx2
    prob['TowerH2.tower2.Fy'] = Fy2
    prob['TowerH2.tower2.Fz'] = Fz2
    prob['TowerH2.tower2.Mxx'] = Mxx2
    prob['TowerH2.tower2.Myy'] = Myy2
    prob['TowerH2.tower2.Mzz'] = Mzz2
    # # # ---------------

    # --- safety factors ---
    prob['gamma_f'] = gamma_f
    prob['gamma_m'] = gamma_m
    prob['gamma_n'] = gamma_n
    prob['gamma_b'] = gamma_b
    # ---------------

    # --- fatigue ---
    prob['gamma_fatigue'] = gamma_fatigue
    prob['life'] = life
    prob['m_SN'] = m_SN
    # ---------------

    # --- constraints ---
    prob['gc.min_d_to_t'] = min_d_to_t
    prob['gc.min_taper'] = min_taper
    # ---------------


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

    np.savetxt('XYZ_%s.txt'%shearExp, np.c_[prob['turbineX'], prob['turbineY'], prob['turbineZ']], header="turbineX, turbineY, turbineZ")
    np.savetxt('dt_%s.txt'%shearExp, np.c_[prob['d_paramH1'], prob['d_paramH2'], prob['t_paramH1'], prob['t_paramH2'],], header="d_paramH1, t_paramH1")
