from florisse.COE import COEGroup
from florisse.GeneralWindFarmComponents import get_z, get_z_DEL, getTurbineZ, AEPobj
from florisse.floris import AEPGroup
from towerse.tower import TowerSE
import numpy as np
import matplotlib.pyplot as plt
from commonse.environment import PowerWind, LogWind
import matplotlib.pyplot as plt
from openmdao.api import Problem, Group, IndepVarComp
import time

"""This is an example run script that includes both COE and the tower model"""

if __name__=="__main__":
    rotor_diameter = 126.4
    rotor_diameter = 30.0

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
        spacing = 5   # turbine grid spacing in diameters
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

    minSpacing = 2

    """set up 3D aspects of wind farm"""
    turbineH1 = 75.
    turbineH2 = 76.
    H1_H2 = np.array([])
    for i in range(nTurbs/2):
        H1_H2 = np.append(H1_H2, 0)
        H1_H2 = np.append(H1_H2, 1)
    if len(H1_H2) < nTurbs:
        H1_H2 = np.append(H1_H2, 0)

    """Define wind flow"""
    air_density = 1.1716    # kg/m^3

    wind = "Amalia"

    """Amalia Wind Arrays"""
    if wind == "Amalia":
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

        nDirections = len(windSpeeds)
        windDirections = np.linspace(0,360-360/nDirections, nDirections)
        index = np.where(windSpeeds==0.0)
        windSpeeds = np.delete(windSpeeds, index[0])
        windFrequencies = np.delete(windFrequencies, index[0])
        windDirections = np.delete(windDirections, index[0])
        nDirections = len(windSpeeds)

    """Manual Wind Arrays"""
    if wind == "Manual":
        nDirections = 64
        windSpeeds = np.ones(nDirections)*10.
        print nDirections
        windDirections = np.linspace(0,360-360/nDirections, nDirections)
        windFrequencies = np.ones(len(windSpeeds))/len(windSpeeds)

    nIntegrationPoints = 50 #Number of points in wind effective wind speed integral

    """Define tower structural properties"""
    # --- geometry ----
    d_param = np.array([6.0, 4.935, 3.87])
    t_param = np.array([0.027*1.3, 0.023*1.3, 0.019*1.3])
    n = 15

    L_reinforced = 30.0*np.ones(n)  # [m] buckling length
    Toweryaw = 0.0

    # --- material props ---
    E = 210.e9*np.ones(n)
    G = 80.8e9*np.ones(n)
    rho = 8500.0*np.ones(n)
    sigma_y = 450.0e6*np.ones(n)

    # --- spring reaction data.  Use float('inf') for rigid constraints. ---
    kidx = np.array([0], dtype=int)  # applied at base
    kx = np.array([float('inf')])
    ky = np.array([float('inf')])
    kz = np.array([float('inf')])
    ktx = np.array([float('inf')])
    kty = np.array([float('inf')])
    ktz = np.array([float('inf')])
    nK = len(kidx)

    # --- extra mass ----
    midx = np.array([n-1], dtype=int)  # RNA mass at top
    m = np.array([285598.8]) *rotor_diameter**3/126.4**3 # mass scales with r^3
    mIxx = np.array([1.14930678e+08]) *rotor_diameter**5/126.4**5 # I scales with r^5
    mIyy = np.array([2.20354030e+07]) *rotor_diameter**5/126.4**5
    mIzz = np.array([1.87597425e+07]) *rotor_diameter**5/126.4**5
    mIxy = np.array([0.00000000e+00]) *rotor_diameter**5/126.4**5
    mIxz = np.array([5.03710467e+05]) *rotor_diameter**5/126.4**5
    mIyz = np.array([0.00000000e+00]) *rotor_diameter**5/126.4**5
    mrhox = np.array([-1.13197635]) # Does not change with rotor_diameter
    mrhoy = np.array([0.])
    mrhoz = np.array([0.50875268])
    nMass = len(midx)
    addGravityLoadForExtraMass = True
    # -----------

    # --- wind ---
    wind_zref = 90.0
    wind_z0 = 0.0
    shearExp = 0.2
    # ---------------

    # if addGravityLoadForExtraMass=True be sure not to double count by adding those force here also
    # # --- loading case 1: max Thrust ---
    wind_Uref1 = 11.73732
    plidx1 = np.array([n-1], dtype=int)  # at  top
    Fx1 = np.array([1284744.19620519]) *rotor_diameter**2/126.4**2 #scales with area (r^2)
    Fy1 = np.array([0.]) *rotor_diameter**2/126.4**2
    Fz1 = np.array([-2914124.84400512]) *rotor_diameter**2/126.4**2
    Mxx1 = np.array([3963732.76208099])
    Myy1 = np.array([-2275104.79420872])
    Mzz1 = np.array([-346781.68192839])
    nPL = len(plidx1)
    # # ---------------

    # # --- loading case 2: max wind speed ---
    wind_Uref2 = 70.0
    plidx2 = np.array([n-1], dtype=int)  # at  top
    Fx2 = np.array([930198.60063279]) *rotor_diameter**2/126.4**2
    Fy2 = np.array([0.]) *rotor_diameter**2/126.4**2
    Fz2 = np.array([-2883106.12368949]) *rotor_diameter**2/126.4**2
    Mxx2 = np.array([-1683669.22411597])
    Myy2 = np.array([-2522475.34625363])
    Mzz2 = np.array([147301.97023764])
    # # ---------------

    # --- safety factors ---
    gamma_f = 1.35
    gamma_m = 1.3
    gamma_n = 1.0
    gamma_b = 1.1
    # ---------------

    # --- fatigue ---
    z_DEL = np.array([0.000, 1.327, 3.982, 6.636, 9.291, 11.945, 14.600, 17.255, 19.909, 22.564, 25.218, 27.873, 30.527, 33.182, 35.836, 38.491, 41.145, 43.800, 46.455, 49.109, 51.764, 54.418, 57.073, 59.727, 62.382, 65.036, 67.691, 70.345, 73.000, 75.655, 78.309, 80.964, 83.618, 86.273, 87.600])
    M_DEL = 1e3*np.array([8.2940E+003, 8.1518E+003, 7.8831E+003, 7.6099E+003, 7.3359E+003, 7.0577E+003, 6.7821E+003, 6.5119E+003, 6.2391E+003, 5.9707E+003, 5.7070E+003, 5.4500E+003, 5.2015E+003, 4.9588E+003, 4.7202E+003, 4.4884E+003, 4.2577E+003, 4.0246E+003, 3.7942E+003, 3.5664E+003, 3.3406E+003, 3.1184E+003, 2.8977E+003, 2.6811E+003, 2.4719E+003, 2.2663E+003, 2.0673E+003, 1.8769E+003, 1.7017E+003, 1.5479E+003, 1.4207E+003, 1.3304E+003, 1.2780E+003, 1.2673E+003, 1.2761E+003])
    nDEL = len(z_DEL)
    gamma_fatigue = 1.35*1.3*1.0
    life = 20.0
    m_SN = 4
    # ---------------


    # --- constraints ---
    min_d_to_t = 120.0
    min_taper = 0.4
    # ---------------

    nPoints = len(d_param)
    nFull = n
    wind = 'PowerWind'

    less = 10.
    diff = 0.
    turbineH1 = 145.946032775 - less
    turbineH2 = 197.872130702 - less
    turbineH2 = turbineH1 - 28
    d_paramH1 = np.array([5.94766211,3.73152215,3.87])
    t_paramH1 = np.array([0.02298807,0.01719386,0.00681397])
    d_paramH2 = np.array([6.3,5.85876458,3.8699])
    t_paramH2 = np.array([0.03602141,0.01748397,0.00681483])

    """set up the problem"""
    prob = Problem()
    root = prob.root = Group()

    root.add('getTurbineZ', getTurbineZ(nTurbs), promotes=['*'])
    root.add('get_z_paramH1', get_z(nPoints))
    root.add('get_z_paramH2', get_z(nPoints))
    root.add('get_z_fullH1', get_z(n))
    root.add('get_z_fullH2', get_z(n))
    root.add('AEPGroup', AEPGroup(nTurbs, nDirections=nDirections,
                use_rotor_components=False, datasize=0, differentiable=True,
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

    prob.setup()

    prob['TowerH1.d_param'] = d_paramH1
    prob['TowerH1.t_param'] = t_paramH1
    prob['TowerH2.d_param'] = d_paramH2
    prob['TowerH2.t_param'] = t_paramH2
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
    prob['Ct_in'] = Ct
    prob['Cp_in'] = Cp
    prob['floris_params:cos_spread'] = 1E12
    prob['zref'] = wind_zref
    prob['z0'] = wind_z0       # turns off cosine spread (just needs to be very large)

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

    prob['TowerH2.wind2.Uref'] = wind_Uref2
    prob['TowerH2.tower2.plidx'] = plidx2
    prob['TowerH2.tower2.Fx'] = Fx2
    prob['TowerH2.tower2.Fy'] = Fy2
    prob['TowerH2.tower2.Fz'] = Fz2
    prob['TowerH2.tower2.Mxx'] = Mxx2
    prob['TowerH2.tower2.Myy'] = Myy2
    prob['TowerH2.tower2.Mzz'] = Mzz2
    # # ---------------

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

    prob.run()
    # ---------------
    print 'TurbineZ: ', prob['turbineZ']
    print 'AEP: ', prob['AEP']
    print 'COE: ', prob['COE']

    print 'H1: ', turbineH1
    print 'H2: ', turbineH2


    # print 'WindSpeeds: ', prob['windSpeeds']
    # for i in range(nDirections):
    #     print 'wtVelocity: ', prob['wtVelocity%i'%i]
    # loss = np.zeros((nTurbs, nDirections))
    # #print the loss of each turbine from each direction to test the wake model
    # for i in range(nTurbs):
    #     for j in range(nDirections):
    #         loss[i][j] = 1. - prob['wtVelocity%i'%j][i]/prob['windSpeeds'][i][j]
    # print 'Loss: ', loss

    for i in range(nTurbs):
        if H1_H2[i] == 0:
            plt.plot(turbineX[i], turbineY[i], 'bo')
        else:
            plt.plot(turbineX[i], turbineY[i], 'ro')
    bbox_props = dict(boxstyle="rarrow,pad=0.3", fc="cyan", ec="b", lw=2)
    t = plt.text(0, 0, " Wind Direction", ha="center", va="center", rotation=-windDirections[0]-90.,
            size=15, bbox=bbox_props)
    plt.axis([np.min(turbineX)-100, np.max(turbineX)+100, np.min(turbineY)-100, np.max(turbineY)+100])
    # plt.show()


    # print 'Tower H1 Mass: ', prob['TowerH1.tower1.mass']
    # print 'Tower H2 Mass: ', prob['TowerH2.tower1.mass']
    #
    # print 'Tower H1 Stress: Max Thrust: ', prob['TowerH1.tower1.stress']
    # print 'Tower H1 Stress: Max Wind Speed: ', prob['TowerH1.tower2.stress']
    # print 'Tower H2 Stress: Max Thrust: ', prob['TowerH2.tower1.stress']
    # print 'Tower H2 Stress: Max Wind Speed: ', prob['TowerH2.tower2.stress']
    #
    # print 'Tower H1 Shell Buckling: Max Thrust: ', prob['TowerH1.tower1.shell_buckling']
    # print 'Tower H1 Shell Buckling: Max Wind Speed: ', prob['TowerH1.tower2.shell_buckling']
    # print 'Tower H2 Shell Buckling: Max Thrust: ', prob['TowerH2.tower1.shell_buckling']
    # print 'Tower H2 Shell Buckling: Max Wind Speed: ', prob['TowerH2.tower2.shell_buckling']
    #
    # print 'Tower H1 Global Buckling: Max Thrust: ', prob['TowerH1.tower1.global_buckling']
    # print 'Tower H1 Global Buckling: Max Wind Speed: ', prob['TowerH1.tower2.global_buckling']
    # print 'Tower H2 Global Buckling: Max Thrust: ', prob['TowerH2.tower1.global_buckling']
    # print 'Tower H2 Global Buckling: Max Wind Speed: ', prob['TowerH2.tower2.global_buckling']
    #
    # print 'Tower H1 Damage: ', prob['TowerH1.tower1.damage']
    # print 'Tower H2 Damage: ', prob['TowerH2.tower1.damage']
