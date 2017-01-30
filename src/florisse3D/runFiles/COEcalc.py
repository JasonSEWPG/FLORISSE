from FLORISSE3D.COE import COEGroup
from FLORISSE3D.GeneralWindFarmComponents import get_z, get_z_DEL, getTurbineZ, AEPobj, actualSpeeds, speedFreq
from FLORISSE3D.floris import AEPGroup
from towerse.tower import TowerSE
import numpy as np
import matplotlib.pyplot as plt
from commonse.environment import PowerWind, LogWind
import matplotlib.pyplot as plt
from openmdao.api import Problem, Group, IndepVarComp
import time
from setupOptimization import *
import cPickle as pickle
from scipy.interpolate import interp1d


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


"""This is an example run script that includes both COE and the tower model"""

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
    farm = "Gr"

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
        spacing = 5  # turbine grid spacing in diameters
        points = np.linspace(start=spacing*rotor_diameter, stop=nRows*spacing*rotor_diameter, num=nRows)
        xpoints, ypoints = np.meshgrid(points, points)
        turbineX = np.ndarray.flatten(xpoints)
        turbineY = np.ndarray.flatten(ypoints)


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
        # windSpeeds1, windFrequencies1, windDirections1, nDirections = amaliaWind()
        windSpeeds, windFrequencies, windDirections, nDirections = amaliaWind()

    """Manual Wind Arrays"""
    if windData == "Manual":
        nDirections = 8
        windSpeeds = np.array([12.,11.25,10.5,9.75,9.,8.25,7.5,6.75])
        windDirections = np.array([225.,270.,315.,0.,45.,90.,135.,180.])
        windFrequencies = np.array([2./9.,7./36.,3./18.,5./36.,1./9.,1./12.,1./18.,1./36.])

    # bins = 13
    # windFrequencies2, windSpeeds2 = frequ(bins, windFrequencies1, windSpeeds1)
    # nDirections = len(windSpeeds2)
    # windDirections2 = np.linspace(0,360-360/nDirections, nDirections)
    #
    # # "Multiple wind speeds"""
    # num = 10
    # # actualSpeeds = windSpeeds
    # speedFreqs = speedFreq(num)
    # speeds = np.zeros((len(windSpeeds2)*num))
    # newWindDirections = np.zeros(num*len(windSpeeds2))
    # newWindFrequencies = np.zeros(num*len(windSpeeds2))
    #
    # for k in range(len(windSpeeds2)):
    #     speeds[k*num:k*num+num] = actualSpeeds(num, windSpeeds2[k])
    #     newWindDirections[k*num:k*num+num] = windDirections2[k]
    #     newWindFrequencies[k*num:k*num+num] = windFrequencies2[k]*speedFreqs
    #
    # windDirections = newWindDirections
    # # if num == 1:
    # #     speeds = windSpeeds
    # windSpeeds = speeds
    # windFrequencies = newWindFrequencies
    # nDirections = len(windSpeeds)

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

    # COE = np.zeros(len(space))
    # AEP = np.zeros(len(space))
    # turbineH1array = np.zeros(len(space))
    # turbineH2array = np.zeros(len(space))
    # diffArray = np.zeros(len(space))
    # cost = np.zeros(len(space))
    # runTime = np.zeros(len(space))
    # opt_filenameXYZ = 'XYZ_warm_0.16.txt'
    # opt_filename_dt = 'dt_warm_0.16.txt'
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


    shear_ex = np.array([0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.3])
    shear_ex = np.array([2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0,8.5,9.0,9.5,10.0])
    COE = np.zeros(len(shear_ex))
    AEP = np.zeros(len(shear_ex))
    shellH11 = np.zeros(len(shear_ex))
    shellH21 = np.zeros(len(shear_ex))
    globalH11 = np.zeros(len(shear_ex))
    globalH21 = np.zeros(len(shear_ex))
    freqH11 = np.zeros(len(shear_ex))
    freqH21 = np.zeros(len(shear_ex))
    shearH11 = np.zeros(len(shear_ex))
    shearH21 = np.zeros(len(shear_ex))
    manH1 = np.zeros(len(shear_ex))
    weldH1 = np.zeros(len(shear_ex))
    manH2 = np.zeros(len(shear_ex))
    weldH2 = np.zeros(len(shear_ex))

    shellH12 = np.zeros(len(shear_ex))
    shellH22 = np.zeros(len(shear_ex))
    globalH12 = np.zeros(len(shear_ex))
    globalH22 = np.zeros(len(shear_ex))
    freqH12 = np.zeros(len(shear_ex))
    freqH22 = np.zeros(len(shear_ex))
    shearH12 = np.zeros(len(shear_ex))
    shearH22 = np.zeros(len(shear_ex))

    diff = np.zeros(len(shear_ex))

    for i in range(len(shear_ex)):

        shearExp = 0.1
        opt_filenameXYZ = 'XYZ_XYZdt_%s.txt'%shear_ex[i]
        opt_filename_dt = 'dt_XYZdt_%s.txt'%shear_ex[i]
        opt_filename_yaw = 'YAW_%s.txt'%shear_ex[i]

        optXYZ = open(opt_filenameXYZ)
        optimizedXYZ = np.loadtxt(optXYZ)
        turbineX = optimizedXYZ[:,0]
        turbineY = optimizedXYZ[:,1]
        turbineZ = optimizedXYZ[:,2]
        turbineH1 = turbineZ[0]
        turbineH2 = turbineZ[1]

        print turbineH1
        print turbineH2
        opt_dt = open(opt_filename_dt)
        optimized_dt = np.loadtxt(opt_dt)
        d_paramH1 = optimized_dt[:,0]
        d_paramH2 = optimized_dt[:,1]
        t_paramH1 = optimized_dt[:,2]
        t_paramH2 = optimized_dt[:,3]
        diff[i] = abs(turbineH1-turbineH2)
        opt_yaw = open(opt_filename_yaw)
        optimizedYaw = np.loadtxt(opt_yaw)
        yaw = np.zeros((nDirections, nTurbs))
        for j in range(nDirections):
            yaw[j] = optimizedYaw[j]

        """set up the problem"""
        prob = Problem()
        root = prob.root = Group()

        root.add('getTurbineZ', getTurbineZ(nTurbs), promotes=['*'])
        root.add('get_z_paramH1', get_z(nPoints))
        root.add('get_z_paramH2', get_z(nPoints))
        root.add('get_z_fullH1', get_z(n))
        root.add('get_z_fullH2', get_z(n))
        root.add('AEPGroup', AEPGroup(nTurbs, nDirections=nDirections,
                    use_rotor_components=use_rotor_components, datasize=datasize, differentiable=True,
                    optimizingLayout=False, nSamples=0), promotes=['*'])
        root.add('COEGroup', COEGroup(nTurbs), promotes=['*'])

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

        prob.setup(check=False)

        if wind == "PowerWind":
            prob['TowerH1.wind1.shearExp'] = shearExp
            prob['TowerH1.wind2.shearExp'] = shearExp
            prob['TowerH2.wind1.shearExp'] = shearExp
            prob['TowerH2.wind2.shearExp'] = shearExp
            prob['shearExp'] = shearExp

        prob['TowerH1.d_param'] = d_paramH1
        prob['TowerH1.t_param'] = t_paramH1
        prob['TowerH2.d_param'] = d_paramH2
        prob['TowerH2.t_param'] = t_paramH2
        prob['turbineH1'] = turbineH1
        prob['turbineH2'] = turbineH2
        prob['H1_H2'] = H1_H2

        prob['turbineX'] = turbineX
        prob['turbineY'] = turbineY
        # prob['yaw0'] = yaw
        for direction in range(nDirections):
            prob['yaw%s'%direction] = yaw[direction]
            print yaw[direction]
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
        prob['floris_params:cos_spread'] = 1E12
        prob['zref'] = wind_zref
        prob['z0'] = wind_z0       # turns off cosine spread (just needs to be very large)

        """tower structural properties"""
        # --- geometry ----
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

        # start = time.time()
        prob.run()
        # runTime[i] = time.time()-start

        COE[i] = prob['COE']
        AEP[i]  = prob['AEP']
        shellH11[i] = np.max(prob['TowerH1.tower1.shell_buckling'])
        shellH21[i] = np.max(prob['TowerH2.tower1.shell_buckling'])
        globalH11[i] = np.max(prob['TowerH1.tower1.global_buckling'])
        globalH21[i] = np.max(prob['TowerH2.tower1.global_buckling'])
        freqH11[i] = np.max(prob['TowerH1.tower1.f1'])
        freqH21[i] = np.max(prob['TowerH2.tower1.f1'])
        shearH11[i] = np.max(prob['TowerH1.tower1.stress'])
        shearH21[i] = np.max(prob['TowerH2.tower1.stress'])

        shellH12[i] = np.max(prob['TowerH1.tower2.shell_buckling'])
        shellH22[i] = np.max(prob['TowerH2.tower2.shell_buckling'])
        globalH12[i] = np.max(prob['TowerH1.tower2.global_buckling'])
        globalH22[i] = np.max(prob['TowerH2.tower2.global_buckling'])
        freqH12[i] = np.max(prob['TowerH1.tower2.f1'])
        freqH22[i] = np.max(prob['TowerH2.tower2.f1'])
        shearH12[i] = np.max(prob['TowerH1.tower2.stress'])
        shearH22[i] = np.max(prob['TowerH2.tower2.stress'])

        manH1[i] = np.max(prob['TowerH1.gc.manufacturability'])
        manH2[i] = np.max(prob['TowerH2.gc.manufacturability'])
        weldH1[i] = np.max(prob['TowerH1.gc.weldability'])
        weldH2[i] = np.max(prob['TowerH2.gc.weldability'])


    print 'COE: ', COE
    print 'AEP: ', AEP

    print 'TowerH1:'
    print 'Shell1: ', shellH11
    print 'Shell2: ', shellH12
    print 'Global1: ', globalH11
    print 'Global2: ', globalH12
    print 'Freq: ', freqH11
    print 'Stress1: ', shearH11
    print 'Stress2: ', shearH12
    print 'manufacturability: ', manH1
    print 'weldability: ', weldH1

    print 'TowerH2:'
    print 'Shell1: ', shellH21
    print 'Shell2: ', shellH22
    print 'Global1: ', globalH21
    print 'Global2: ', globalH22
    print 'Freq: ', freqH21
    print 'Stress1: ', shearH21
    print 'Stress2: ', shearH22
    print 'manufacturability: ', manH2
    print 'weldability: ', weldH2

    # print 'WindSpeeds: ', prob['windSpeeds']
    # for i in range(nDirections):
    #     print 'wtVelocity: ', prob['wtVelocity%i'%i]
    # loss = np.zeros((nTurbs, nDirections))
    # #print the loss of each turbine from each direction to test the wake model
    # for i in range(nTurbs):
    #     for j in range(nDirections):
    #         loss[i][j] = 1. - prob['wtVelocity%i'%j][i]/prob['windSpeeds'][i][j]
    # print 'Loss: ', loss

    # for i in range(nTurbs):
    #     if H1_H2[i] == 0:
    #         plt.plot(turbineX[i], turbineY[i], 'bo')
    #     else:
    #         plt.plot(turbineX[i], turbineY[i], 'ro')
    # bbox_props = dict(boxstyle="rarrow,pad=0.3", fc="cyan", ec="b", lw=2)
    # t = plt.text(0, 0, " Wind Direction", ha="center", va="center", rotation=-windDirections[0]-90.,
    #         size=15, bbox=bbox_props)
    # plt.axis([np.min(turbineX)-100, np.max(turbineX)+100, np.min(turbineY)-100, np.max(turbineY)+100])
    # # plt.show()


    # print 'Tower H1 Mass: ', prob['TowerH1.tower1.mass']
    # print 'Tower H2 Mass: ', prob['TowerH2.tower1.mass']
    #
    # print 'Tower H1 Stress: Max Thrust: ', prob['TowerH1.tower1.stress']
    # print 'Tower H1 Stress: Max Wind Speed: ', prob['TowerH1.tower2.stress']
    # print 'Tower H2 Stress: Max Thrust: ', prob['TowerH2.tower1.stress']
    # print 'Tower H2 Stress: Max Wind Speed: ', prob['TowerH2.tower2.stress']
    #
    print 'Tower H1 Shell Buckling: Max Thrust: ', prob['TowerH1.tower1.shell_buckling']
    print 'Tower H1 Shell Buckling: Max Wind Speed: ', prob['TowerH1.tower2.shell_buckling']
    print 'Tower H2 Shell Buckling: Max Thrust: ', prob['TowerH2.tower1.shell_buckling']
    print 'Tower H2 Shell Buckling: Max Wind Speed: ', prob['TowerH2.tower2.shell_buckling']
    print 'Diff: ', diff
    #
    # print 'Tower H1 Global Buckling: Max Thrust: ', prob['TowerH1.tower1.global_buckling']
    # print 'Tower H1 Global Buckling: Max Wind Speed: ', prob['TowerH1.tower2.global_buckling']
    # print 'Tower H2 Global Buckling: Max Thrust: ', prob['TowerH2.tower1.global_buckling']
    # print 'Tower H2 Global Buckling: Max Wind Speed: ', prob['TowerH2.tower2.global_buckling']
    #
    # print 'Tower H1 Damage: ', prob['TowerH1.tower1.damage']
    # print 'Tower H2 Damage: ', prob['TowerH2.tower1.damage']
    #
    # print 'Tower H1 weldability: ', prob['TowerH1.gc.weldability']
    # print 'Tower H1 manufacturability: ', prob['TowerH1.gc.manufacturability']
    # print 'Tower H2 weldability: ', prob['TowerH2.gc.weldability']
    # print 'Tower H2 manufacturability: ', prob['TowerH2.gc.manufacturability']
    #
    # print 'Tower H1 frequency: ', prob['TowerH1.tower1.f1']
    # print 'Tower H2 frequency: ', prob['TowerH2.tower1.f1']



# Tower H1 Shell Buckling: Max Thrust:  [ 0.78857099  0.79857385  0.81076871  0.82590174  0.84503244  0.86976227
#   0.90262347  0.94782946  0.98295925  1.          0.98831994  0.93475474
#   0.82858079  0.69162262  1.        ]
# Tower H1 Shell Buckling: Max Wind Speed:  [ 1.          0.97645784  0.95247314  0.93044097  0.91183225  0.89846967
#   0.89293136  0.89919019  0.88815917  0.85989702  0.80849361  0.72739636
#   0.61247569  0.47887171  0.77211627]
# Tower H2 Shell Buckling: Max Thrust:  [ 0.78857099  0.79857385  0.81076871  0.82590174  0.84503244  0.86976227
#   0.90262347  0.94782946  0.98295925  1.          0.98831994  0.93475474
#   0.82858079  0.69162262  1.        ]
# Tower H2 Shell Buckling: Max Wind Speed:  [ 1.          0.97645784  0.95247314  0.93044097  0.91183225  0.89846967
#   0.89293136  0.89919019  0.88815917  0.85989702  0.80849361  0.72739636
#   0.61247569  0.47887171  0.77211627]
# Tower H1 Global Buckling: Max Thrust:  [ 0.79249925  0.78337674  0.77357919  0.76299179  0.7514582   0.73877094
#   0.7246515   0.70871909  0.73398565  0.75779673  0.77855217  0.79388833
#   0.80077337  0.8015905   1.        ]
# Tower H1 Global Buckling: Max Wind Speed:  [ 0.87587972  0.85103275  0.82483545  0.7976298   0.7695356   0.74062964
#   0.71096492  0.68057812  0.69048898  0.69977919  0.70829583  0.7159452
#   0.72298165  0.7340041   0.9788892 ]
# Tower H2 Global Buckling: Max Thrust:  [ 0.79249925  0.78337674  0.77357919  0.76299179  0.7514582   0.73877094
#   0.7246515   0.70871909  0.73398565  0.75779673  0.77855217  0.79388833
#   0.80077337  0.8015905   1.        ]
# Tower H1 frequency:  0.22
# Tower H2 frequency:  0.22
