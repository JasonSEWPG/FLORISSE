from florisse.COE import COEGroup
from florisse.GeneralWindFarmComponents import get_z, get_z_DEL, getTurbineZ, AEPobj, actualSpeeds, speedFreq
from florisse.floris import AEPGroup
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
        nRows = 3
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

    minSpacing = 2

    """set up 3D aspects of wind farm"""
    H1_H2 = np.array([])
    for i in range(nTurbs/2):
        H1_H2 = np.append(H1_H2, 0)
        H1_H2 = np.append(H1_H2, 1)
    if len(H1_H2) < nTurbs:
        H1_H2 = np.append(H1_H2, 0)
    # H1_H2 = np.zeros(nTurbs)

    """Define wind flow"""
    air_density = 1.1716    # kg/m^3

    windData = "Amalia"

    """Amalia Wind Arrays"""
    if windData == "Amalia":
        windSpeeds1, windFrequencies1, windDirections1, nDirections = amaliaWind()

    """Manual Wind Arrays"""
    if windData == "Manual":
        nDirections = 8
        windSpeeds = np.array([12.,11.25,10.5,9.75,9.,8.25,7.5,6.75])
        windDirections = np.array([225.,270.,315.,0.,45.,90.,135.,180.])
        windFrequencies = np.array([2./9.,7./36.,3./18.,5./36.,1./9.,1./12.,1./18.,1./36.])

    bins = 13
    windFrequencies2, windSpeeds2 = frequ(bins, windFrequencies1, windSpeeds1)
    nDirections = len(windSpeeds2)
    windDirections2 = np.linspace(0,360-360/nDirections, nDirections)

    # "Multiple wind speeds"""
    num = 10
    # actualSpeeds = windSpeeds
    speedFreqs = speedFreq(num)
    speeds = np.zeros((len(windSpeeds2)*num))
    newWindDirections = np.zeros(num*len(windSpeeds2))
    newWindFrequencies = np.zeros(num*len(windSpeeds2))

    for k in range(len(windSpeeds2)):
        speeds[k*num:k*num+num] = actualSpeeds(num, windSpeeds2[k])
        newWindDirections[k*num:k*num+num] = windDirections2[k]
        newWindFrequencies[k*num:k*num+num] = windFrequencies2[k]*speedFreqs

    windDirections = newWindDirections
    # if num == 1:
    #     speeds = windSpeeds
    windSpeeds = speeds
    windFrequencies = newWindFrequencies
    nDirections = len(windSpeeds)



    nIntegrationPoints = 1 #Number of points in wind effective wind speed integral

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

    opt_filenameXYZ = 'XYZ3_warmSAME_6.0.txt'
    opt_filename_dt = 'dt3_warmSAME_6.0.txt'

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

    shearExp = 0.3

    num = 200
    turbine = np.zeros(nTurbs)
    turbine[13] = 1.
    turbineH1new = np.linspace(turbineH1-2,turbineH1+2,num)
    turbineH2new = np.linspace(turbineH2-2,turbineH2+2,num)

    d_paramH1new = np.zeros((num,3))
    d_paramH2new = np.zeros((num,3))
    t_paramH1new = np.zeros((num,3))
    t_paramH2new = np.zeros((num,3))
    turbineXnew = np.zeros((num, nTurbs))
    turbineYnew = np.zeros((num, nTurbs))
    for i in range(num):
        d_paramH1new[i] = (d_paramH1+(-1+i*2./num))*np.ones(3)
        d_paramH2new[i] = (d_paramH2+(-1+i*2./num))*np.ones(3)

        t_paramH1new[i] = (t_paramH1+(-0.0005+i*0.001/num))*np.ones(3)
        t_paramH2new[i] = (t_paramH2+(-0.0005+i*0.001/num))*np.ones(3)

        turbineXnew[i] = turbineX+(-5.+i*10/num)*turbine
        turbineYnew[i] = turbineY+(-5.+i*10/num)*turbine
    print d_paramH2new
    print t_paramH2new

    COE = np.zeros(num)
    AEP = np.zeros(num)
    stressMS = np.zeros(num)
    stressMT = np.zeros(num)
    gbucklingMS = np.zeros(num)
    gbucklingMT = np.zeros(num)
    sbucklingMS = np.zeros(num)
    sbucklingMT = np.zeros(num)

    for i in range(num):
        # turbineH1 = turbineH1new[i]
        # turbineH2 = turbineH2new[i]
        # d_paramH1 = d_paramH1new[i]
        # d_paramH2 = d_paramH2new[i]
        turbineX = turbineXnew[i]

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
        prob['TowerH2.d_param'] = d_paramH1
        prob['TowerH2.t_param'] = t_paramH1
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

        print i
        COE[i] = prob['COE']
        AEP[i] = prob['AEP']
        stressMT[i] = np.max(prob['TowerH1.tower1.stress'])
        stressMS[i] = np.max(prob['TowerH1.tower2.stress'])
        gbucklingMT[i] = np.max(prob['TowerH1.tower1.shell_buckling'])
        gbucklingMS[i] = np.max(prob['TowerH1.tower2.shell_buckling'])
        sbucklingMT[i] = np.max(prob['TowerH1.tower1.global_buckling'])
        sbucklingMS[i] = np.max(prob['TowerH1.tower2.global_buckling'])

    np.savetxt('continuityX.txt', np.c_[COE, AEP], header="COE, AEP")

    print 'COE: ', COE
    print 'stressMT: ', stressMT
    print 'stressMS: ', stressMS
    print 'gbucklingMT: ', gbucklingMT
    print 'gbucklingMS: ', gbucklingMS
    print 'sbucklingMT: ', sbucklingMT
    print 'sbucklingMS: ', sbucklingMS
