from FLORISSE3D.GeneralWindFarmComponents import get_z, getTurbineZ
from towerse.tower import TowerSE
import numpy as np
import matplotlib.pyplot as plt
from openmdao.api import Problem, Group, IndepVarComp
from FLORISSE3D.setupOptimization import *
from FLORISSE3D.simpleTower import freq, averageI, Tower
import matplotlib.pyplot as plt



"""This is an example run script that includes both COE and the tower model"""

if __name__=="__main__":

    nTurbs = 1

    """set up 3D aspects of wind farm"""
    H1_H2 = np.array([])
    for i in range(nTurbs/2):
        H1_H2 = np.append(H1_H2, 0)
        H1_H2 = np.append(H1_H2, 1)
    if len(H1_H2) < nTurbs:
        H1_H2 = np.append(H1_H2, 0)

    """Define tower structural properties"""
    # --- geometry ----
    d_param = np.array([6.0, 5.0, 3.0])
    t_param = np.array([0.02, 0.018, 0.015])
    n = 15

    turbineH1 = 135.

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
    # freqTOWER = np.zeros(len(turbH))
    # freqSIMPLE = np.zeros(len(turbH))
    shearExp = 0.1
    # num = np.linspace(0.,49.,50)

    # d_param = np.array([6.0, 5.+0.02*num[i], 4.+0.04*num[i]])
    # t_param = np.array([0.027+.002*num[i], 0.025+.0015*num[i], 0.018+.001*num[i]])
    shear_ex = np.array([2.0,2.5,3.0,3.5,4.0,4.5,5.5,6.0,6.5,7.0,7.5,8.0])
    # shear_ex = np.array([0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29])

    weld=-100.
    man=-100.
    freqHigh = -100.
    freqLow = 100.
    for i in range(len(shear_ex)):

        shearExp = shear_ex[i]

        opt_filenameXYZ = 'Z2_XYZ_XYZdtSAME_%s.txt'%shear_ex[i]
        opt_filename_dt = 'Z2_dt_XYZdtSAME_%s.txt'%shear_ex[i]
        # opt_filenameXYZ = 'XYZ_XYZdt7_analytic_%s.txt'%0.11
        # opt_filename_dt = 'dt_XYZdt7_analytic_%s.txt'%0.11
        # opt_filename_yaw = 'YAW_%s.txt'%shear_ex[i]

        optXYZ = open(opt_filenameXYZ)
        optimizedXYZ = np.loadtxt(optXYZ)
        turbineX = optimizedXYZ[:,0]
        turbineY = optimizedXYZ[:,1]
        turbineZ = optimizedXYZ[:,2]
        turbineH1 = np.float(turbineZ[0])
        turbineH2 = np.float(turbineZ[1])

        opt_dt = open(opt_filename_dt)
        optimized_dt = np.loadtxt(opt_dt)
        d_paramH1 = optimized_dt[:,0]
        d_paramH2 = optimized_dt[:,1]
        t_paramH1 = optimized_dt[:,2]
        t_paramH2 = optimized_dt[:,3]

        d_param = d_paramH1
        t_param = t_paramH1
        turbineH1 = turbineH1

        """set up the problem"""

        prob = Problem()
        root = prob.root = Group()

        root.add('getTurbineZ', getTurbineZ(nTurbs), promotes=['*'])
        root.add('get_z_paramH1', get_z(nPoints))
        root.add('get_z_fullH1', get_z(n))
        #For Constraints
        root.add('TowerH1', TowerSE(nPoints, nFull, nK, nMass, nPL, nDEL, wind=wind), promotes=['L_reinforced',
                            'G','sigma_y','kidx','kx','ky','kz','ktx','kty', 'E',
                            'ktz','midx','mIxx','mIyy','mIzz','mIxy','mIxz','mIyz',
                            'mrhox','mrhoy','mrhoz','addGravityLoadForExtraMass','g',
                            'gamma_f','gamma_m','gamma_n','gamma_b','life','m_SN',
                            'gc.min_d_to_t','gc.min_taper','M_DEL','gamma_fatigue'])
        root.add('Tower', Tower(nPoints, nFull), promotes=['*'])

        root.connect('turbineH1', 'get_z_paramH1.turbineZ')
        root.connect('turbineH1', 'get_z_fullH1.turbineZ')
        root.connect('get_z_paramH1.z_param', 'TowerH1.z_param')
        root.connect('get_z_fullH1.z_param', 'TowerH1.z_full')
        root.connect('turbineH1', 'L')

        root.connect('get_z_fullH1.z_param', 'z_full')
        root.connect('get_z_paramH1.z_param', 'z_param')

        prob.setup(check=False)

        prob['t_param'] = t_param
        prob['d_param'] = d_param
        prob['TowerH1.wind1.shearExp'] = shearExp
        prob['TowerH1.wind2.shearExp'] = shearExp
        prob['TowerH1.d_param'] = d_param
        prob['TowerH1.t_param'] = t_param
        prob['rho'] = rho

        prob['turbineH1'] = turbineH1
        prob['H1_H2'] = H1_H2

        """tower structural properties"""
        # --- geometry ----
        prob['L_reinforced'] = L_reinforced
        prob['TowerH1.yaw'] = Toweryaw
        prob['Mt'] = m[0]

        # --- material props ---
        prob['E'] = E
        prob['G'] = G
        prob['TowerH1.tower1.rho'] = rho
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
        prob['TowerH1.m'] = m
        prob['midx'] = midx
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
        prob['TowerH1.z0'] = wind_z0
        # ---------------

        # # --- loading case 1: max Thrust ---
        prob['TowerH1.wind1.Uref'] = wind_Uref1
        # prob['TowerH1.wind1.Uref'] = 0.0
        prob['TowerH1.tower1.plidx'] = plidx1
        prob['TowerH1.tower1.Fx'] = Fx1
        prob['TowerH1.tower1.Fy'] = Fy1
        prob['TowerH1.tower1.Fz'] = Fz1
        prob['TowerH1.tower1.Mxx'] = Mxx1
        prob['TowerH1.tower1.Myy'] = Myy1
        prob['TowerH1.tower1.Mzz'] = Mzz1
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
        # # ---------------

        # --- safety factors ---
        prob['gamma_f'] = gamma_f
        prob['gamma_m'] = gamma_m
        prob['gamma_n'] = gamma_n
        prob['gamma_b'] = gamma_b
        # ---------------

        prob['It'] = mIzz[0]

        prob.run()
        # freqSIMPLE[i] = prob['freq']
        # freqTOWER[i] = prob['TowerH1.tower1.f1']

        print 'TowerSE Frequency: ', prob['TowerH1.tower1.f1']

        print 'Simple Frequency: ', prob['freq']

        if np.max(prob['TowerH1.gc.weldability']) > weld:
            weld = np.max(prob['TowerH1.gc.weldability'])
        if np.max(prob['TowerH1.gc.manufacturability']) > man:
            man = np.max(prob['TowerH1.gc.manufacturability'])

        if prob['freq'] < freqLow:
            freqLow = prob['freq']
        if prob['freq'] > freqHigh:
            freqHigh = prob['freq']

    print 'Weldability: ', prob['TowerH1.gc.weldability']
    print 'Manufacturability: ', prob['TowerH1.gc.manufacturability']

    print weld
    print man

    print 'FreqLow: ', freqLow
    print 'FreqHigh: ', freqHigh
