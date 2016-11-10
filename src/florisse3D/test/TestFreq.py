from FLORISSE3D.GeneralWindFarmComponents import get_z, getTurbineZ
from towerse.tower import TowerSE
import numpy as np
import matplotlib.pyplot as plt
from openmdao.api import Problem, Group, IndepVarComp
from setupOptimization import *
from FLORISSE3D.simpleTower import calcFreq



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
    d_param = np.array([6.0, 5.0, 4.0])
    t_param = np.array([0.027*1.3, 0.025*1.3, 0.018*1.3])
    n = 15

    turbineH1 = 120.

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

    shearExp = 0.2

    num = 100
    turbH1 = np.linspace(72.,135.,num)
    D = np.linspace(3.,7., num)
    towerFreq = np.zeros(num)
    simpleFreq = np.zeros(num)
    ratio = np.zeros(num)
    for i in range(num):
        turbineH1 = turbH1[i]
        """set up the problem"""
        prob = Problem()
        root = prob.root = Group()

        root.add('getTurbineZ', getTurbineZ(nTurbs), promotes=['*'])
        root.add('get_z_paramH1', get_z(nPoints))
        root.add('get_z_fullH1', get_z(n))
        #For Constraints
        root.add('TowerH1', TowerSE(nPoints, nFull, nK, nMass, nPL, nDEL, wind=wind), promotes=['L_reinforced',
                            'E','G','sigma_y','kidx','kx','ky','kz','ktx','kty',
                            'ktz','midx','m','mIxx','mIyy','mIzz','mIxy','mIxz','mIyz',
                            'mrhox','mrhoy','mrhoz','addGravityLoadForExtraMass','g',
                            'gamma_f','gamma_m','gamma_n','gamma_b','life','m_SN',
                            'gc.min_d_to_t','gc.min_taper','M_DEL','gamma_fatigue'])
        root.add('calcFreq', calcFreq(), promotes=['d_param','t_param','rho','turbineH'])

        root.connect('turbineH1', 'get_z_paramH1.turbineZ')
        root.connect('turbineH1', 'get_z_fullH1.turbineZ')
        root.connect('get_z_paramH1.z_param', 'TowerH1.z_param')
        root.connect('get_z_fullH1.z_param', 'TowerH1.z_full')

        prob.setup(check=False)

        prob['TowerH1.wind1.shearExp'] = shearExp
        prob['TowerH1.wind2.shearExp'] = shearExp
        prob['TowerH1.d_param'] = d_param
        prob['TowerH1.t_param'] = t_param

        prob['turbineH1'] = turbineH1
        prob['H1_H2'] = H1_H2

        """tower structural properties"""
        # --- geometry ----
        prob['L_reinforced'] = L_reinforced
        prob['TowerH1.yaw'] = Toweryaw

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
        prob['TowerH1.z0'] = wind_z0
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

        prob['rho'] = rho[0]
        prob['turbineH'] = turbineH1
        prob['d_param'] = d_param
        prob['t_param'] = t_param
        prob['calcFreq.E'] = E[0]
        prob['calcFreq.topM'] = m
        prob['calcFreq.topI'] = mIzz


        prob.run()

        print 'TowerSE: ', prob['TowerH1.tower1.f1']
        print 'Simple Tower Model: ', prob['calcFreq.freq']
        towerFreq[i] = prob['TowerH1.tower1.f1']
        simpleFreq[i] = prob['calcFreq.freq']*1.15
        ratio[i] = abs(towerFreq[i]/simpleFreq[i])

    print 'Tower: ', towerFreq
    print 'Simple: ', simpleFreq
    print 'Ratio: ', ratio

    plt.plot(turbH1, ratio)
    plt.show()
