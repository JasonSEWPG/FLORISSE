import numpy as np
import sys
from openmdao.api import ScipyOptimizer, IndepVarComp
from FLORISSE3D.GeneralWindFarmComponents import AEPobj, get_z, getTurbineZ, get_z_DEL
from FLORISSE3D.GeneralWindFarmComponents import calculate_boundary, SpacingComp,\
BoundaryComp, get_z, organizeWindSpeeds, getTurbineZ, AEPobj, speedFreq, actualSpeeds
from FLORISSE3D.simpleTower import calcMass, calcFreq
import matplotlib.pyplot as plt
from FLORISSE3D.floris import AEPGroup
from openmdao.api import Problem, Group, IndepVarComp, pyOptSparseDriver, ExecComp, ScipyOptimizer
import time
import cPickle as pickle
from setupOptimization import *
from towerse.tower import TowerSE

if __name__=="__main__":

    """Define tower structural properties"""
    # --- geometry ----
    d_param = np.array([6.0, 6.0, 6.0])
    t_param = np.array([0.05,0.05,0.05])
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

    shearExp = 0.1
    turbineH = 125.

    num = np.linspace(-1.,1.,30)
    numT = np.linspace(-0.02,0.02,30)
    frequency = np.zeros(len(num))
    for i in range(len(num)):
        d_paramNEW = np.zeros(3)
        d_paramNEW[:] = d_param[:]
        d_paramNEW[1] += num[i]
        t_paramNEW = np.zeros(3)
        t_paramNEW[:] = t_param[:]
        t_paramNEW[1] += numT[i]
        print d_paramNEW
        prob = Problem()
        root = prob.root = Group()

        root.add('turbineH1', IndepVarComp('turbineH1', turbineH), promotes=['*'])
        root.add('get_z_paramH1', get_z(nPoints))
        root.add('get_z_fullH1', get_z(n))
        #For Constraints
        root.add('TowerH1', TowerSE(nPoints, nFull, nK, nMass, nPL, nDEL, wind=wind), promotes=['L_reinforced',
        'E','G','sigma_y','kidx','kx','ky','kz','ktx','kty',
        'ktz','midx','m','mIxx','mIyy','mIzz','mIxy','mIxz','mIyz',
        'mrhox','mrhoy','mrhoz','addGravityLoadForExtraMass','g',
        'gamma_f','gamma_m','gamma_n','gamma_b','life','m_SN',
        'gc.min_d_to_t','gc.min_taper','M_DEL','gamma_fatigue'])

        root.add('calcFreq', calcFreq())
        root.add('calcMass', calcMass())

        root.connect('turbineH1', 'calcFreq.turbineH')
        root.connect('turbineH1', 'get_z_paramH1.turbineZ')
        root.connect('turbineH1', 'get_z_fullH1.turbineZ')
        root.connect('get_z_paramH1.z_param', 'TowerH1.z_param')
        root.connect('get_z_fullH1.z_param', 'TowerH1.z_full')
        root.connect('calcMass.mass', 'calcFreq.mass')

        prob.setup()

        prob['calcFreq.d_param'] = d_paramNEW
        prob['calcFreq.t_param'] = t_paramNEW
        prob['calcFreq.E'] = E[0]
        prob['calcMass.d_param'] = d_paramNEW
        prob['calcMass.t_param'] = t_paramNEW
        prob['calcMass.turbineH'] = turbineH
        prob['calcMass.rho'] = rho[0]


        if wind == "PowerWind":
            prob['TowerH1.wind1.shearExp'] = shearExp
            prob['TowerH1.wind2.shearExp'] = shearExp

        prob['TowerH1.d_param'] = d_paramNEW
        prob['TowerH1.t_param'] = t_paramNEW

        prob['turbineH1'] = turbineH

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

        # --- fatigue ---
        prob['gamma_fatigue'] = gamma_fatigue
        prob['life'] = life
        prob['m_SN'] = m_SN
        # ---------------

        prob.run()

        frequency[i] = prob['calcFreq.freq']/prob['TowerH1.tower1.f1']

        print 'TowerSE'
        print 'Tower H1 Mass: ', prob['TowerH1.tower1.mass']
        print 'Tower frequency: ', prob['TowerH1.tower1.f1']

        print 'Simple'
        print 'Mass: ', prob['calcMass.mass']
        print 'Freq: ', prob['calcFreq.freq']

        print 'Ratio: ', prob['calcFreq.freq']/prob['TowerH1.tower1.f1']
    print 'Frequency: ', frequency
    print 'Max: ', np.max(frequency)
    print 'Min: ', np.min(frequency)
