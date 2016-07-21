#!/usr/bin/env python
# encoding: utf-8
"""
test_environment_gradients.py

Created by Andrew Ning on 2013-12-18.
Copyright (c) NREL. All rights reserved.
"""


import unittest
import numpy as np
from openmdao.api import pyOptSparseDriver, Problem, Group
import sys
from openmdao.api import ScipyOptimizer, IndepVarComp
from florisse.COE import *
from towerse.tower import TowerSE
from florisse.GeneralWindFarmComponents import AEPobj, get_z, getTurbineZ, get_z_DEL



class TestCost(unittest.TestCase):


    def setUp(self):
        nTurbines = 9
        mass1 = 870000.
        mass2 = 150000.
        H1_H2 = np.array([])
        for i in range(nTurbines/2):
            H1_H2 = np.append(H1_H2, 0)
            H1_H2 = np.append(H1_H2, 1)
        if len(H1_H2) < nTurbines:
            H1_H2 = np.append(H1_H2, 0)

        prob = Problem()
        root = prob.root = Group()

        root.add('mass1', IndepVarComp('mass1', mass1), promotes=['*'])
        root.add('mass2', IndepVarComp('mass2', mass2), promotes=['*'])
        root.add('farmCost', farmCost(nTurbines), promotes=['*'])

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'SNOPT'
        prob.driver.opt_settings['Major iterations limit'] = 1000
        prob.driver.opt_settings['Major optimality tolerance'] = 1E-6

        prob.driver.add_objective('cost', scaler=1E2)
        prob.driver.add_desvar('mass1', lower=1000., upper=None)
        prob.driver.add_desvar('mass2', lower=1000., upper=None)

        prob.setup()

        prob['H1_H2'] = H1_H2


        prob.run()

        self.J = prob.check_total_derivatives(out_stream=None)
        print self.J

    def test_mass(self):
        np.testing.assert_allclose(self.J[('cost', 'mass1')]['J_fwd'], self.J[('cost', 'mass1')]['J_fd'], 1e-3, 1e-3)
        np.testing.assert_allclose(self.J[('cost', 'mass2')]['J_fwd'], self.J[('cost', 'mass2')]['J_fd'], 1e-3, 1e-3)


class TestZEP(unittest.TestCase):


    def setUp(self):
        nRows = 2
        nTurbs = nRows**2
        rotorDiameter = np.zeros(nTurbs)
        axialInduction = np.zeros(nTurbs)
        Ct = np.zeros(nTurbs)
        Cp = np.zeros(nTurbs)
        generatorEfficiency = np.zeros(nTurbs)
        yaw = np.zeros(nTurbs)
        rotor_diameter = 126.4

        # define initial values
        for turbI in range(0, nTurbs):
            rotorDiameter[turbI] = rotor_diameter            # m
            axialInduction[turbI] = 1.0/3.0
            Ct[turbI] = 4.0*axialInduction[turbI]*(1.0-axialInduction[turbI])
            # Cp[turbI] = 0.7737/0.944 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
            Cp[turbI] = 0.7737 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
            generatorEfficiency[turbI] = 1.0#0.944
            yaw[turbI] = 0.     # deg.

        """set up locations of turbines"""
        spacing = 2     # turbine grid spacing in diameters

        # Set up position arrays
        points = np.linspace(start=spacing*rotor_diameter, stop=nRows*spacing*rotor_diameter, num=nRows)
        xpoints, ypoints = np.meshgrid(points, points)
        turbineX = np.ndarray.flatten(xpoints)
        turbineY = np.ndarray.flatten(ypoints)
        turbineZ = np.ones(nTurbs)*100.
        turbineZ[2] = 300.
        #turbineX = np.array([  849.26857068,  1263.9343377,    632.71670699,  1099.70742498])
        #turbineY = np.array([  632.18556535,   632.14896811,  1263.81443236,  1263.85102664])

        turbineX = np.array([0,0,0,0,0])
        turbineY = np.array([0,100,200,300,400])
        turbineZ = np.array([100,110,120,130,140])
        nTurbs = len(turbineX)
        """set up 3D aspects of wind farm"""
        turbineH1 = 87.6
        turbineH2 = 150.2
        H1_H2 = np.array([])
        for i in range(nTurbs/2):
            H1_H2 = np.append(H1_H2, 0)
            H1_H2 = np.append(H1_H2, 1)
        if len(H1_H2) < nTurbs:
            H1_H2 = np.append(H1_H2, 0)

        """Define wind flow"""
        air_density = 1.1716    # kg/m^3

        windSpeeds = np.array([4.,5.,6.,3.])
        nDirections = len(windSpeeds)
        windDirections = np.linspace(0,360-360/nDirections, nDirections)
        windFrequencies = np.ones(len(windSpeeds))/len(windSpeeds)

        nIntegrationPoints = 50
        wind = 'PowerWind'
        wind_zref = 90.0
        wind_z0 = 0.0
        shearExp = 0.2

        prob = Problem()
        root = prob.root = Group()

        root.fd_options['form'] = 'central'
        root.fd_options['step_size'] = 1.0e-5
        root.fd_options['step_type'] = 'relative'

        # root.add('turbineH1', IndepVarComp('turbineH1', turbineH1), promotes=['*'])
        # root.add('turbineH2', IndepVarComp('turbineH2', turbineH2), promotes=['*'])
        # root.add('getTurbineZ', getTurbineZ(nTurbs), promotes=['*'])
        root.add('turbineZ', IndepVarComp('turbineZ', turbineZ), promotes=['*'])
        # root.add('Uref', IndepVarComp('Uref', windSpeeds), promotes=['*'])
        root.add('AEPGroup', AEPGroup(nTurbs, nDirections=nDirections,
                    use_rotor_components=False, datasize=0, differentiable=True,
                    optimizingLayout=False, nSamples=0), promotes=['*'])
        root.add('maxAEP', AEPobj(), promotes=['*'])

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'SNOPT'
        prob.driver.opt_settings['Major iterations limit'] = 1000
        prob.driver.opt_settings['Major optimality tolerance'] = 1E-3

        prob.driver.add_objective('maxAEP', scaler=1E-8)
        prob.driver.add_desvar('turbineZ', lower=np.ones(nTurbs)*75., upper=None)
        # prob.driver.add_desvar('turbineH1', lower=10., upper=None)
        # prob.driver.add_desvar('turbineH2', lower=10., upper=None)
        prob.driver.add_desvar('turbineX', lower=np.ones(nTurbs)*min(turbineX), upper=np.ones(nTurbs)*max(turbineX))
        prob.driver.add_desvar('turbineY', lower=np.ones(nTurbs)*min(turbineY), upper=np.ones(nTurbs)*max(turbineY))
        # prob.driver.add_desvar('Uref', lower=np.zeros(nDirections), upper=None)

        prob.setup()

        #prob['turbineH1'] = turbineH1
        #prob['turbineH2'] = turbineH2

        # prob['H1_H2'] = H1_H2
        # prob['turbineZ'] = turbineZ
        prob['turbineX'] = turbineX
        prob['turbineY'] = turbineY
        prob['yaw0'] = yaw

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

        prob.run()

        self.J = prob.check_total_derivatives(out_stream=None)
        print "###################################################3"
        print self.J
        # print 'H2: ', self.J[('maxAEP', 'Uref')]

        print 'AEP: ', prob['maxAEP']

    def test_XY(self):
        np.testing.assert_allclose(self.J[('maxAEP', 'turbineX')]['J_fwd'], self.J[('maxAEP', 'turbineX')]['J_fd'], 1e-6, 1e-6)
        np.testing.assert_allclose(self.J[('maxAEP', 'turbineY')]['J_fwd'], self.J[('maxAEP', 'turbineY')]['J_fd'], 1e-6, 1e-6)

    # def testZ(self):
    #     np.testing.assert_allclose(self.J[('maxAEP', 'Uref')]['J_fwd'], self.J[('maxAEP', 'Uref')]['J_fd'], 1e-6, 1e-6)

    # def testZ(self):
    #     np.testing.assert_allclose(self.J[('AEP', 'turbineZ')]['J_fwd'], self.J[('AEP', 'turbineZ')]['J_fd'], 1e-6, 1e-6)

    # def test_H1(self):
    #     np.testing.assert_allclose(self.J[('AEP', 'turbineH2')]['J_fwd'], self.J[('AEP', 'turbineH2')]['J_fd'], 1e-6, 1e-6)
    #
    # def test_H2(self):
    #     np.testing.assert_allclose(self.J[('AEP', 'turbineH1')]['J_fwd'], self.J[('AEP', 'turbineH1')]['J_fd'], 1e-6, 1e-6)
    #

class TestMass(unittest.TestCase):


    def setUp(self):
        nTurbs = 9
        # --- geometry ----
        d_param = np.array([6.0, 4.935, 3.87]) # not going to modify this right now
        t_param = np.array([0.027*1.3, 0.023*1.3, 0.019*1.3]) # not going to modify this right now
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
        m = np.array([285598.8])
        mIxx = np.array([1.14930678e+08])
        mIyy = np.array([2.20354030e+07])
        mIzz = np.array([1.87597425e+07])
        mIxy = np.array([0.00000000e+00])
        mIxz = np.array([5.03710467e+05])
        mIyz = np.array([0.00000000e+00])
        mrhox = np.array([-1.13197635])
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
        Fx1 = np.array([1284744.19620519])
        Fy1 = np.array([0.])
        Fz1 = np.array([-2914124.84400512])
        Mxx1 = np.array([3963732.76208099])
        Myy1 = np.array([-2275104.79420872])
        Mzz1 = np.array([-346781.68192839])
        nPL = len(plidx1)
        # # ---------------

        # # --- loading case 2: max wind speed ---
        wind_Uref2 = 70.0
        plidx2 = np.array([n-1], dtype=int)  # at  top
        Fx2 = np.array([930198.60063279])
        Fy2 = np.array([0.])
        Fz2 = np.array([-2883106.12368949])
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

        H1_H2 = np.array([])
        for i in range(nTurbs/2):
            H1_H2 = np.append(H1_H2, 0)
            H1_H2 = np.append(H1_H2, 1)
        if len(H1_H2) < nTurbs:
            H1_H2 = np.append(H1_H2, 0)

        nPoints = len(d_param)
        nFull = n
        wind = 'PowerWind'

        turbineH1 = 100.
        turbineH2 = 120.
        rotor_diameter = 126.4

        prob = Problem()
        root = prob.root = Group()


        root.add('turbineH1', IndepVarComp('turbineH1', turbineH1), promotes=['*'])
        root.add('turbineH2', IndepVarComp('turbineH2', turbineH2), promotes=['*'])
        root.add('d_paramH1', IndepVarComp('d_paramH1', d_param), promotes=['*'])
        root.add('t_paramH1', IndepVarComp('t_paramH1', t_param), promotes=['*'])
        root.add('d_paramH2', IndepVarComp('d_paramH2', d_param), promotes=['*'])
        root.add('t_paramH2', IndepVarComp('t_paramH2', t_param), promotes=['*'])
        root.add('getTurbineZ', getTurbineZ(nTurbs), promotes=['*'])
        root.add('get_z_paramH1', get_z(nPoints))
        root.add('get_z_paramH2', get_z(nPoints))
        root.add('get_z_fullH1', get_z(n))
        root.add('get_z_fullH2', get_z(n))
        root.add('get_zDELH1', get_z_DEL())
        root.add('get_zDELH2', get_z_DEL())
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
        root.add('farmCost', farmCost(nTurbs), promotes=['*'])

        root.connect('turbineH1', 'get_z_paramH1.turbineZ')
        root.connect('turbineH2', 'get_z_paramH2.turbineZ')
        root.connect('turbineH1', 'get_z_fullH1.turbineZ')
        root.connect('turbineH2', 'get_z_fullH2.turbineZ')
        root.connect('get_z_paramH1.z_param', 'TowerH1.z_param')
        root.connect('get_z_fullH1.z_param', 'TowerH1.z_full')
        root.connect('get_z_paramH2.z_param', 'TowerH2.z_param')
        root.connect('get_z_fullH2.z_param', 'TowerH2.z_full')
        root.connect('turbineH1', 'get_zDELH1.turbineZ')
        root.connect('turbineH2', 'get_zDELH2.turbineZ')
        root.connect('get_zDELH1.z_DEL', 'TowerH1.z_DEL')
        root.connect('get_zDELH2.z_DEL', 'TowerH2.z_DEL')
        root.connect('TowerH1.tower1.mass', 'mass1')
        root.connect('TowerH2.tower1.mass', 'mass2')
        root.connect('d_paramH1', 'TowerH1.d_param')
        root.connect('d_paramH2', 'TowerH2.d_param')
        root.connect('t_paramH1', 'TowerH1.t_param')
        root.connect('t_paramH2', 'TowerH2.t_param')

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'SNOPT'
        prob.driver.opt_settings['Major iterations limit'] = 1000
        prob.driver.opt_settings['Major optimality tolerance'] = 1E-2

        # --- Objective ---
        prob.driver.add_objective('cost', scaler=1E-6)

        # --- Design Variables ---
        prob.driver.add_desvar('turbineH1', lower=rotor_diameter/2.+10, upper=None, scaler=1.0)
        prob.driver.add_desvar('turbineH2', lower=rotor_diameter/2.+10, upper=None, scaler=1.0)
        # prob.driver.add_desvar('d_paramH1', lower=np.array([1.0, 1.0, d_param[nPoints-1]]), upper=np.array([6.3, 6.3, 6.3]), scaler=1.0)
        # prob.driver.add_desvar('t_paramH1', lower=np.ones(nPoints)*.001, upper=None, scaler=1.0)
        # prob.driver.add_desvar('d_paramH2', lower=np.array([1.0, 1.0, d_param[nPoints-1]-0.0001]), upper=np.array([6.3, 6.3, d_param[nPoints-1]+0.0001]), scaler=1.0)
        # prob.driver.add_desvar('t_paramH2', lower=np.ones(nPoints)*.001, upper=None, scaler=1.0)

        prob.setup()

        prob['L_reinforced'] = L_reinforced
        prob['TowerH1.yaw'] = Toweryaw
        prob['TowerH2.yaw'] = Toweryaw

        # --- material props ---
        prob['d_paramH1'] = d_param
        prob['d_paramH2'] = d_param
        prob['t_paramH1'] = t_param
        prob['t_paramH2'] = t_param
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
        #prob['TowerH1.z_DEL'] = z_DEL*turbineH1/87.6
        #prob['TowerH2.z_DEL'] = z_DEL*turbineH2/87.6
        prob['M_DEL'] = M_DEL
        prob['gamma_fatigue'] = gamma_fatigue
        prob['life'] = life
        prob['m_SN'] = m_SN
        # ---------------

        # --- constraints ---
        prob['gc.min_d_to_t'] = min_d_to_t
        prob['gc.min_taper'] = min_taper
        # ---------------

        prob['H1_H2'] = H1_H2


        prob.run()

        print 'Cost: ', prob['cost']
        self.J = prob.check_total_derivatives(out_stream=None)
        print self.J

    def test_XY(self):
        np.testing.assert_allclose(self.J[('cost', 'turbineH1')]['J_fwd'], self.J[('cost', 'turbineH1')]['J_fd'], 1e-6, 1e-6)

if __name__ == '__main__':
    unittest.main()
