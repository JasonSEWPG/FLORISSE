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
from FLORISSE3D.COE import *
from towerse.tower import TowerSE
from FLORISSE3D.GeneralWindFarmComponents import AEPobj, get_z, getTurbineZ, get_z_DEL
from FLORISSE3D.COE import COEGroup
from FLORISSE3D.GeneralWindFarmComponents import calculate_boundary, SpacingComp,\
            BoundaryComp, get_z, organizeWindSpeeds, getTurbineZ, AEPobj, speedFreq, actualSpeeds
from FLORISSE3D.simpleTower import calcMass
import matplotlib.pyplot as plt
from FLORISSE3D.floris import AEPGroup
from openmdao.api import Problem, Group, IndepVarComp, pyOptSparseDriver, ExecComp, ScipyOptimizer
import time
# from setupOptimization import *
import cPickle as pickle
from setupOptimization import *
from FLORISSE3D.simpleTower import hoopStressEurocode

"""PASS"""
# # Good and Fast
# class TestCostwrtMass(unittest.TestCase):
#
#
#     def setUp(self):
#         nTurbines = 9
#         mass1 = 520000.
#         mass2 = 154500.
#         H1_H2 = np.array([])
#         for i in range(nTurbines/2):
#             H1_H2 = np.append(H1_H2, 0)
#             H1_H2 = np.append(H1_H2, 1)
#         if len(H1_H2) < nTurbines:
#             H1_H2 = np.append(H1_H2, 0)
#
#         prob = Problem()
#         root = prob.root = Group()
#
#         root.add('mass1', IndepVarComp('mass1', mass1), promotes=['*'])
#         root.add('mass2', IndepVarComp('mass2', mass2), promotes=['*'])
#         root.add('farmCost', farmCost(nTurbines), promotes=['*'])
#
#         prob.driver = pyOptSparseDriver()
#         prob.driver.options['optimizer'] = 'SNOPT'
#         prob.driver.opt_settings['Major iterations limit'] = 1000
#         prob.driver.opt_settings['Major optimality tolerance'] = 1E-6
#
#         prob.driver.add_objective('cost', scaler=1E2)
#         prob.driver.add_desvar('mass1', lower=1000., upper=None)
#         prob.driver.add_desvar('mass2', lower=1000., upper=None)
#
#         prob.setup()
#
#         prob['H1_H2'] = H1_H2
#
#
#         prob.run()
#
#         self.J = prob.check_total_derivatives(out_stream=None)
#         print 'Analytic'
#         print self.J[('cost', 'mass1')]['J_fwd']
#         print 'Finite Difference'
#         print self.J[('cost', 'mass1')]['J_fd']
#
#         print 'Analytic'
#         print self.J[('cost', 'mass2')]['J_fwd']
#         print 'Finite Difference'
#         print self.J[('cost', 'mass2')]['J_fd']
#
#     def test_mass(self):
#         np.testing.assert_allclose(self.J[('cost', 'mass1')]['J_fwd'], self.J[('cost', 'mass1')]['J_fd'], 1e-3, 1e-3)
#         np.testing.assert_allclose(self.J[('cost', 'mass2')]['J_fwd'], self.J[('cost', 'mass2')]['J_fd'], 1e-3, 1e-3)


# class TestSpeedwrtHeight(unittest.TestCase):
#
#
#     def setUp(self):
#
#         use_rotor_components = True
#
#         if use_rotor_components:
#             NREL5MWCPCT = pickle.load(open('../../../doc/tune/NREL5MWCPCT_smooth_dict.p'))
#             # print(NREL5MWCPCT)
#             # NREL5MWCPCT = pickle.Unpickler(open('./input_files/NREL5MWCPCT.p')).load()
#             datasize = NREL5MWCPCT['CP'].size
#         else:
#             datasize = 0
#
#         rotor_diameter = 126.4
#
#         nRows = 2
#         nTurbs = nRows**2
#         spacing = 3   # turbine grid spacing in diameters
#         points = np.linspace(start=spacing*rotor_diameter, stop=nRows*spacing*rotor_diameter, num=nRows)
#         xpoints, ypoints = np.meshgrid(points, points)
#         turbineX = np.ndarray.flatten(xpoints)
#         turbineY = np.ndarray.flatten(ypoints)
#
#
#         turbineH1 = 88.25
#         turbineH2 = 120.
#
#         rotorDiameter = np.zeros(nTurbs)
#         axialInduction = np.zeros(nTurbs)
#         Ct = np.zeros(nTurbs)
#         Cp = np.zeros(nTurbs)
#         generatorEfficiency = np.zeros(nTurbs)
#         yaw = np.zeros(nTurbs)
#
#
#         # define initial values
#         for turbI in range(0, nTurbs):
#             rotorDiameter[turbI] = rotor_diameter            # m
#             axialInduction[turbI] = 1.0/3.0
#             Ct[turbI] = 4.0*axialInduction[turbI]*(1.0-axialInduction[turbI])
#             # Cp[turbI] = 0.7737/0.944 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
#             Cp[turbI] = 0.7737 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
#             generatorEfficiency[turbI] = 1.0#0.944
#             yaw[turbI] = 0.     # deg.
#
#         """Define wind flow"""
#         air_density = 1.1716    # kg/m^3
#
#         windSpeeds, windFrequencies, windDirections, nDirections = amaliaWind()
#         windSpeeds = np.array([10.])
#         windFrequencies = np.array([0.25])
#         windDirections = np.array([0.])
#         nDirections = len(windSpeeds)
#
#         shearExp = 0.1
#
#         """set up 3D aspects of wind farm"""
#         H1_H2 = np.array([])
#         for i in range(nTurbs/2):
#             H1_H2 = np.append(H1_H2, 0)
#             H1_H2 = np.append(H1_H2, 1)
#         if len(H1_H2) < nTurbs:
#             H1_H2 = np.append(H1_H2, 0)
#
#         """set up the problem"""
#         prob = Problem()
#         root = prob.root = Group()
#
#         prob.driver = pyOptSparseDriver()
#         prob.driver.options['optimizer'] = 'SNOPT'
#
#         root.add('turbineH1', IndepVarComp('turbineH1', turbineH1), promotes=['*'])
#         root.add('turbineH2', IndepVarComp('turbineH2', turbineH2), promotes=['*'])
#         root.add('Uref', IndepVarComp('Uref', windSpeeds), promotes=['*'])
#         root.add('getTurbineZ', getTurbineZ(nTurbs), promotes=['*'])
#         root.add('AEPGroup', AEPGroup(nTurbs, nDirections=nDirections,
#                     use_rotor_components=use_rotor_components, datasize=datasize, differentiable=True,
#                     optimizingLayout=False, nSamples=0), promotes=['*'])
#
#         prob.driver.add_objective('windSpeeds')
#         prob.driver.add_desvar('Uref', lower=None, upper=None)
#         prob.driver.add_desvar('turbineH1', lower=75., upper=None)
#         prob.driver.add_desvar('turbineH2', lower=75., upper=None)
#
#         prob.setup()
#
#         prob['turbineH1'] = turbineH1
#         prob['turbineH2'] = turbineH2
#         prob['H1_H2'] = H1_H2
#
#         prob['turbineX'] = turbineX
#         prob['turbineY'] = turbineY
#         prob['yaw0'] = yaw
#         prob['ratedPower'] = np.ones_like(turbineX)*5000 # in kw
#
#         # assign values to constant inputs (not design variables)
#         prob['rotorDiameter'] = rotorDiameter
#         prob['axialInduction'] = axialInduction
#         prob['generatorEfficiency'] = generatorEfficiency
#         prob['air_density'] = air_density
#         prob['windDirections'] = np.array([windDirections])
#         prob['windFrequencies'] = np.array([windFrequencies])
#         if use_rotor_components == True:
#             prob['gen_params:windSpeedToCPCT_CP'] = NREL5MWCPCT['CP']
#             prob['gen_params:windSpeedToCPCT_CT'] = NREL5MWCPCT['CT']
#             prob['gen_params:windSpeedToCPCT_wind_speed'] = NREL5MWCPCT['wind_speed']
#         else:
#             prob['Ct_in'] = Ct
#             prob['Cp_in'] = Cp
#         prob['floris_params:cos_spread'] = 1E12
#         prob['shearExp'] = shearExp
#         # prob['Uref'] = windSpeeds
#         prob['zref'] = 90.
#         prob['z0'] = 0.
#
#         prob.run_once()
#
#         self.J = prob.check_total_derivatives(out_stream=None)
#         # print self.J
#         print 'Finite Difference'
#         print self.J[('windSpeeds', 'turbineH1')]['J_fd']
#
#         print 'Analytic'
#         print self.J[('windSpeeds', 'turbineH1')]['J_fwd']
#
#         # print 'Analytic'
#         # print self.J[('windSpeeds', 'turbineH1')]['J_fwd']
#
#
#         # print 'Analytic'
#         # print self.J[('windSpeeds', 'turbineH2')]['J_fwd']
#         # print 'Finite Difference'
#         # print self.J[('windSpeeds', 'turbineH2')]['J_fd']
#
#         # print 'Analytic'
#         # print self.J[('turbineZ', 'turbineH2')]['J_fwd']
#         # print 'Finite Difference'
#         # print self.J[('turbineZ', 'turbineH2')]['J_fd']
#
#     def test_mass(self):
#         np.testing.assert_allclose(self.J[('windSpeeds', 'turbineH1')]['J_fwd'], self.J[('windSpeeds', 'turbineH1')]['J_fd'], 1e-3, 1e-3)
#         np.testing.assert_allclose(self.J[('windSpeeds', 'turbineH2')]['J_fwd'], self.J[('windSpeeds', 'turbineH2')]['J_fd'], 1e-3, 1e-3)
#
# #
# #
# #


# class TestMassComponent(unittest.TestCase):
#
#
#     def setUp(self):
#
#         turbineH = 90.
#         d_param = np.array([6.0,5.5,3.0])
#         t_param = np.array([.044564563,.03121,.0245646])
#
#         """Define tower structural properties"""
#         # --- geometry ----
#         n = 15
#
#         L_reinforced, Toweryaw, E, G, rho, sigma_y, kidx, kx, ky, kz, ktx, kty, ktz, nK, \
#                     midx, m, mIxx, mIyy, mIzz, mIxy, mIxz, mIyz, mrhox, mrhoy, mrhoz, nMass, \
#                     addGravityLoadForExtraMass, wind_zref, wind_z0, shearExp, wind_Uref1, \
#                     plidx1, Fx1, Fy1, Fz1, Mxx1, Myy1, Mzz1, nPL, wind_Uref2, \
#                     plidx2, Fx2, Fy2, Fz2, Mxx2, Myy2, Mzz2, gamma_f, gamma_m, gamma_n, \
#                     gamma_b, z_DEL, M_DEL, nDEL, gamma_fatigue, life, m_SN, min_d_to_t, min_taper \
#                     = setupTower(n)
#
#         nPoints = len(d_param)
#         nFull = n
#         wind = 'PowerWind'
#
#         shearExp = 0.1
#
#         """set up the problem"""
#         prob = Problem()
#         root = prob.root = Group()
#
#         prob.driver = pyOptSparseDriver()
#         prob.driver.options['optimizer'] = 'SNOPT'
#
#         root.add('turbineH', IndepVarComp('turbineH', turbineH), promotes=['*'])
#         root.add('d_param', IndepVarComp('d_param', d_param), promotes=['*'])
#         root.add('t_param', IndepVarComp('t_param', t_param), promotes=['*'])
#         root.add('TowerH1', TowerSE(nPoints, nFull, nK, nMass, nPL, nDEL, wind=wind), promotes=['L_reinforced',
#                         'E','G','sigma_y','kidx','kx','ky','kz','ktx','kty',
#                         'ktz','midx','m','mIxx','mIyy','mIzz','mIxy','mIxz','mIyz',
#                         'mrhox','mrhoy','mrhoz','addGravityLoadForExtraMass','g',
#                         'gamma_f','gamma_m','gamma_n','gamma_b','life','m_SN',
#                         'gc.min_d_to_t','gc.min_taper','M_DEL','gamma_fatigue'])
#
#         root.add('get_z_paramH1', get_z(nPoints))
#         root.add('get_z_fullH1', get_z(n))
#
#         root.add('calcMass', calcMass(), promotes=['*'])
#
#         root.connect('turbineH', 'get_z_paramH1.turbineZ')
#         root.connect('turbineH', 'get_z_fullH1.turbineZ')
#         root.connect('get_z_paramH1.z_param', 'TowerH1.z_param')
#         root.connect('get_z_fullH1.z_param', 'TowerH1.z_full')
#         root.connect('d_param', 'TowerH1.d_param')
#         root.connect('t_param', 'TowerH1.t_param')
#
#         # prob.driver.add_objective('AEP', scaler=1E-8)
#         prob.driver.add_objective('mass', scaler=1)
#
#         prob.driver.add_desvar('turbineH', lower=None, upper=None)
#         prob.driver.add_desvar('d_param', lower=None, upper=None)
#         prob.driver.add_desvar('t_param', lower=None, upper=None)
#
#         prob.setup()
#
#         prob['rho'] = rho[0]
#
#         if wind == "PowerWind":
#             prob['TowerH1.wind1.shearExp'] = shearExp
#             prob['TowerH1.wind2.shearExp'] = shearExp
#
#         """tower structural properties"""
#         # --- geometry ----
#         prob['L_reinforced'] = L_reinforced
#         prob['TowerH1.yaw'] = Toweryaw
#
#         # --- material props ---
#         prob['E'] = E
#         prob['G'] = G
#         prob['TowerH1.tower1.rho'] = rho
#         prob['sigma_y'] = sigma_y
#
#         # --- spring reaction data.  Use float('inf') for rigid constraints. ---
#         prob['kidx'] = kidx
#         prob['kx'] = kx
#         prob['ky'] = ky
#         prob['kz'] = kz
#         prob['ktx'] = ktx
#         prob['kty'] = kty
#         prob['ktz'] = ktz
#
#         # --- extra mass ----
#         prob['midx'] = midx
#         prob['m'] = m
#         prob['mIxx'] = mIxx
#         prob['mIyy'] = mIyy
#         prob['mIzz'] = mIzz
#         prob['mIxy'] = mIxy
#         prob['mIxz'] = mIxz
#         prob['mIyz'] = mIyz
#         prob['mrhox'] = mrhox
#         prob['mrhoy'] = mrhoy
#         prob['mrhoz'] = mrhoz
#         prob['addGravityLoadForExtraMass'] = addGravityLoadForExtraMass
#         # -----------
#
#         # --- wind ---
#         prob['TowerH1.zref'] = wind_zref
#         prob['TowerH1.z0'] = wind_z0
#         # ---------------
#
#         # # --- loading case 1: max Thrust ---
#         prob['TowerH1.wind1.Uref'] = wind_Uref1
#         prob['TowerH1.tower1.plidx'] = plidx1
#         prob['TowerH1.tower1.Fx'] = Fx1
#         prob['TowerH1.tower1.Fy'] = Fy1
#         prob['TowerH1.tower1.Fz'] = Fz1
#         prob['TowerH1.tower1.Mxx'] = Mxx1
#         prob['TowerH1.tower1.Myy'] = Myy1
#         prob['TowerH1.tower1.Mzz'] = Mzz1
#         # # ---------------
#
#         # # --- loading case 2: max Wind Speed ---
#         prob['TowerH1.wind2.Uref'] = wind_Uref2
#         prob['TowerH1.tower2.plidx'] = plidx2
#         prob['TowerH1.tower2.Fx'] = Fx2
#         prob['TowerH1.tower2.Fy'] = Fy2
#         prob['TowerH1.tower2.Fz'] = Fz2
#         prob['TowerH1.tower2.Mxx'] = Mxx2
#         prob['TowerH1.tower2.Myy'] = Myy2
#         prob['TowerH1.tower2.Mzz'] = Mzz2
#
#         prob.run_once()
#
#         print 'MASS MINE: ', prob['mass']
#         print 'TOWERSE MASS: ', prob['TowerH1.tower1.mass']
#
#         self.J = prob.check_total_derivatives(out_stream=None)
#         # print self.J
#         print 'Finite Difference'
#         print self.J[('mass', 'turbineH')]['J_fd']
#         print 'Analytic'
#         print self.J[('mass', 'turbineH')]['J_fwd']
#
#         print 'Finite Difference'
#         print self.J[('mass', 'd_param')]['J_fd']
#         print 'Analytic'
#         print self.J[('mass', 'd_param')]['J_fwd']
#
#         print 'Finite Difference'
#         print self.J[('mass', 't_param')]['J_fd']
#         print 'Analytic'
#         print self.J[('mass', 't_param')]['J_fwd']
#
#     def testWRT_turbineH(self):
#         np.testing.assert_allclose(self.J[('mass', 'turbineH')]['J_fwd'], self.J[('mass', 'turbineH')]['J_fd'], 1e-3, 1e-3)
#
#     def testWRT_d_param(self):
#         np.testing.assert_allclose(self.J[('mass', 'd_param')]['J_fwd'], self.J[('mass', 'd_param')]['J_fd'], 1e-3, 1e-3)
#
#     def testWRT_t_param(self):
#         np.testing.assert_allclose(self.J[('mass', 't_param')]['J_fwd'], self.J[('mass', 't_param')]['J_fd'], 1e-3, 1e-3)


# class TestCostwrtHeight_t_d(unittest.TestCase):


    # def setUp(self):
    #     # --- geometry ----
    #     d_param = np.array([6.0, 4.935, 3.87]) # not going to modify this right now
    #     t_param = np.array([0.027*1.3, 0.023*1.3, 0.019*1.3]) # not going to modify this right now
    #
    #     nTurbs = 9
    #
    #     H1_H2 = np.array([])
    #     for i in range(nTurbs/2):
    #         H1_H2 = np.append(H1_H2, 0)
    #         H1_H2 = np.append(H1_H2, 1)
    #     if len(H1_H2) < nTurbs:
    #         H1_H2 = np.append(H1_H2, 0)
    #
    #     turbineH1 = 154.
    #     turbineH2 = 108.
    #
    #     prob = Problem()
    #     root = prob.root = Group()
    #
    #     root.add('turbineH1', IndepVarComp('turbineH1', turbineH1), promotes=['*'])
    #     root.add('turbineH2', IndepVarComp('turbineH2', turbineH2), promotes=['*'])
    #     root.add('d_paramH1', IndepVarComp('d_paramH1', d_param), promotes=['*'])
    #     root.add('t_paramH1', IndepVarComp('t_paramH1', t_param), promotes=['*'])
    #     root.add('d_paramH2', IndepVarComp('d_paramH2', d_param), promotes=['*'])
    #     root.add('t_paramH2', IndepVarComp('t_paramH2', t_param), promotes=['*'])
    #     root.add('massH1', calcMass())
    #     root.add('massH2', calcMass())
    #     root.add('farmCost', farmCost(nTurbs), promotes=['*'])
    #
    #     root.connect('massH1.mass', 'mass1')
    #     root.connect('massH2.mass', 'mass2')
    #     root.connect('turbineH1', 'massH1.turbineH')
    #     root.connect('turbineH2', 'massH2.turbineH')
    #     root.connect('d_paramH1', 'massH1.d_param')
    #     root.connect('d_paramH2', 'massH2.d_param')
    #     root.connect('t_paramH1', 'massH1.t_param')
    #     root.connect('t_paramH2', 'massH2.t_param')
    #
    #     prob.driver = pyOptSparseDriver()
    #     prob.driver.options['optimizer'] = 'SNOPT'
    #     prob.driver.opt_settings['Major iterations limit'] = 1000
    #     prob.driver.opt_settings['Major optimality tolerance'] = 1E-6
    #
    #     # --- Objective ---
    #     prob.driver.add_objective('cost', scaler=1E-6)
    #
    #     # --- Design Variables ---
    #     prob.driver.add_desvar('turbineH1', lower=None, upper=None, scaler=1.0)
    #     prob.driver.add_desvar('turbineH2', lower=None, upper=None, scaler=1.0)
    #     prob.driver.add_desvar('d_paramH1', lower=None, upper=None, scaler=1.0)
    #     prob.driver.add_desvar('t_paramH1', lower=None, upper=None, scaler=1.0)
    #     prob.driver.add_desvar('d_paramH2', lower=None, upper=None, scaler=1.0)
    #     prob.driver.add_desvar('t_paramH2', lower=None, upper=None, scaler=1.0)
    #
    #
    #     prob.setup()
    #
    #     prob['massH1.rho'] = 8500.0
    #     prob['massH2.rho'] = 8500.0
    #     prob['H1_H2'] = H1_H2
    #
    #     prob.run_once()
    #
    #     self.J = prob.check_total_derivatives(out_stream=None)
    #     print 'H1'
    #     print 'Analytic'
    #     print self.J[('cost', 'turbineH1')]['J_fwd']
    #     print 'Finite Difference'
    #     print self.J[('cost', 'turbineH1')]['J_fd']
    #
    #     print 'H2'
    #     print 'Analytic'
    #     print self.J[('cost', 'turbineH2')]['J_fwd']
    #     print 'Finite Difference'
    #     print self.J[('cost', 'turbineH2')]['J_fd']
    #
    #     print 'D1'
    #     print 'Analytic'
    #     print self.J[('cost', 'd_paramH1')]['J_fwd']
    #     print 'Finite Difference'
    #     print self.J[('cost', 'd_paramH1')]['J_fd']
    #
    #     print 'D2'
    #     print 'Analytic'
    #     print self.J[('cost', 'd_paramH2')]['J_fwd']
    #     print 'Finite Difference'
    #     print self.J[('cost', 'd_paramH2')]['J_fd']
    #
    #     print 'T1'
    #     print 'Analytic'
    #     print self.J[('cost', 't_paramH1')]['J_fwd']
    #     print 'Finite Difference'
    #     print self.J[('cost', 't_paramH1')]['J_fd']
    #
    #     print 'T2'
    #     print 'Analytic'
    #     print self.J[('cost', 't_paramH2')]['J_fwd']
    #     print 'Finite Difference'
    #     print self.J[('cost', 't_paramH2')]['J_fd']
    #
    #
    # def test_cost_H(self):
    #     np.testing.assert_allclose(self.J[('cost', 'turbineH1')]['J_fwd'], self.J[('cost', 'turbineH1')]['J_fd'], 1e-6, 1e-6)
    #     np.testing.assert_allclose(self.J[('cost', 'turbineH2')]['J_fwd'], self.J[('cost', 'turbineH2')]['J_fd'], 1e-6, 1e-6)
    #
    # def test_cost_d(self):
    #     np.testing.assert_allclose(self.J[('cost', 'd_paramH1')]['J_fwd'], self.J[('cost', 'd_paramH1')]['J_fd'], 1e-6, 1e-6)
    #     np.testing.assert_allclose(self.J[('cost', 'd_paramH2')]['J_fwd'], self.J[('cost', 'd_paramH2')]['J_fd'], 1e-6, 1e-6)
    #
    # def test_cost_t(self):
    #     np.testing.assert_allclose(self.J[('cost', 't_paramH1')]['J_fwd'], self.J[('cost', 't_paramH1')]['J_fd'], 1e-6, 1e-6)
    #     np.testing.assert_allclose(self.J[('cost', 't_paramH2')]['J_fwd'], self.J[('cost', 't_paramH2')]['J_fd'], 1e-6, 1e-6)
    #


"""FAIL"""




"""CHECK STILL"""
# class TestZEP(unittest.TestCase):
#
#
#     def setUp(self):
#         use_rotor_components = True
#
#         if use_rotor_components:
#             NREL5MWCPCT = pickle.load(open('../../../doc/tune/NREL5MWCPCT_smooth_dict.p'))
#             # print(NREL5MWCPCT)
#             # NREL5MWCPCT = pickle.Unpickler(open('./input_files/NREL5MWCPCT.p')).load()
#             datasize = NREL5MWCPCT['CP'].size
#         else:
#             datasize = 0
#
#         rotor_diameter = 126.4
#
#         nTurbs = 20
#         turbineX = np.random.rand(nTurbs)*10000
#         turbineY = np.random.rand(nTurbs)*10000
#         turbineX = np.zeros(nTurbs)
#
#         rotorDiameter = np.zeros(nTurbs)
#         axialInduction = np.zeros(nTurbs)
#         Ct = np.zeros(nTurbs)
#         Cp = np.zeros(nTurbs)
#         generatorEfficiency = np.zeros(nTurbs)
#         yaw = np.zeros(nTurbs)
#
#         # define initial values
#         for turbI in range(0, nTurbs):
#             rotorDiameter[turbI] = rotor_diameter            # m
#             axialInduction[turbI] = 1.0/3.0
#             Ct[turbI] = 4.0*axialInduction[turbI]*(1.0-axialInduction[turbI])
#             # Cp[turbI] = 0.7737/0.944 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
#             Cp[turbI] = 0.7737 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
#             generatorEfficiency[turbI] = 1.0#0.944
#             yaw[turbI] = 0.     # deg.
#
#         minSpacing = 2
#
#         """Define wind flow"""
#         air_density = 1.1716    # kg/m^3
#
#         windData = 'Manual'
#
#         """Amalia Wind Arrays"""
#         if windData == "Amalia":
#             windSpeeds, windFrequencies, windDirections, nDirections = amaliaWind()
#
#         """Manual Wind Arrays"""
#         if windData == "Manual":
#             nDirections = 1
#             windSpeeds = np.ones(nDirections)*10.
#             # print nDirections
#             # windDirections = np.linspace(0,360-360/nDirections, nDirections)
#             # windFrequencies = np.ones(len(windSpeeds))/len(windSpeeds)
#             windDirections = np.array([0.])
#             windFrequencies = np.array([1.])
#
#
#         nIntegrationPoints = 1 #Number of points in wind effective wind speed integral
#
#         wind = 'PowerWind'
#
#         shearExp = 0.15
#
#         """set up 3D aspects of wind farm"""
#         diff = 0.
#         turbineH1 = 90.
#         turbineH2 = 91.
#         H1_H2 = np.array([])
#         for i in range(nTurbs/2):
#             H1_H2 = np.append(H1_H2, 0)
#             H1_H2 = np.append(H1_H2, 1)
#         if len(H1_H2) < nTurbs:
#             H1_H2 = np.append(H1_H2, 0)
#
#         """set up the problem"""
#         prob = Problem()
#         root = prob.root = Group()
#
#         # root.deriv_options['step_size'] = 1.E-6
#         # root.deriv_options['step_calc'] = 'relative'
#         # root.deriv_options['form'] = 'forward'
#
#
#         root.add('turbineH1', IndepVarComp('turbineH1', turbineH1), promotes=['*'])
#         root.add('turbineH2', IndepVarComp('turbineH2', turbineH2), promotes=['*'])
#         #This component takes turbineH1, turbineH2, and H1_H2 and gives back an array
#         #of turbineZ
#         root.add('getTurbineZ', getTurbineZ(nTurbs), promotes=['*'])
#         #These components adjust the parameterized z locations for TowerSE calculations
#         #with respect to turbineZ
#         root.add('AEPGroup', AEPGroup(nTurbs, nDirections=nDirections,
#                     use_rotor_components=use_rotor_components, datasize=datasize, differentiable=True,
#                     optimizingLayout=False, nSamples=0), promotes=['*'])
#
#         #For Constraints
#         root.add('spacing_comp', SpacingComp(nTurbines=nTurbs), promotes=['*'])
#
#         # add constraint definitions
#         root.add('spacing_con', ExecComp('sc = wtSeparationSquared-(minSpacing*rotorDiameter[0])**2',
#                                      minSpacing=minSpacing, rotorDiameter=np.zeros(nTurbs),
#                                      sc=np.zeros(((nTurbs-1.)*nTurbs/2.)),
#                                      wtSeparationSquared=np.zeros(((nTurbs-1.)*nTurbs/2.))),
#                                      promotes=['*'])
#
#         prob.driver = pyOptSparseDriver()
#         prob.driver.options['optimizer'] = 'SNOPT'
#         prob.driver.opt_settings['Major iterations limit'] = 1000
#         prob.driver.opt_settings['Major optimality tolerance'] = 1E-6
#         # prob.driver = ScipyOptimizer()
#         # prob.driver.options['optimizer'] = 'SLSQP'
#
#         # --- Objective ---
#         prob.driver.add_objective('AEP', scaler=1.0E-8)
#
#         # # --- Design Variables ---
#         prob.driver.add_desvar('turbineH1', lower=rotor_diameter/2.+10, upper=None, scaler=1.0E-1)
#         prob.driver.add_desvar('turbineH2', lower=rotor_diameter/2.+10, upper=None, scaler=1.0E-1)
#         prob.driver.add_desvar('turbineX', lower=None, upper=None, scaler=1.0E-2)
#         prob.driver.add_desvar('turbineY', lower=None, upper=None, scaler=1.0E-2)
#
#         # spacing constraint
#         prob.driver.add_constraint('sc', lower=np.zeros(((nTurbs-1.)*nTurbs/2.)), scaler=1.0/rotor_diameter)
#
#         # ----------------------
#
#         prob.root.ln_solver.options['single_voi_relevance_reduction'] = True
#
#         start = time.time()
#         prob.setup()
#
#         """run the problem"""
#
#         if wind == "PowerWind":
#             prob['shearExp'] = shearExp
#         prob['turbineH1'] = turbineH1
#         prob['turbineH2'] = turbineH2
#         prob['H1_H2'] = H1_H2
#
#         prob['turbineX'] = turbineX
#         prob['turbineY'] = turbineY
#         prob['yaw0'] = yaw
#         prob['ratedPower'] = np.ones_like(turbineX)*5000 # in kw
#
#
#         # assign values to constant inputs (not design variables)
#         prob['nIntegrationPoints'] = nIntegrationPoints
#         prob['rotorDiameter'] = rotorDiameter
#         prob['axialInduction'] = axialInduction
#         prob['generatorEfficiency'] = generatorEfficiency
#         prob['air_density'] = air_density
#         prob['windDirections'] = np.array([windDirections])
#         prob['windFrequencies'] = np.array([windFrequencies])
#         prob['Uref'] = windSpeeds
#         if use_rotor_components == True:
#             prob['gen_params:windSpeedToCPCT_CP'] = NREL5MWCPCT['CP']
#             prob['gen_params:windSpeedToCPCT_CT'] = NREL5MWCPCT['CT']
#             prob['gen_params:windSpeedToCPCT_wind_speed'] = NREL5MWCPCT['wind_speed']
#         else:
#             prob['Ct_in'] = Ct
#             prob['Cp_in'] = Cp
#         prob['floris_params:cos_spread'] = 1E12
#
#         prob.run_once()
#
#         self.J = prob.check_total_derivatives(out_stream=None)
#         print "###################################################3"
#
#         # print self.J
#         print 'wrt turbineX'
#         print 'Analytic'
#         print self.J[('AEP', 'turbineX')]['J_fwd']
#         print 'Finite Difference'
#         print self.J[('AEP', 'turbineX')]['J_fd']
#
#         #
#         # print 'wrt turbineY'
#         # # print 'Analytic'
#         # # print self.J[('AEP', 'turbineY')]['J_fwd']
#         # print 'Finite Difference'
#         # print self.J[('AEP', 'turbineY')]['J_fd']
#         #
#         # print 'wrt turbineH1'
#         # print 'Analytic'
#         # print self.J[('AEP', 'turbineH1')]['J_fwd']
#         # print 'Finite Difference'
#         # print self.J[('AEP', 'turbineH1')]['J_fd']
#
#         print turbineX
#         print turbineY
#         plt.plot(turbineX,turbineY,'ob')
#         plt.show()
#         #
#         # print 'wrt turbineH2'
#         # # print 'Analytic'
#         # # print self.J[('AEP', 'turbineH2')]['J_fwd']
#         # print 'Finite Difference'
#         # print self.J[('AEP', 'turbineH2')]['J_fd']
#
#     # def test_X(self):
#         # np.testing.assert_allclose(self.J[('AEP', 'turbineX')]['J_fwd'], self.J[('AEP', 'turbineX')]['J_fd'], 1e-2, 1e-2)
#
#     # def test_Y(self):
#     #     np.testing.assert_allclose(self.J[('AEP', 'turbineY')]['J_rev'], self.J[('AEP', 'turbineY')]['J_fd'], 1e-4, 1e-4)
#     #
#     def testZ(self):
#         np.testing.assert_allclose(self.J[('AEP', 'turbineH1')]['J_fwd'], self.J[('AEP', 'turbineH1')]['J_fd'], 1e-6, 1e-6)
#         np.testing.assert_allclose(self.J[('AEP', 'turbineH2')]['J_fwd'], self.J[('AEP', 'turbineH2')]['J_fd'], 1e-6, 1e-6)
#
#
#



# class TestTotalDerivatives(unittest.TestCase):
#
#     def setUp(self):
#
#         use_rotor_components = True
#
#         if use_rotor_components:
#             NREL5MWCPCT = pickle.load(open('../../../doc/tune/NREL5MWCPCT_smooth_dict.p'))
#             # print(NREL5MWCPCT)
#             # NREL5MWCPCT = pickle.Unpickler(open('./input_files/NREL5MWCPCT.p')).load()
#             datasize = NREL5MWCPCT['CP'].size
#         else:
#             datasize = 0
#
#         rotor_diameter = 126.4
#
#         nTurbs = 25
#
#
#         rotorDiameter = np.zeros(nTurbs)
#         axialInduction = np.zeros(nTurbs)
#         Ct = np.zeros(nTurbs)
#         Cp = np.zeros(nTurbs)
#         generatorEfficiency = np.zeros(nTurbs)
#         yaw = np.zeros(nTurbs)
#
#
#         # define initial values
#         for turbI in range(0, nTurbs):
#             rotorDiameter[turbI] = rotor_diameter            # m
#             axialInduction[turbI] = 1.0/3.0
#             Ct[turbI] = 4.0*axialInduction[turbI]*(1.0-axialInduction[turbI])
#             # Cp[turbI] = 0.7737/0.944 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
#             Cp[turbI] = 0.7737 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
#             generatorEfficiency[turbI] = 1.0#0.944
#             yaw[turbI] = 0.     # deg.
#
#         minSpacing = 2
#
#
#
#         """Define wind flow"""
#         air_density = 1.1716    # kg/m^3
#
#         windData = "Manual"
#
#         """Amalia Wind Arrays"""
#         if windData == "Amalia":
#             windSpeeds, windFrequencies, windDirections, nDirections = amaliaWind()
#
#         """Manual Wind Arrays"""
#         if windData == "Manual":
#             nDirections = 1
#             windSpeeds = np.ones(nDirections)*10.
#             # print nDirections
#             # windDirections = np.linspace(0,360-360/nDirections, nDirections)
#             # windFrequencies = np.ones(len(windSpeeds))/len(windSpeeds)
#             windDirections = np.array([38.])
#             windFrequencies = np.array([1.])
#
#         nIntegrationPoints = 1 #Number of points in wind effective wind speed integral
#
#         """Define tower structural properties"""
#         # --- geometry ----
#         d_param = np.array([6.0, 4.935, 3.87])
#         t_param = np.array([0.027*1.3, 0.023*1.3, 0.019*1.3])
#         n = 15
#
#         L_reinforced, Toweryaw, E, G, rho, sigma_y, kidx, kx, ky, kz, ktx, kty, ktz, nK, \
#                     midx, m, mIxx, mIyy, mIzz, mIxy, mIxz, mIyz, mrhox, mrhoy, mrhoz, nMass, \
#                     addGravityLoadForExtraMass, wind_zref, wind_z0, shearExp, wind_Uref1, \
#                     plidx1, Fx1, Fy1, Fz1, Mxx1, Myy1, Mzz1, nPL, wind_Uref2, \
#                     plidx2, Fx2, Fy2, Fz2, Mxx2, Myy2, Mzz2, gamma_f, gamma_m, gamma_n, \
#                     gamma_b, z_DEL, M_DEL, nDEL, gamma_fatigue, life, m_SN, min_d_to_t, min_taper \
#                     = setupTower(n, rotor_diameter)
#
#         nPoints = len(d_param)
#         nFull = n
#         wind = 'PowerWind'
#
#         shearExp = 0.1
#
#         nRows = 5
#         nTurbs = nRows**2
#         spacing = 3   # turbine grid spacing in diameters
#         points = np.linspace(start=spacing*rotor_diameter, stop=nRows*spacing*rotor_diameter, num=nRows)
#         xpoints, ypoints = np.meshgrid(points, points)
#         turbineX = np.ndarray.flatten(xpoints)
#         turbineY = np.ndarray.flatten(ypoints)
#         # turbineX = np.array([0,1000,0,1000,500])*0.6
#         # turbineY = np.array([0,0,1000,1000,500])*0.6
#         # nTurbs = 5
#
#         # generate boundary constraint
#         locations = np.zeros((len(turbineX),2))
#         for i in range(len(turbineX)):
#             locations[i][0] = turbineX[i]
#             locations[i][1] = turbineY[i]
#         print locations
#         boundaryVertices, boundaryNormals = calculate_boundary(locations)
#         nVertices = boundaryVertices.shape[0]
#
#         """set up 3D aspects of wind farm"""
#         diff = 0.
#         turbineH1 = 113.
#         turbineH2 = 90.
#         H1_H2 = np.array([])
#         for i in range(nTurbs/2):
#             H1_H2 = np.append(H1_H2, 0)
#             H1_H2 = np.append(H1_H2, 1)
#         if len(H1_H2) < nTurbs:
#             H1_H2 = np.append(H1_H2, 0)
#
#         """set up the problem"""
#         prob = Problem()
#         root = prob.root = Group()
#
#         root.fd_options['step_size'] = 1.E-6
#         root.fd_options['step_type'] = 'relative'
#         # root.fd_options['force_fd'] = True
#
#         root.add('turbineH1', IndepVarComp('turbineH1', turbineH1), promotes=['*'])
#         root.add('turbineH2', IndepVarComp('turbineH2', turbineH2), promotes=['*'])
#         root.add('d_paramH1', IndepVarComp('d_paramH1', d_param), promotes=['*'])
#         root.add('t_paramH1', IndepVarComp('t_paramH1', t_param), promotes=['*'])
#         root.add('d_paramH2', IndepVarComp('d_paramH2', d_param), promotes=['*'])
#         root.add('t_paramH2', IndepVarComp('t_paramH2', t_param), promotes=['*'])
#         #This component takes turbineH1, turbineH2, and H1_H2 and gives back an array
#         #of turbineZ
#         root.add('getTurbineZ', getTurbineZ(nTurbs), promotes=['*'])
#         #These components adjust the parameterized z locations for TowerSE calculations
#         #with respect to turbineZ
#         root.add('get_z_paramH1', get_z(nPoints))
#         root.add('get_z_paramH2', get_z(nPoints))
#         root.add('get_z_fullH1', get_z(n))
#         root.add('get_z_fullH2', get_z(n))
#         root.add('AEPGroup', AEPGroup(nTurbs, nDirections=nDirections,
#                     use_rotor_components=use_rotor_components, datasize=datasize, differentiable=True,
#                     optimizingLayout=False, nSamples=0), promotes=['*'])
#         root.add('COEGroup', COEGroup(nTurbs), promotes=['*'])
#         root.add('AEPobj', AEPobj(), promotes=['*'])
#
#         #For Constraints
#         root.add('TowerH1', TowerSE(nPoints, nFull, nK, nMass, nPL, nDEL, wind=wind), promotes=['L_reinforced',
#                             'E','G','sigma_y','kidx','kx','ky','kz','ktx','kty',
#                             'ktz','midx','m','mIxx','mIyy','mIzz','mIxy','mIxz','mIyz',
#                             'mrhox','mrhoy','mrhoz','addGravityLoadForExtraMass','g',
#                             'gamma_f','gamma_m','gamma_n','gamma_b','life','m_SN',
#                             'gc.min_d_to_t','gc.min_taper','M_DEL','gamma_fatigue'])
#         root.add('TowerH2', TowerSE(nPoints, nFull, nK, nMass, nPL, nDEL, wind=wind), promotes=['L_reinforced',
#                             'E','G','sigma_y','kidx','kx','ky','kz','ktx','kty',
#                             'ktz','midx','m','mIxx','mIyy','mIzz','mIxy','mIxz','mIyz',
#                             'mrhox','mrhoy','mrhoz','addGravityLoadForExtraMass','g',
#                             'gamma_f','gamma_m','gamma_n','gamma_b','life','m_SN',
#                             'gc.min_d_to_t','gc.min_taper','M_DEL','gamma_fatigue'])
#         root.add('spacing_comp', SpacingComp(nTurbines=nTurbs), promotes=['*'])
#
#         # add constraint definitions
#         root.add('spacing_con', ExecComp('sc = wtSeparationSquared-(minSpacing*rotorDiameter[0])**2',
#                                      minSpacing=minSpacing, rotorDiameter=np.zeros(nTurbs),
#                                      sc=np.zeros(((nTurbs-1.)*nTurbs/2.)),
#                                      wtSeparationSquared=np.zeros(((nTurbs-1.)*nTurbs/2.))),
#                                      promotes=['*'])
#
#         if nVertices > 0:
#             # add component that enforces a convex hull wind farm boundary
#             root.add('boundary_con', BoundaryComp(nVertices=nVertices, nTurbines=nTurbs), promotes=['*'])
#
#         root.connect('turbineH1', 'get_z_paramH1.turbineZ')
#         root.connect('turbineH2', 'get_z_paramH2.turbineZ')
#         root.connect('turbineH1', 'get_z_fullH1.turbineZ')
#         root.connect('turbineH2', 'get_z_fullH2.turbineZ')
#         root.connect('get_z_paramH1.z_param', 'TowerH1.z_param')
#         root.connect('get_z_fullH1.z_param', 'TowerH1.z_full')
#         root.connect('get_z_paramH2.z_param', 'TowerH2.z_param')
#         root.connect('get_z_fullH2.z_param', 'TowerH2.z_full')
#         root.connect('TowerH1.tower1.mass', 'mass1')
#         root.connect('TowerH2.tower1.mass', 'mass2')
#         root.connect('d_paramH1', 'TowerH1.d_param')
#         root.connect('d_paramH2', 'TowerH2.d_param')
#         root.connect('t_paramH1', 'TowerH1.t_param')
#         root.connect('t_paramH2', 'TowerH2.t_param')
#
#         prob.driver = pyOptSparseDriver()
#         prob.driver.options['optimizer'] = 'SNOPT'
#         prob.driver.opt_settings['Major iterations limit'] = 1000
#         prob.driver.opt_settings['Major optimality tolerance'] = 1E-6
#         # prob.driver = ScipyOptimizer()
#         # prob.driver.options['optimizer'] = 'SLSQP'
#
#         # --- Objective ---
#         prob.driver.add_objective('COE', scaler=1.0E-1)
#
#         # # --- Design Variables ---
#         prob.driver.add_desvar('turbineH1', lower=rotor_diameter/2.+10, upper=None, scaler=1.0)
#         prob.driver.add_desvar('turbineH2', lower=rotor_diameter/2.+10, upper=None, scaler=1.0)
#         prob.driver.add_desvar('turbineX', lower=None, upper=None, scaler=1.0)
#         prob.driver.add_desvar('turbineY', lower=None, upper=None, scaler=1.0)
#         prob.driver.add_desvar('d_paramH1', lower=np.array([1.0, 1.0, d_param[nPoints-1]]), upper=np.ones(nPoints)*4.3, scaler=1.0)
#         prob.driver.add_desvar('t_paramH1', lower=np.ones(nPoints)*.001, upper=np.ones(nPoints)*0.05, scaler=1.0)
#         prob.driver.add_desvar('d_paramH2', lower=np.array([1.0, 1.0, d_param[nPoints-1]]), upper=np.ones(nPoints)*4.3, scaler=1.0)
#         prob.driver.add_desvar('t_paramH2', lower=np.ones(nPoints)*.001, upper=np.ones(nPoints)*0.05, scaler=1.0)
#
#         # --- Constraints ---
#         # TowerH1 structure
#         prob.driver.add_constraint('TowerH1.tower1.stress', upper=1.0)
#         prob.driver.add_constraint('TowerH1.tower2.stress', upper=1.0)
#         prob.driver.add_constraint('TowerH1.tower1.global_buckling', upper=1.0)
#         prob.driver.add_constraint('TowerH1.tower2.global_buckling', upper=1.0)
#         prob.driver.add_constraint('TowerH1.tower1.shell_buckling', upper=1.0)
#         prob.driver.add_constraint('TowerH1.tower2.shell_buckling', upper=1.0)
#         prob.driver.add_constraint('TowerH1.tower1.damage', upper=1.0)
#         prob.driver.add_constraint('TowerH1.gc.weldability', upper=0.0)
#         prob.driver.add_constraint('TowerH1.gc.manufacturability', upper=0.0)
#         freq1p = 0.2  # 1P freq in Hz
#         prob.driver.add_constraint('TowerH1.tower1.f1', lower=1.1*freq1p)
#
#         # #TowerH2 structure
#         prob.driver.add_constraint('TowerH2.tower1.stress', upper=1.0)
#         prob.driver.add_constraint('TowerH2.tower2.stress', upper=1.0)
#         prob.driver.add_constraint('TowerH2.tower1.global_buckling', upper=1.0)
#         prob.driver.add_constraint('TowerH2.tower2.global_buckling', upper=1.0)
#         prob.driver.add_constraint('TowerH2.tower1.shell_buckling', upper=1.0)
#         prob.driver.add_constraint('TowerH2.tower2.shell_buckling', upper=1.0)
#         prob.driver.add_constraint('TowerH2.gc.weldability', upper=0.0)
#         prob.driver.add_constraint('TowerH2.gc.manufacturability', upper=0.0)
#         freq1p = 0.2  # 1P freq in Hz
#         prob.driver.add_constraint('TowerH2.tower1.f1', lower=1.1*freq1p)
#
#         # boundary constraint (convex hull)
#         prob.driver.add_constraint('boundaryDistances', lower=np.zeros(nVertices*nTurbs), scaler=1.0)
#         # spacing constraint
#         prob.driver.add_constraint('sc', lower=np.zeros(((nTurbs-1.)*nTurbs/2.)), scaler=1.0/rotor_diameter)
#
#         # ----------------------
#
#         prob.root.ln_solver.options['single_voi_relevance_reduction'] = True
#
#         start = time.time()
#         prob.setup()
#
#         """run the problem"""
#
#         if wind == "PowerWind":
#             prob['TowerH1.wind1.shearExp'] = shearExp
#             prob['TowerH1.wind2.shearExp'] = shearExp
#             prob['TowerH2.wind1.shearExp'] = shearExp
#             prob['TowerH2.wind2.shearExp'] = shearExp
#             prob['shearExp'] = shearExp
#         prob['turbineH1'] = turbineH1
#         prob['turbineH2'] = turbineH2
#         prob['H1_H2'] = H1_H2
#         prob['diameter'] = rotor_diameter
#
#         prob['turbineX'] = turbineX
#         prob['turbineY'] = turbineY
#         prob['yaw0'] = yaw
#         prob['ratedPower'] = np.ones_like(turbineX)*5000 # in kw
#
#         prob['boundaryVertices'] = boundaryVertices
#         prob['boundaryNormals'] = boundaryNormals
#
#         # assign values to constant inputs (not design variables)
#         prob['nIntegrationPoints'] = nIntegrationPoints
#         prob['rotorDiameter'] = rotorDiameter
#         prob['axialInduction'] = axialInduction
#         prob['generatorEfficiency'] = generatorEfficiency
#         prob['air_density'] = air_density
#         prob['windDirections'] = np.array([windDirections])
#         prob['windFrequencies'] = np.array([windFrequencies])
#         prob['Uref'] = windSpeeds
#         if use_rotor_components == True:
#             prob['gen_params:windSpeedToCPCT_CP'] = NREL5MWCPCT['CP']
#             prob['gen_params:windSpeedToCPCT_CT'] = NREL5MWCPCT['CT']
#             prob['gen_params:windSpeedToCPCT_wind_speed'] = NREL5MWCPCT['wind_speed']
#         else:
#             prob['Ct_in'] = Ct
#             prob['Cp_in'] = Cp
#         prob['floris_params:cos_spread'] = 1E12
#         prob['zref'] = wind_zref
#         prob['z0'] = wind_z0       # turns off cosine spread (just needs to be very large)
#
#         """tower structural properties"""
#         # --- geometry ----
#         # prob['d_param'] = d_param
#         # prob['t_param'] = t_param
#
#         prob['L_reinforced'] = L_reinforced
#         prob['TowerH1.yaw'] = Toweryaw
#         prob['TowerH2.yaw'] = Toweryaw
#
#         # --- material props ---
#         prob['E'] = E
#         prob['G'] = G
#         prob['TowerH1.tower1.rho'] = rho
#         prob['TowerH2.tower1.rho'] = rho
#         prob['sigma_y'] = sigma_y
#
#         # --- spring reaction data.  Use float('inf') for rigid constraints. ---
#         prob['kidx'] = kidx
#         prob['kx'] = kx
#         prob['ky'] = ky
#         prob['kz'] = kz
#         prob['ktx'] = ktx
#         prob['kty'] = kty
#         prob['ktz'] = ktz
#
#         # --- extra mass ----
#         prob['midx'] = midx
#         prob['m'] = m
#         prob['mIxx'] = mIxx
#         prob['mIyy'] = mIyy
#         prob['mIzz'] = mIzz
#         prob['mIxy'] = mIxy
#         prob['mIxz'] = mIxz
#         prob['mIyz'] = mIyz
#         prob['mrhox'] = mrhox
#         prob['mrhoy'] = mrhoy
#         prob['mrhoz'] = mrhoz
#         prob['addGravityLoadForExtraMass'] = addGravityLoadForExtraMass
#         # -----------
#
#         # --- wind ---
#         prob['TowerH1.zref'] = wind_zref
#         prob['TowerH2.zref'] = wind_zref
#         prob['TowerH1.z0'] = wind_z0
#         prob['TowerH2.z0'] = wind_z0
#         # ---------------
#
#         # # --- loading case 1: max Thrust ---
#         prob['TowerH1.wind1.Uref'] = wind_Uref1
#         prob['TowerH1.tower1.plidx'] = plidx1
#         prob['TowerH1.tower1.Fx'] = Fx1
#         prob['TowerH1.tower1.Fy'] = Fy1
#         prob['TowerH1.tower1.Fz'] = Fz1
#         prob['TowerH1.tower1.Mxx'] = Mxx1
#         prob['TowerH1.tower1.Myy'] = Myy1
#         prob['TowerH1.tower1.Mzz'] = Mzz1
#
#         prob['TowerH2.wind1.Uref'] = wind_Uref1
#         prob['TowerH2.tower1.plidx'] = plidx1
#         prob['TowerH2.tower1.Fx'] = Fx1
#         prob['TowerH2.tower1.Fy'] = Fy1
#         prob['TowerH2.tower1.Fz'] = Fz1
#         prob['TowerH2.tower1.Mxx'] = Mxx1
#         prob['TowerH2.tower1.Myy'] = Myy1
#         prob['TowerH2.tower1.Mzz'] = Mzz1
#         # # ---------------
#
#         # # --- loading case 2: max Wind Speed ---
#         prob['TowerH1.wind2.Uref'] = wind_Uref2
#         prob['TowerH1.tower2.plidx'] = plidx2
#         prob['TowerH1.tower2.Fx'] = Fx2
#         prob['TowerH1.tower2.Fy'] = Fy2
#         prob['TowerH1.tower2.Fz'] = Fz2
#         prob['TowerH1.tower2.Mxx'] = Mxx2
#         prob['TowerH1.tower2.Myy'] = Myy2
#         prob['TowerH1.tower2.Mzz'] = Mzz2
#
#         prob['TowerH2.wind2.Uref'] = wind_Uref2
#         prob['TowerH2.tower2.plidx'] = plidx2
#         prob['TowerH2.tower2.Fx'] = Fx2
#         prob['TowerH2.tower2.Fy'] = Fy2
#         prob['TowerH2.tower2.Fz'] = Fz2
#         prob['TowerH2.tower2.Mxx'] = Mxx2
#         prob['TowerH2.tower2.Myy'] = Myy2
#         prob['TowerH2.tower2.Mzz'] = Mzz2
#         # # ---------------
#
#         # --- safety factors ---
#         prob['gamma_f'] = gamma_f
#         prob['gamma_m'] = gamma_m
#         prob['gamma_n'] = gamma_n
#         prob['gamma_b'] = gamma_b
#         # ---------------
#
#         # --- fatigue ---
#         prob['gamma_fatigue'] = gamma_fatigue
#         prob['life'] = life
#         prob['m_SN'] = m_SN
#         # ---------------
#
#         # --- constraints ---
#         prob['gc.min_d_to_t'] = min_d_to_t
#         prob['gc.min_taper'] = min_taper
#         # ---------------
#
#         prob.run_once()
#
#         print 'Mass H1: ', prob['TowerH1.tower1.mass']
#         print 'Mass H2: ', prob['TowerH2.tower1.mass']
#         print 'd: ', prob['d_paramH1']
#         print 't: ', prob['t_paramH1']
#         self.J = prob.check_total_derivatives(out_stream=None)
#         print '****************************************************************'
#         print '^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^'
#         # print 'H1 fwd: ', self.J[('COE', 'turbineH1')]['J_fwd']
#         print 'H1 fd: ', self.J[('COE', 'turbineH1')]['J_fd']
#         # print 'H2 fwd: ', self.J[('COE', 'turbineH2')]['J_fwd']
#         print 'H2 fd: ', self.J[('COE', 'turbineH2')]['J_fd']
#         # print 'X fwd: ', self.J[('COE', 'turbineX')]['J_fwd']
#         print 'X fd: ', self.J[('COE', 'turbineX')]['J_fd']
#         # print 'Y fwd: ', self.J[('COE', 'turbineY')]['J_fwd']
#         print 'Y fd: ', self.J[('COE', 'turbineY')]['J_fd']
#         # print 'd1 fwd: ', self.J[('COE', 'd_paramH1')]['J_fwd']
#         print 'd1 fd: ', self.J[('COE', 'd_paramH1')]['J_fd']
#         # print 'd2 fwd: ', self.J[('COE', 'd_paramH2')]['J_fwd']
#         print 'd2 fd: ', self.J[('COE', 'd_paramH2')]['J_fd']
#         # print 't1 fwd: ', self.J[('COE', 't_paramH1')]['J_fwd']
#         print 't1 fd: ', self.J[('COE', 't_paramH1')]['J_fd']
#         # print 't2 fwd: ', self.J[('COE', 't_paramH2')]['J_fwd']
#         print 't2 fd: ', self.J[('COE', 't_paramH2')]['J_fd']
#         print 'T1, stress: ', self.J[('COE', 'TowerH1.tower1.stress')]['J_fd']
#
#     def test_COE_H1(self):
#         np.testing.assert_allclose(self.J[('COE', 'turbineH1')]['J_fwd'], self.J[('COE', 'turbineH1')]['J_fd'], 1e-6, 1e-6)

    # def test_COE_H2(self):
    #     np.testing.assert_allclose(self.J[('COE', 'turbineH2')]['J_fwd'], self.J[('COE', 'turbineH2')]['J_fd'], 1e-6, 1e-6)
    #
    # def test_COE_d_t(self):
    #     np.testing.assert_allclose(self.J[('COE', 'd_paramH1')]['J_fwd'], self.J[('COE', 'd_paramH1')]['J_fd'], 1e-6, 1e-6)
    #     np.testing.assert_allclose(self.J[('COE', 't_paramH1')]['J_fwd'], self.J[('COE', 't_paramH1')]['J_fd'], 1e-6, 1e-6)
    #
    #     np.testing.assert_allclose(self.J[('COE', 't_paramH2')]['J_fwd'], self.J[('COE', 't_paramH2')]['J_fd'], 1e-6, 1e-6)
    #     np.testing.assert_allclose(self.J[('COE', 't_paramH2')]['J_fwd'], self.J[('COE', 't_paramH2')]['J_fd'], 1e-6, 1e-6)
    #
    # def test_COE_X_Y(self):
    #     np.testing.assert_allclose(self.J[('COE', 'turbineX')]['J_fwd'], self.J[('COE', 'turbineX')]['J_fd'], 1e-6, 1e-6)
    #     np.testing.assert_allclose(self.J[('COE', 'turbineY')]['J_fwd'], self.J[('COE', 'turbineY')]['J_fd'], 1e-6, 1e-6)



# class TestTotalDerivatives(unittest.TestCase):
#
#     def setUp(self):
#
#         use_rotor_components = True
#
#         if use_rotor_components:
#             NREL5MWCPCT = pickle.load(open('../../../doc/tune/NREL5MWCPCT_smooth_dict.p'))
#             # print(NREL5MWCPCT)
#             # NREL5MWCPCT = pickle.Unpickler(open('./input_files/NREL5MWCPCT.p')).load()
#             datasize = NREL5MWCPCT['CP'].size
#         else:
#             datasize = 0
#
#         rotor_diameter = 126.4
#
#         nTurbs = 25
#
#
#         rotorDiameter = np.zeros(nTurbs)
#         axialInduction = np.zeros(nTurbs)
#         Ct = np.zeros(nTurbs)
#         Cp = np.zeros(nTurbs)
#         generatorEfficiency = np.zeros(nTurbs)
#         yaw = np.zeros(nTurbs)
#
#
#         # define initial values
#         for turbI in range(0, nTurbs):
#             rotorDiameter[turbI] = rotor_diameter            # m
#             axialInduction[turbI] = 1.0/3.0
#             Ct[turbI] = 4.0*axialInduction[turbI]*(1.0-axialInduction[turbI])
#             # Cp[turbI] = 0.7737/0.944 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
#             Cp[turbI] = 0.7737 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
#             generatorEfficiency[turbI] = 1.0#0.944
#             yaw[turbI] = 0.     # deg.
#
#         minSpacing = 2
#
#
#
#         """Define wind flow"""
#         air_density = 1.1716    # kg/m^3
#
#         windData = "Amalia"
#
#         """Amalia Wind Arrays"""
#         if windData == "Amalia":
#             windSpeeds, windFrequencies, windDirections, nDirections = amaliaWind()
#
#         """Manual Wind Arrays"""
#         if windData == "Manual":
#             nDirections = 1
#             windSpeeds = np.ones(nDirections)*10.
#             # print nDirections
#             # windDirections = np.linspace(0,360-360/nDirections, nDirections)
#             # windFrequencies = np.ones(len(windSpeeds))/len(windSpeeds)
#             windDirections = np.array([38.])
#             windFrequencies = np.array([1.])
#
#         nIntegrationPoints = 1 #Number of points in wind effective wind speed integral
#
#         """Define tower structural properties"""
#         # --- geometry ----
#         d_param = np.array([6.0, 4.935, 3.87])
#         t_param = np.array([0.027*1.3, 0.023*1.3, 0.019*1.3])
#         n = 15
#
#         L_reinforced, Toweryaw, E, G, rho, sigma_y, kidx, kx, ky, kz, ktx, kty, ktz, nK, \
#                     midx, m, mIxx, mIyy, mIzz, mIxy, mIxz, mIyz, mrhox, mrhoy, mrhoz, nMass, \
#                     addGravityLoadForExtraMass, wind_zref, wind_z0, shearExp, wind_Uref1, \
#                     plidx1, Fx1, Fy1, Fz1, Mxx1, Myy1, Mzz1, nPL, wind_Uref2, \
#                     plidx2, Fx2, Fy2, Fz2, Mxx2, Myy2, Mzz2, gamma_f, gamma_m, gamma_n, \
#                     gamma_b, z_DEL, M_DEL, nDEL, gamma_fatigue, life, m_SN, min_d_to_t, min_taper \
#                     = setupTower(n, rotor_diameter)
#
#         nPoints = len(d_param)
#         nFull = n
#         wind = 'PowerWind'
#
#         shearExp = 0.1
#
#         nRows = 5
#         nTurbs = nRows**2
#         spacing = 5   # turbine grid spacing in diameters
#         points = np.linspace(start=spacing*rotor_diameter, stop=nRows*spacing*rotor_diameter, num=nRows)
#         xpoints, ypoints = np.meshgrid(points, points)
#         turbineX = np.ndarray.flatten(xpoints)
#         turbineY = np.ndarray.flatten(ypoints)
#         # turbineX = np.array([0,1000,0,1000,500])*0.6
#         # turbineY = np.array([0,0,1000,1000,500])*0.6
#         # nTurbs = 5
#
#         # generate boundary constraint
#         locations = np.zeros((len(turbineX),2))
#         for i in range(len(turbineX)):
#             locations[i][0] = turbineX[i]
#             locations[i][1] = turbineY[i]
#         print locations
#         boundaryVertices, boundaryNormals = calculate_boundary(locations)
#         nVertices = boundaryVertices.shape[0]
#
#         """set up 3D aspects of wind farm"""
#         diff = 0.
#         turbineH1 = 113.
#         turbineH2 = 90.
#         H1_H2 = np.array([])
#         for i in range(nTurbs/2):
#             H1_H2 = np.append(H1_H2, 0)
#             H1_H2 = np.append(H1_H2, 1)
#         if len(H1_H2) < nTurbs:
#             H1_H2 = np.append(H1_H2, 0)
#
#         """set up the problem"""
#         prob = Problem()
#         root = prob.root = Group()
#
#         root.fd_options['step_size'] = 1.E-6
#         root.fd_options['step_type'] = 'relative'
#         # root.fd_options['force_fd'] = True
#
#         root.add('turbineH1', IndepVarComp('turbineH1', turbineH1), promotes=['*'])
#         root.add('turbineH2', IndepVarComp('turbineH2', turbineH2), promotes=['*'])
#         root.add('d_paramH1', IndepVarComp('d_paramH1', d_param), promotes=['*'])
#         root.add('t_paramH1', IndepVarComp('t_paramH1', t_param), promotes=['*'])
#         root.add('d_paramH2', IndepVarComp('d_paramH2', d_param), promotes=['*'])
#         root.add('t_paramH2', IndepVarComp('t_paramH2', t_param), promotes=['*'])
#         #This component takes turbineH1, turbineH2, and H1_H2 and gives back an array
#         #of turbineZ
#         root.add('getTurbineZ', getTurbineZ(nTurbs), promotes=['*'])
#         #These components adjust the parameterized z locations for TowerSE calculations
#         #with respect to turbineZ
#         root.add('get_z_paramH1', get_z(nPoints))
#         root.add('get_z_paramH2', get_z(nPoints))
#         root.add('get_z_fullH1', get_z(n))
#         root.add('get_z_fullH2', get_z(n))
#         root.add('AEPGroup', AEPGroup(nTurbs, nDirections=nDirections,
#                     use_rotor_components=use_rotor_components, datasize=datasize, differentiable=True,
#                     optimizingLayout=False, nSamples=0), promotes=['*'])
#         root.add('COEGroup', COEGroup(nTurbs), promotes=['*'])
#         root.add('AEPobj', AEPobj(), promotes=['*'])
#
#         #For Constraints
#         root.add('TowerH1', TowerSE(nPoints, nFull, nK, nMass, nPL, nDEL, wind=wind), promotes=['L_reinforced',
#                             'E','G','sigma_y','kidx','kx','ky','kz','ktx','kty',
#                             'ktz','midx','m','mIxx','mIyy','mIzz','mIxy','mIxz','mIyz',
#                             'mrhox','mrhoy','mrhoz','addGravityLoadForExtraMass','g',
#                             'gamma_f','gamma_m','gamma_n','gamma_b','life','m_SN',
#                             'gc.min_d_to_t','gc.min_taper','M_DEL','gamma_fatigue'])
#         root.add('TowerH2', TowerSE(nPoints, nFull, nK, nMass, nPL, nDEL, wind=wind), promotes=['L_reinforced',
#                             'E','G','sigma_y','kidx','kx','ky','kz','ktx','kty',
#                             'ktz','midx','m','mIxx','mIyy','mIzz','mIxy','mIxz','mIyz',
#                             'mrhox','mrhoy','mrhoz','addGravityLoadForExtraMass','g',
#                             'gamma_f','gamma_m','gamma_n','gamma_b','life','m_SN',
#                             'gc.min_d_to_t','gc.min_taper','M_DEL','gamma_fatigue'])
#         root.add('spacing_comp', SpacingComp(nTurbines=nTurbs), promotes=['*'])
#
#         # add constraint definitions
#         root.add('spacing_con', ExecComp('sc = wtSeparationSquared-(minSpacing*rotorDiameter[0])**2',
#                                      minSpacing=minSpacing, rotorDiameter=np.zeros(nTurbs),
#                                      sc=np.zeros(((nTurbs-1.)*nTurbs/2.)),
#                                      wtSeparationSquared=np.zeros(((nTurbs-1.)*nTurbs/2.))),
#                                      promotes=['*'])
#
#         if nVertices > 0:
#             # add component that enforces a convex hull wind farm boundary
#             root.add('boundary_con', BoundaryComp(nVertices=nVertices, nTurbines=nTurbs), promotes=['*'])
#
#         root.connect('turbineH1', 'get_z_paramH1.turbineZ')
#         root.connect('turbineH2', 'get_z_paramH2.turbineZ')
#         root.connect('turbineH1', 'get_z_fullH1.turbineZ')
#         root.connect('turbineH2', 'get_z_fullH2.turbineZ')
#         root.connect('get_z_paramH1.z_param', 'TowerH1.z_param')
#         root.connect('get_z_fullH1.z_param', 'TowerH1.z_full')
#         root.connect('get_z_paramH2.z_param', 'TowerH2.z_param')
#         root.connect('get_z_fullH2.z_param', 'TowerH2.z_full')
#         root.connect('TowerH1.tower1.mass', 'mass1')
#         root.connect('TowerH2.tower1.mass', 'mass2')
#         root.connect('d_paramH1', 'TowerH1.d_param')
#         root.connect('d_paramH2', 'TowerH2.d_param')
#         root.connect('t_paramH1', 'TowerH1.t_param')
#         root.connect('t_paramH2', 'TowerH2.t_param')
#
#         prob.driver = pyOptSparseDriver()
#         prob.driver.options['optimizer'] = 'SNOPT'
#         prob.driver.opt_settings['Major iterations limit'] = 1000
#         prob.driver.opt_settings['Major optimality tolerance'] = 1E-6
#         # prob.driver = ScipyOptimizer()
#         # prob.driver.options['optimizer'] = 'SLSQP'
#
#         # --- Objective ---
#         prob.driver.add_objective('COE', scaler=1.0E-1)
#
#         # # --- Design Variables ---
#         prob.driver.add_desvar('turbineH1', lower=rotor_diameter/2.+10, upper=None, scaler=1.0)
#         prob.driver.add_desvar('turbineH2', lower=rotor_diameter/2.+10, upper=None, scaler=1.0)
#         prob.driver.add_desvar('turbineX', lower=None, upper=None, scaler=1.0)
#         prob.driver.add_desvar('turbineY', lower=None, upper=None, scaler=1.0)
#         prob.driver.add_desvar('d_paramH1', lower=np.array([1.0, 1.0, d_param[nPoints-1]]), upper=np.ones(nPoints)*4.3, scaler=1.0)
#         prob.driver.add_desvar('t_paramH1', lower=np.ones(nPoints)*.001, upper=np.ones(nPoints)*0.05, scaler=1.0)
#         prob.driver.add_desvar('d_paramH2', lower=np.array([1.0, 1.0, d_param[nPoints-1]]), upper=np.ones(nPoints)*4.3, scaler=1.0)
#         prob.driver.add_desvar('t_paramH2', lower=np.ones(nPoints)*.001, upper=np.ones(nPoints)*0.05, scaler=1.0)
#
#         # --- Constraints ---
#         # TowerH1 structure
#         prob.driver.add_constraint('TowerH1.tower1.stress', upper=1.0)
#         prob.driver.add_constraint('TowerH1.tower2.stress', upper=1.0)
#         prob.driver.add_constraint('TowerH1.tower1.global_buckling', upper=1.0)
#         prob.driver.add_constraint('TowerH1.tower2.global_buckling', upper=1.0)
#         prob.driver.add_constraint('TowerH1.tower1.shell_buckling', upper=1.0)
#         prob.driver.add_constraint('TowerH1.tower2.shell_buckling', upper=1.0)
#         prob.driver.add_constraint('TowerH1.tower1.damage', upper=1.0)
#         prob.driver.add_constraint('TowerH1.gc.weldability', upper=0.0)
#         prob.driver.add_constraint('TowerH1.gc.manufacturability', upper=0.0)
#         freq1p = 0.2  # 1P freq in Hz
#         prob.driver.add_constraint('TowerH1.tower1.f1', lower=1.1*freq1p)
#
#         # #TowerH2 structure
#         prob.driver.add_constraint('TowerH2.tower1.stress', upper=1.0)
#         prob.driver.add_constraint('TowerH2.tower2.stress', upper=1.0)
#         prob.driver.add_constraint('TowerH2.tower1.global_buckling', upper=1.0)
#         prob.driver.add_constraint('TowerH2.tower2.global_buckling', upper=1.0)
#         prob.driver.add_constraint('TowerH2.tower1.shell_buckling', upper=1.0)
#         prob.driver.add_constraint('TowerH2.tower2.shell_buckling', upper=1.0)
#         prob.driver.add_constraint('TowerH2.gc.weldability', upper=0.0)
#         prob.driver.add_constraint('TowerH2.gc.manufacturability', upper=0.0)
#         freq1p = 0.2  # 1P freq in Hz
#         prob.driver.add_constraint('TowerH2.tower1.f1', lower=1.1*freq1p)
#
#         # boundary constraint (convex hull)
#         prob.driver.add_constraint('boundaryDistances', lower=np.zeros(nVertices*nTurbs), scaler=1.0)
#         # spacing constraint
#         prob.driver.add_constraint('sc', lower=np.zeros(((nTurbs-1.)*nTurbs/2.)), scaler=1.0/rotor_diameter)
#
#         # ----------------------
#
#         prob.root.ln_solver.options['single_voi_relevance_reduction'] = True
#
#         start = time.time()
#         prob.setup()
#
#         """run the problem"""
#
#         if wind == "PowerWind":
#             prob['TowerH1.wind1.shearExp'] = shearExp
#             prob['TowerH1.wind2.shearExp'] = shearExp
#             prob['TowerH2.wind1.shearExp'] = shearExp
#             prob['TowerH2.wind2.shearExp'] = shearExp
#             prob['shearExp'] = shearExp
#         prob['turbineH1'] = turbineH1
#         prob['turbineH2'] = turbineH2
#         prob['H1_H2'] = H1_H2
#         prob['diameter'] = rotor_diameter
#
#         prob['turbineX'] = turbineX
#         prob['turbineY'] = turbineY
#         prob['yaw0'] = yaw
#         prob['ratedPower'] = np.ones_like(turbineX)*5000 # in kw
#
#         prob['boundaryVertices'] = boundaryVertices
#         prob['boundaryNormals'] = boundaryNormals
#
#         # assign values to constant inputs (not design variables)
#         prob['nIntegrationPoints'] = nIntegrationPoints
#         prob['rotorDiameter'] = rotorDiameter
#         prob['axialInduction'] = axialInduction
#         prob['generatorEfficiency'] = generatorEfficiency
#         prob['air_density'] = air_density
#         prob['windDirections'] = np.array([windDirections])
#         prob['windFrequencies'] = np.array([windFrequencies])
#         prob['Uref'] = windSpeeds
#         if use_rotor_components == True:
#             prob['gen_params:windSpeedToCPCT_CP'] = NREL5MWCPCT['CP']
#             prob['gen_params:windSpeedToCPCT_CT'] = NREL5MWCPCT['CT']
#             prob['gen_params:windSpeedToCPCT_wind_speed'] = NREL5MWCPCT['wind_speed']
#         else:
#             prob['Ct_in'] = Ct
#             prob['Cp_in'] = Cp
#         prob['floris_params:cos_spread'] = 1E12
#         prob['zref'] = wind_zref
#         prob['z0'] = wind_z0       # turns off cosine spread (just needs to be very large)
#
#         """tower structural properties"""
#         # --- geometry ----
#         # prob['d_param'] = d_param
#         # prob['t_param'] = t_param
#
#         prob['L_reinforced'] = L_reinforced
#         prob['TowerH1.yaw'] = Toweryaw
#         prob['TowerH2.yaw'] = Toweryaw
#
#         # --- material props ---
#         prob['E'] = E
#         prob['G'] = G
#         prob['TowerH1.tower1.rho'] = rho
#         prob['TowerH2.tower1.rho'] = rho
#         prob['sigma_y'] = sigma_y
#
#         # --- spring reaction data.  Use float('inf') for rigid constraints. ---
#         prob['kidx'] = kidx
#         prob['kx'] = kx
#         prob['ky'] = ky
#         prob['kz'] = kz
#         prob['ktx'] = ktx
#         prob['kty'] = kty
#         prob['ktz'] = ktz
#
#         # --- extra mass ----
#         prob['midx'] = midx
#         prob['m'] = m
#         prob['mIxx'] = mIxx
#         prob['mIyy'] = mIyy
#         prob['mIzz'] = mIzz
#         prob['mIxy'] = mIxy
#         prob['mIxz'] = mIxz
#         prob['mIyz'] = mIyz
#         prob['mrhox'] = mrhox
#         prob['mrhoy'] = mrhoy
#         prob['mrhoz'] = mrhoz
#         prob['addGravityLoadForExtraMass'] = addGravityLoadForExtraMass
#         # -----------
#
#         # --- wind ---
#         prob['TowerH1.zref'] = wind_zref
#         prob['TowerH2.zref'] = wind_zref
#         prob['TowerH1.z0'] = wind_z0
#         prob['TowerH2.z0'] = wind_z0
#         # ---------------
#
#         # # --- loading case 1: max Thrust ---
#         prob['TowerH1.wind1.Uref'] = wind_Uref1
#         prob['TowerH1.tower1.plidx'] = plidx1
#         prob['TowerH1.tower1.Fx'] = Fx1
#         prob['TowerH1.tower1.Fy'] = Fy1
#         prob['TowerH1.tower1.Fz'] = Fz1
#         prob['TowerH1.tower1.Mxx'] = Mxx1
#         prob['TowerH1.tower1.Myy'] = Myy1
#         prob['TowerH1.tower1.Mzz'] = Mzz1
#
#         prob['TowerH2.wind1.Uref'] = wind_Uref1
#         prob['TowerH2.tower1.plidx'] = plidx1
#         prob['TowerH2.tower1.Fx'] = Fx1
#         prob['TowerH2.tower1.Fy'] = Fy1
#         prob['TowerH2.tower1.Fz'] = Fz1
#         prob['TowerH2.tower1.Mxx'] = Mxx1
#         prob['TowerH2.tower1.Myy'] = Myy1
#         prob['TowerH2.tower1.Mzz'] = Mzz1
#         # # ---------------
#
#         # # --- loading case 2: max Wind Speed ---
#         prob['TowerH1.wind2.Uref'] = wind_Uref2
#         prob['TowerH1.tower2.plidx'] = plidx2
#         prob['TowerH1.tower2.Fx'] = Fx2
#         prob['TowerH1.tower2.Fy'] = Fy2
#         prob['TowerH1.tower2.Fz'] = Fz2
#         prob['TowerH1.tower2.Mxx'] = Mxx2
#         prob['TowerH1.tower2.Myy'] = Myy2
#         prob['TowerH1.tower2.Mzz'] = Mzz2
#
#         prob['TowerH2.wind2.Uref'] = wind_Uref2
#         prob['TowerH2.tower2.plidx'] = plidx2
#         prob['TowerH2.tower2.Fx'] = Fx2
#         prob['TowerH2.tower2.Fy'] = Fy2
#         prob['TowerH2.tower2.Fz'] = Fz2
#         prob['TowerH2.tower2.Mxx'] = Mxx2
#         prob['TowerH2.tower2.Myy'] = Myy2
#         prob['TowerH2.tower2.Mzz'] = Mzz2
#         # # ---------------
#
#         # --- safety factors ---
#         prob['gamma_f'] = gamma_f
#         prob['gamma_m'] = gamma_m
#         prob['gamma_n'] = gamma_n
#         prob['gamma_b'] = gamma_b
#         # ---------------
#
#         # --- fatigue ---
#         prob['gamma_fatigue'] = gamma_fatigue
#         prob['life'] = life
#         prob['m_SN'] = m_SN
#         # ---------------
#
#         # --- constraints ---
#         prob['gc.min_d_to_t'] = min_d_to_t
#         prob['gc.min_taper'] = min_taper
#         # ---------------
#
#         prob.run_once()
#
#         print 'Mass H1: ', prob['TowerH1.tower1.mass']
#         print 'Mass H2: ', prob['TowerH2.tower1.mass']
#         print 'd: ', prob['d_paramH1']
#         print 't: ', prob['t_paramH1']
#         self.J = prob.check_total_derivatives(out_stream=None)
#         print '****************************************************************'
#         print '^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^'
#         # print 'H1 fwd: ', self.J[('COE', 'turbineH1')]['J_fwd']
#         print 'H1 fd: ', self.J[('COE', 'turbineH1')]['J_fd']
#         # print 'H2 fwd: ', self.J[('COE', 'turbineH2')]['J_fwd']
#         print 'H2 fd: ', self.J[('COE', 'turbineH2')]['J_fd']
#         # print 'X fwd: ', self.J[('COE', 'turbineX')]['J_fwd']
#         print 'X fd: ', self.J[('COE', 'turbineX')]['J_fd']
#         # print 'Y fwd: ', self.J[('COE', 'turbineY')]['J_fwd']
#         print 'Y fd: ', self.J[('COE', 'turbineY')]['J_fd']
#         # print 'd1 fwd: ', self.J[('COE', 'd_paramH1')]['J_fwd']
#         print 'd1 fd: ', self.J[('COE', 'd_paramH1')]['J_fd']
#         # print 'd2 fwd: ', self.J[('COE', 'd_paramH2')]['J_fwd']
#         print 'd2 fd: ', self.J[('COE', 'd_paramH2')]['J_fd']
#         # print 't1 fwd: ', self.J[('COE', 't_paramH1')]['J_fwd']
#         print 't1 fd: ', self.J[('COE', 't_paramH1')]['J_fd']
#         # print 't2 fwd: ', self.J[('COE', 't_paramH2')]['J_fwd']
#         print 't2 fd: ', self.J[('COE', 't_paramH2')]['J_fd']
#
#     def test_COE_H1(self):
#         np.testing.assert_allclose(self.J[('TowerH1.gc.manufacturability', 'turbineH1')]['J_fwd'], self.J[('COE', 'turbineH1')]['J_fd'], 1e-6, 1e-6)
#
class TestAEP(unittest.TestCase):


    def setUp(self):

        use_rotor_components = True

        if use_rotor_components:
            NREL5MWCPCT = pickle.load(open('../../../doc/tune/NREL5MWCPCT_smooth_dict.p'))
            # print(NREL5MWCPCT)
            # NREL5MWCPCT = pickle.Unpickler(open('./input_files/NREL5MWCPCT.p')).load()
            datasize = NREL5MWCPCT['CP'].size
        else:
            datasize = 0

        rotor_diameter = 126.4

        # nRows = 1
        # nTurbs = nRows**2
        # spacing = 3   # turbine grid spacing in diameters
        # points = np.linspace(start=spacing*rotor_diameter, stop=nRows*spacing*rotor_diameter, num=nRows)
        # xpoints, ypoints = np.meshgrid(points, points)
        # turbineX = np.ndarray.flatten(xpoints)
        # turbineY = np.ndarray.flatten(ypoints)

        nTurbs = 4
        num = 100.
        turbineX = np.array([0.,1.*num,2.*num,3.*num])
        turbineY = np.array([0.,200.,400.,600.])
        # nTurbs = 2
        # num = 0.
        # turbineX = np.array([0.,1.*num])
        # turbineY = np.array([0.,200.])


        turbineH1 = 125.5
        turbineH2 = 135.

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

        """Define wind flow"""
        air_density = 1.1716    # kg/m^3

        windSpeeds, windFrequencies, windDirections, nDirections = amaliaWind()
        nDirections = 1
        windSpeeds = np.ones(nDirections)*10.
        windFrequencies = np.ones(nDirections)/nDirections
        windDirections = np.linspace(0.,360.-360./nDirections,nDirections)
        # windSpeeds = np.array([5.,10.])
        # windDirections = np.array([0.,90.])

        shearExp = 0.2

        """set up 3D aspects of wind farm"""
        H1_H2 = np.array([])
        for i in range(nTurbs/2):
            H1_H2 = np.append(H1_H2, 0)
            H1_H2 = np.append(H1_H2, 1)
        if len(H1_H2) < nTurbs:
            H1_H2 = np.append(H1_H2, 0)

        """set up the problem"""
        prob = Problem()
        root = prob.root = Group()

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'SNOPT'

        # root.deriv_options['form'] = 'central'
        # root.deriv_options['step_size'] = 500.
        # root.deriv_options['step_calc'] = 'relative'

        root.add('turbineH1', IndepVarComp('turbineH1', turbineH1), promotes=['*'])
        root.add('turbineH2', IndepVarComp('turbineH2', turbineH2), promotes=['*'])
        root.add('Uref', IndepVarComp('Uref', windSpeeds), promotes=['*'])
        root.add('getTurbineZ', getTurbineZ(nTurbs), promotes=['*'])
        root.add('AEPGroup', AEPGroup(nTurbs, nDirections=nDirections,
                    use_rotor_components=use_rotor_components, datasize=datasize, differentiable=True,
                    optimizingLayout=False, nSamples=0), promotes=['*'])

        prob.driver.add_objective('AEP', scaler=1.0E-8)
        # prob.driver.add_objective('wtVelocity0', scaler=1.0)

        prob.driver.add_desvar('turbineH1', lower=75., upper=None)
        prob.driver.add_desvar('turbineH2', lower=75., upper=None)
        prob.driver.add_desvar('turbineX', lower=75., upper=None)
        prob.driver.add_desvar('turbineY', lower=75., upper=None)
        prob.driver.add_desvar('Uref', lower=None, upper=None)

        prob.setup()

        prob['turbineH1'] = turbineH1
        prob['turbineH2'] = turbineH2
        prob['H1_H2'] = H1_H2

        prob['turbineX'] = turbineX
        prob['turbineY'] = turbineY
        prob['yaw0'] = yaw
        prob['ratedPower'] = np.ones_like(turbineX)*5000 # in kw

        # assign values to constant inputs (not design variables)
        prob['rotorDiameter'] = rotorDiameter
        prob['axialInduction'] = axialInduction
        prob['generatorEfficiency'] = generatorEfficiency
        prob['air_density'] = air_density
        prob['windDirections'] = np.array([windDirections])
        prob['windFrequencies'] = np.array([windFrequencies])
        if use_rotor_components == True:
            prob['gen_params:windSpeedToCPCT_CP'] = NREL5MWCPCT['CP']
            prob['gen_params:windSpeedToCPCT_CT'] = NREL5MWCPCT['CT']
            prob['gen_params:windSpeedToCPCT_wind_speed'] = NREL5MWCPCT['wind_speed']
        else:
            prob['Ct_in'] = Ct
            prob['Cp_in'] = Cp
        prob['floris_params:cos_spread'] = 1E12
        prob['shearExp'] = shearExp
        # prob['Uref'] = windSpeeds
        prob['zref'] = 90.
        prob['z0'] = 0.

        prob.run_once()
        # self.J = prob.check_partial_derivatives(out_stream=sys.stdout)
        self.J = prob.check_total_derivatives(out_stream=None)
        print self.J
        # print self.J
        print 'H1'
        print 'Finite Difference'
        print self.J[('AEP', 'turbineH1')]['J_fd']
        print 'Analytic'
        print self.J[('AEP', 'turbineH1')]['J_fwd']

        print 'H2'
        print 'Finite Difference'
        print self.J[('AEP', 'turbineH2')]['J_fd']
        print 'Analytic'
        print self.J[('AEP', 'turbineH2')]['J_fwd']

        print 'X'
        print 'Finite Difference'
        print self.J[('AEP', 'turbineX')]['J_fd']
        print 'Analytic'
        print self.J[('AEP', 'turbineX')]['J_fwd']

        print 'Y'
        print 'Finite Difference'
        print self.J[('AEP', 'turbineY')]['J_fd']
        print 'Rev'
        print self.J[('AEP', 'turbineY')]['J_rev']
        print 'Analytic'
        print self.J[('AEP', 'turbineY')]['J_fwd']

        print 'Speeds?: ', prob['wtVelocity0']

        # print 'Analytic'
        # print self.J[('windSpeeds', 'turbineH1')]['J_fwd']
        #
        #
        # print 'Analytic'
        # print self.J[('windSpeeds', 'turbineH2')]['J_fwd']
        # print 'Finite Difference'
        # print self.J[('windSpeeds', 'turbineH2')]['J_fd']
        #
        # print 'Analytic'
        # print self.J[('turbineZ', 'turbineH2')]['J_fwd']
        # print 'Finite Difference'
        # print self.J[('turbineZ', 'turbineH2')]['J_fd']

    def testWRT_x_y(self):
    #     np.testing.assert_allclose(self.J[('AEP', 'turbineX')]['J_fwd'], self.J[('AEP', 'turbineX')]['J_fd'], 1e-3, 1e-3)
    #     np.testing.assert_allclose(self.J[('AEP', 'turbineY')]['J_fwd'], self.J[('AEP', 'turbineY')]['J_fd'], 1e-3, 1e-3)
        np.testing.assert_allclose(self.J[('AEP', 'turbineX')]['J_fwd'], self.J[('AEP', 'turbineX')]['J_fd'], 1e-4, 1e-4)
        np.testing.assert_allclose(self.J[('AEP', 'turbineY')]['J_fwd'], self.J[('AEP', 'turbineY')]['J_fd'], 1e-4, 1e-4)

    def testWRT_H1_H2(self):
    #     np.testing.assert_allclose(self.J[('AEP', 'turbineX')]['J_fwd'], self.J[('AEP', 'turbineX')]['J_fd'], 1e-3, 1e-3)
    #     np.testing.assert_allclose(self.J[('AEP', 'turbineY')]['J_fwd'], self.J[('AEP', 'turbineY')]['J_fd'], 1e-3, 1e-3)
        np.testing.assert_allclose(self.J[('AEP', 'turbineH1')]['J_fwd'], self.J[('AEP', 'turbineH1')]['J_fd'], 1e-4, 1e-4)
        np.testing.assert_allclose(self.J[('AEP', 'turbineH2')]['J_fwd'], self.J[('AEP', 'turbineH2')]['J_fd'], 1e-4, 1e-4)

    # def test_Uref(self):
    #     np.testing.assert_allclose(self.J[('AEP', 'Uref')]['J_fwd'], self.J[('AEP', 'Uref')]['J_fd'], 1e-3, 1e-3)

# # class TestOrganizeWindSpeeds(unittest.TestCase):
# #
# #
# #     def setUp(self):
# #         nTurbines = 9
# #         speeds = np.array([10.,10.,5.,3.])
# #         nDirections = len(speeds)
# #
# #         windSpeeds = np.zeros((nTurbines,nDirections))
# #         for i in range(nTurbines):
# #             windSpeeds[i] = speeds
# #
# #
# #         prob = Problem()
# #         root = prob.root = Group()
# #
# #         root.add('windSpeeds', IndepVarComp('windSpeeds', windSpeeds), promotes=['*'])
# #         root.add('organizeWindSpeeds', organizeWindSpeeds(nTurbines, nDirections), promotes=['*'])
# #
# #         prob.driver = pyOptSparseDriver()
# #         prob.driver.options['optimizer'] = 'SNOPT'
# #         prob.driver.opt_settings['Major iterations limit'] = 1000
# #         prob.driver.opt_settings['Major optimality tolerance'] = 1E-6
# #
# #         prob.driver.add_objective('output0')
# #         prob.driver.add_desvar('windSpeeds', lower=None, upper=None)
# #
# #         prob.setup()
# #
# #         prob.run()
# #
# #         self.J = prob.check_total_derivatives(out_stream=None)
# #         print 'Analytic'
# #         print self.J[('output0', 'windSpeeds')]['J_fwd']
# #         print 'Finite Difference'
# #         print self.J[('output0', 'windSpeeds')]['J_fd']
# #
# #     def test_mass(self):
# #         np.testing.assert_allclose(self.J[('output0', 'windSpeeds')]['J_fwd'], self.J[('output0', 'windSpeeds')]['J_fd'], 1e-3, 1e-3)
# #         np.testing.assert_allclose(self.J[('output0', 'windSpeeds')]['J_fwd'], self.J[('output0', 'windSpeeds')]['J_fd'], 1e-3, 1e-3)
# #
#
class Hoop(unittest.TestCase):

    def setUp(self):
        d_full = np.array([4.])
        t_full = np.array([0.05])
        L_reinforced = np.array([10.])
        rhoAir = 1.225
        Vel = 75.

        prob = Problem()
        root = prob.root = Group()

        root.add('d_full', IndepVarComp('d_full', d_full), promotes=['*'])
        root.add('t_full', IndepVarComp('t_full', t_full), promotes=['*'])
        root.add('hoopStressEurocode', hoopStressEurocode(1), promotes=['*'])

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'SNOPT'
        prob.driver.opt_settings['Major iterations limit'] = 1000
        prob.driver.opt_settings['Major optimality tolerance'] = 1E-6

        prob.driver.add_objective('hoop_stress')
        prob.driver.add_desvar('d_full', lower=None, upper=None)
        prob.driver.add_desvar('t_full', lower=None, upper=None)

        prob.setup()

        prob['L_reinforced'] = L_reinforced
        prob['rhoAir'] = rhoAir
        prob['Vel'] = Vel


        prob.run_once()


        self.J = prob.check_total_derivatives(out_stream=None)
        print self.J
        print 'Hoop Stress: ', prob['hoop_stress']

        print 'D Full'
        print 'Analytic'
        print self.J[('hoop_stress', 'd_full')]['J_fwd']
        print 'Finite Difference'
        print self.J[('hoop_stress', 'd_full')]['J_fd']

        print 'T Full'
        print 'Analytic'
        print self.J[('hoop_stress', 't_full')]['J_fwd']
        print 'Finite Difference'
        print self.J[('hoop_stress', 't_full')]['J_fd']

    def test_mass(self):
        np.testing.assert_allclose(self.J[('hoop_stress', 'd_full')]['J_fwd'], self.J[('hoop_stress', 'd_full')]['J_fd'], 1e-3, 1e-3)
        np.testing.assert_allclose(self.J[('hoop_stress', 't_full')]['J_fwd'], self.J[('hoop_stress', 't_full')]['J_fd'], 1e-3, 1e-3)



if __name__ == '__main__':
    unittest.main()
