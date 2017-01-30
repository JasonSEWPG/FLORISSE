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
# from towerse.tower import TowerSE
from FLORISSE3D.GeneralWindFarmComponents import AEPobj, get_z, getTurbineZ, get_z_DEL, hGroups
from FLORISSE3D.COE import COEGroup
from FLORISSE3D.GeneralWindFarmComponents import calculate_boundary, SpacingComp,\
            BoundaryComp, get_z, organizeWindSpeeds, getTurbineZ, AEPobj, speedFreq, actualSpeeds, DeMUX
from FLORISSE3D.simpleTower import calcMass
import matplotlib.pyplot as plt
from FLORISSE3D.floris import AEPGroup
from openmdao.api import Problem, Group, IndepVarComp, pyOptSparseDriver, ExecComp, ScipyOptimizer
import time
# from setupOptimization import *
import cPickle as pickle
from FLORISSE3D.setupOptimization import *
from FLORISSE3D.simpleTower import hoopStressEurocode, Tower, TowerDiscretization, dynamicQgroup


"""PASS!!!!!!!!"""
"""1"""
# class TestOrganizeWindSpeeds(unittest.TestCase):
#
#
#     def setUp(self):
#         nTurbines = 9
#         speds = np.array([10.,10.,5.,3.])
#         nDirections = len(speeds)
#
#         windSpeeds = np.zeros((nTurbines,nDirections))
#         for i in range(nTurbines):
#             windSpeeds[i] = speeds
#
#
#         prob = Problem()
#         root = prob.root = Group()
#
#         root.add('windSpeeds', IndepVarComp('windSpeeds', windSpeeds), promotes=['*'])
#         root.add('organizeWindSpeeds', organizeWindSpeeds(nTurbines, nDirections), promotes=['*'])
#
#         prob.driver = pyOptSparseDriver()
#         prob.driver.options['optimizer'] = 'SNOPT'
#         prob.driver.opt_settings['Major iterations limit'] = 1000
#         prob.driver.opt_settings['Major optimality tolerance'] = 1E-6
#
#         prob.driver.add_objective('output0')
#         prob.driver.add_desvar('windSpeeds', lower=None, upper=None)
#
#         prob.setup()
#
#         prob.run_once()
#
#         self.J = prob.check_total_derivatives(out_stream=None)
#         print 'Analytic'
#         print self.J[('output0', 'windSpeeds')]['J_fwd']
#         print 'Finite Difference'
#         print self.J[('output0', 'windSpeeds')]['J_fd']
#
#     def test_mass(self):
#         np.testing.assert_allclose(self.J[('output0', 'windSpeeds')]['J_fwd'], self.J[('output0', 'windSpeeds')]['J_fd'], 1e-3, 1e-3)
#         np.testing.assert_allclose(self.J[('output0', 'windSpeeds')]['J_fwd'], self.J[('output0', 'windSpeeds')]['J_fd'], 1e-3, 1e-3)


"""2"""
# class TestCostwrtHeight_t_d(unittest.TestCase):
#
#
#     def setUp(self):
#         # --- geometry ----
#         d_param = np.array([6.0, 4.935, 3.87]) # not going to modify this right now
#         t_param = np.array([0.027*1.3, 0.023*1.3, 0.019*1.3]) # not going to modify this right now
#
#         nTurbs = 9
#         hGroup = np.array([0,1,0,0,0,1,1,1,0])
#         nGroups = 2
#
#         n = 15
#
#         turbineH1 = 100.
#         turbineH2 = 120.
#
#         prob = Problem()
#         root = prob.root = Group()
#
#         root.add('turbineH1', IndepVarComp('turbineH1', turbineH1), promotes=['*'])
#         root.add('turbineH2', IndepVarComp('turbineH2', turbineH2), promotes=['*'])
#         root.add('d_paramH1', IndepVarComp('d_paramH1', d_param), promotes=['*'])
#         root.add('t_paramH1', IndepVarComp('t_paramH1', t_param), promotes=['*'])
#         root.add('d_paramH2', IndepVarComp('d_paramH2', d_param), promotes=['*'])
#         root.add('t_paramH2', IndepVarComp('t_paramH2', t_param), promotes=['*'])
#         root.add('discretizeH1', TowerDiscretization(len(d_param), n))
#         root.add('discretizeH2', TowerDiscretization(len(d_param), n))
#         root.add('massH1', calcMass(n))
#         root.add('massH2', calcMass(n))
#         root.add('farmCost', farmCost(nTurbs, nGroups), promotes=['*'])
#         root.add('get_z_paramH1', get_z(3))
#         root.add('get_z_fullH1', get_z(n))
#         root.add('get_z_paramH2', get_z(3))
#         root.add('get_z_fullH2', get_z(n))
#
#         root.connect('turbineH1', 'get_z_paramH1.turbineZ')
#         root.connect('turbineH2', 'get_z_paramH2.turbineZ')
#         root.connect('turbineH1', 'get_z_fullH1.turbineZ')
#         root.connect('turbineH2', 'get_z_fullH2.turbineZ')
#         root.connect('get_z_paramH1.z_param', 'discretizeH1.z_param')
#         root.connect('get_z_paramH2.z_param', 'discretizeH2.z_param')
#         root.connect('get_z_fullH1.z_param', 'discretizeH1.z_full')
#         root.connect('get_z_fullH2.z_param', 'discretizeH2.z_full')
#         root.connect('massH1.mass', 'mass0')
#         root.connect('massH2.mass', 'mass1')
#         root.connect('discretizeH1.d_full', 'massH1.d_full')
#         root.connect('discretizeH2.d_full', 'massH2.d_full')
#         root.connect('discretizeH1.t_full', 'massH1.t_full')
#         root.connect('discretizeH2.t_full', 'massH2.t_full')
#         root.connect('discretizeH1.z_full', 'massH1.z_full')
#         root.connect('discretizeH2.z_full', 'massH2.z_full')
#         root.connect('d_paramH1', 'discretizeH1.d_param')
#         root.connect('d_paramH2', 'discretizeH2.d_param')
#         root.connect('t_paramH1', 'discretizeH1.t_param')
#         root.connect('t_paramH2', 'discretizeH2.t_param')
#
#         prob.driver = pyOptSparseDriver()
#         prob.driver.options['optimizer'] = 'SNOPT'
#
#         # --- Objective ---
#         prob.driver.add_objective('cost', scaler=1E-6)
#
#         # --- Design Variables ---
#         prob.driver.add_desvar('turbineH1', lower=None, upper=None, scaler=1.0)
#         prob.driver.add_desvar('turbineH2', lower=None, upper=None, scaler=1.0)
#         prob.driver.add_desvar('d_paramH1', lower=None, upper=None, scaler=1.0)
#         prob.driver.add_desvar('t_paramH1', lower=None, upper=None, scaler=1.0)
#         prob.driver.add_desvar('d_paramH2', lower=None, upper=None, scaler=1.0)
#         prob.driver.add_desvar('t_paramH2', lower=None, upper=None, scaler=1.0)
#
#
#         prob.setup()
#
#         prob['massH1.rho'] = np.ones(n)*8500.0
#         prob['massH2.rho'] = np.ones(n)*8500.0
#         prob['hGroup'] = hGroup
#
#         prob.run_once()
#
#         print 'mass0: ', prob['mass0']
#         print 'mass1: ', prob['mass1']
#
#         self.J = prob.check_total_derivatives(out_stream=None)
#         print 'H1'
#         print 'Analytic'
#         print self.J[('cost', 'turbineH1')]['J_fwd']
#         print 'Finite Difference'
#         print self.J[('cost', 'turbineH1')]['J_fd']
#
#         print 'H2'
#         print 'Analytic'
#         print self.J[('cost', 'turbineH2')]['J_fwd']
#         print 'Finite Difference'
#         print self.J[('cost', 'turbineH2')]['J_fd']
#
#         print 'D1'
#         print 'Analytic'
#         print self.J[('cost', 'd_paramH1')]['J_fwd']
#         print 'Finite Difference'
#         print self.J[('cost', 'd_paramH1')]['J_fd']
#
#         print 'D2'
#         print 'Analytic'
#         print self.J[('cost', 'd_paramH2')]['J_fwd']
#         print 'Finite Difference'
#         print self.J[('cost', 'd_paramH2')]['J_fd']
#
#         print 'T1'
#         print 'Analytic'
#         print self.J[('cost', 't_paramH1')]['J_fwd']
#         print 'Finite Difference'
#         print self.J[('cost', 't_paramH1')]['J_fd']
#
#         print 'T2'
#         print 'Analytic'
#         print self.J[('cost', 't_paramH2')]['J_fwd']
#         print 'Finite Difference'
#         print self.J[('cost', 't_paramH2')]['J_fd']
#
#
#     def test_cost_H(self):
#         np.testing.assert_allclose(self.J[('cost', 'turbineH1')]['J_fwd'], self.J[('cost', 'turbineH1')]['J_fd'], 1e-6, 1e-6)
#         np.testing.assert_allclose(self.J[('cost', 'turbineH2')]['J_fwd'], self.J[('cost', 'turbineH2')]['J_fd'], 1e-6, 1e-6)
#
#     def test_cost_d(self):
#         np.testing.assert_allclose(self.J[('cost', 'd_paramH1')]['J_fwd'], self.J[('cost', 'd_paramH1')]['J_fd'], 1e-6, 1e-6)
#         np.testing.assert_allclose(self.J[('cost', 'd_paramH2')]['J_fwd'], self.J[('cost', 'd_paramH2')]['J_fd'], 1e-6, 1e-6)
#
#     def test_cost_t(self):
#         np.testing.assert_allclose(self.J[('cost', 't_paramH1')]['J_fwd'], self.J[('cost', 't_paramH1')]['J_fd'], 1e-6, 1e-6)
#         np.testing.assert_allclose(self.J[('cost', 't_paramH2')]['J_fwd'], self.J[('cost', 't_paramH2')]['J_fd'], 1e-6, 1e-6)


"""3"""
# class TestSpeedwrtHeight(unittest.TestCase):
#
#
#     def setUp(self):
#
#         use_rotor_components = True
#
#         if use_rotor_components:
#             NREL5MWCPCT = pickle.load(open('doc/tune/NREL5MWCPCT_smooth_dict.p'))
#             # print(NREL5MWCPCT)
#             # NREL5MWCPCT = pickle.Unpickler(open('./input_files/NREL5MWCPCT.p')).load()
#             datasize = NREL5MWCPCT['CP'].size
#         else:
#             datasize = 0
#
#         rotor_diameter = 126.4
#
#         nRows = 4
#         nTurbs = nRows**2
#         spacing = 3   # turbine grid spacing in diameters
#         points = np.linspace(start=spacing*rotor_diameter, stop=nRows*spacing*rotor_diameter, num=nRows)
#         xpoints, ypoints = np.meshgrid(points, points)
#         turbineX = np.ndarray.flatten(xpoints)
#         turbineY = np.ndarray.flatten(ypoints)
#
#         nGroups = 5
#         self.nGroups = nGroups
#
#         turbineH = np.zeros(nGroups)
#         for i in range(nGroups):
#             turbineH[i] = np.random.rand(1)*60.+73.2
#
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
#         """Define wind flow"""
#         air_density = 1.1716    # kg/m^3
#
#         windSpeeds, windFrequencies, windDirections, nDirections = amaliaWind()
#         windSpeeds = np.array([10.])
#         windFrequencies = np.array([0.25])
#         windDirections = np.array([0.])
#         nDirections = len(windSpeeds)
#
#         shearExp = float(np.random.rand(1)*0.22+0.08)
#
#         """set up the problem"""
#         prob = Problem()
#         root = prob.root = Group()
#
#         prob.driver = pyOptSparseDriver()
#         prob.driver.options['optimizer'] = 'SNOPT'
#
#         for i in range(nGroups):
#             root.add('turbineH%s'%i, IndepVarComp('turbineH%s'%i, float(turbineH[i])), promotes=['*'])
#
#         root.add('Uref', IndepVarComp('Uref', windSpeeds), promotes=['*'])
#
#         root.add('hGroups', hGroups(nTurbs), promotes=['*'])
#         root.add('AEPGroup', AEPGroup(nTurbs, nDirections=nDirections,
#                     use_rotor_components=use_rotor_components, datasize=datasize, differentiable=True,
#                     optimizingLayout=False, nSamples=0), promotes=['*'])
#
#         prob.driver.add_objective('windSpeeds')
#
#         prob.driver.add_desvar('Uref', lower=None, upper=None)
#         for i in range(nGroups):
#             prob.driver.add_desvar('turbineH%s'%i, lower=75., upper=None)
#
#         prob.setup()
#
#         prob['nGroups'] = nGroups
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
#         prob['zref'] = 90.
#         prob['z0'] = 0.
#
#         prob.run_once()
#
#         self.J = prob.check_total_derivatives(out_stream=None)
#         # print self.J
#         print 'turbineZ: ', prob['turbineZ']
#
#         for i in range(nGroups):
#             print 'Finite Difference: Group %s'%i
#             print self.J[('windSpeeds', 'turbineH%s'%i)]['J_fd']
#             print 'Analytic: Group %s'%i
#             print self.J[('windSpeeds', 'turbineH%s'%i)]['J_fwd']
#
#
#     def test_mass(self):
#         for i in range(self.nGroups):
#             np.testing.assert_allclose(self.J[('windSpeeds', 'turbineH%s'%i)]['J_fwd'], self.J[('windSpeeds', 'turbineH%s'%i)]['J_fd'], 1e-3, 1e-3)


"""4"""
# class Hoop(unittest.TestCase):
#
#     def setUp(self):
#         n = 8
#         d_full = np.random.rand(n)*4.+2.
#         # d_full = np.array([6.3,6.2,6.1,5.5,4.5,3.6])
#         t_full = np.random.rand(n)*0.04+0.012
#         # t_full = np.array([0.04,0.035,0.03,0.025,0.02,0.02])
#         L_reinforced = np.ones(n)*10.
#         rhoAir = 1.225
#         Vel = 25.
#         shearExp = 0.12
#         turbineH = 150.
#
#         prob = Problem()
#         root = prob.root = Group()
#
#         root.add('d_full', IndepVarComp('d_full', d_full), promotes=['*'])
#         root.add('t_full', IndepVarComp('t_full', t_full), promotes=['*'])
#         root.add('dynamicQgroup', dynamicQgroup(n), promotes=['*'])
#         root.add('get_z_full', get_z(n))
#         root.add('hoopStressEurocode', hoopStressEurocode(n), promotes=['*'])
#
#         root.connect('get_z_full.z_param', 'z_full')
#
#         prob.driver = pyOptSparseDriver()
#         prob.driver.options['optimizer'] = 'SNOPT'
#         prob.driver.opt_settings['Major iterations limit'] = 1000
#         prob.driver.opt_settings['Major optimality tolerance'] = 1E-6
#
#         root.deriv_options['form'] = 'central'
#         root.deriv_options['step_size'] = 1.E-2
#         root.deriv_options['step_calc'] = 'relative'
#
#         prob.driver.add_objective('hoop_stress')
#         prob.driver.add_desvar('d_full', lower=None, upper=None)
#         prob.driver.add_desvar('t_full', lower=None, upper=None)
#
#         prob.setup()
#
#         prob['L_reinforced'] = L_reinforced
#         prob['rhoAir'] = rhoAir
#         prob['Vel'] = Vel
#         prob['shearExp'] = shearExp
#         prob['get_z_full.turbineZ'] = turbineH
#
#
#         prob.run_once()
#
#
#         self.J = prob.check_total_derivatives(out_stream=None)
#         # print self.J
#         print 'Hoop Stress: ', prob['hoop_stress']
#
#         print 'D Full'
#         print 'Analytic'
#         print self.J[('hoop_stress', 'd_full')]['J_fwd']
#         print 'Finite Difference'
#         print self.J[('hoop_stress', 'd_full')]['J_fd']
#
#         print 'T Full'
#         print 'Analytic'
#         print self.J[('hoop_stress', 't_full')]['J_fwd']
#         print 'Finite Difference'
#         print self.J[('hoop_stress', 't_full')]['J_fd']
#
#     def test_mass(self):
#         np.testing.assert_allclose(self.J[('hoop_stress', 'd_full')]['J_fwd'], self.J[('hoop_stress', 'd_full')]['J_fd'], 1e-3, 1e-3)
#         np.testing.assert_allclose(self.J[('hoop_stress', 't_full')]['J_fwd'], self.J[('hoop_stress', 't_full')]['J_fd'], 1e-3, 1e-3)


"""5"""
# class TestAEP(unittest.TestCase):
#
#
#     def setUp(self):
#
#         use_rotor_components = False
#
#         if use_rotor_components:
#             NREL5MWCPCT = pickle.load(open('../doc/tune/NREL5MWCPCT_smooth_dict.p'))
#             # NREL5MWCPCT = pickle.load(open('doc/tune/NREL5MWCPCT_smooth_dict.p'))
#             # print(NREL5MWCPCT)
#             # NREL5MWCPCT = pickle.Unpickler(open('./input_files/NREL5MWCPCT.p')).load()
#             datasize = NREL5MWCPCT['CP'].size
#         else:
#             datasize = 0
#
#         rotor_diameter = 126.4
#
#         nRows = 4
#         nTurbs = nRows**2
#         spacing = 3.0  # turbine grid spacing in diameters
#         points = np.linspace(start=spacing*rotor_diameter, stop=nRows*spacing*rotor_diameter, num=nRows)
#         xpoints, ypoints = np.meshgrid(points, points)
#         turbineX_bounds = np.ndarray.flatten(xpoints)
#         turbineY_bounds = np.ndarray.flatten(ypoints)
#         xmin = min(turbineX_bounds)
#         xmax = max(turbineX_bounds)
#         ymin = min(turbineY_bounds)
#         ymax = max(turbineY_bounds)
#         print ymin, ymax, xmin, xmax
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
#             Cp[turbI] = 0.7737 * 4.0 * 1.0/3.0 * np.power((1. - 1.0/3.0), 2)
#             generatorEfficiency[turbI] = 1.0#0.944
#             yaw[turbI] = 0.     # deg.
#
#         minSpacing = 2.0
#
#         # generate boundary constraint
#         locations = np.zeros((len(turbineX_bounds),2))
#         for i in range(len(turbineX_bounds)):
#             locations[i][0] = turbineX_bounds[i]
#             locations[i][1] = turbineY_bounds[i]
#         print locations
#         boundaryVertices, boundaryNormals = calculate_boundary(locations)
#         nVertices = boundaryVertices.shape[0]
#
#         """Define wind flow"""
#         air_density = 1.1716    # kg/m^3
#
#         windData = "Amalia"
#
#         """Amalia Wind Arrays"""
#         if windData == "Amalia":
#             windSpeeds, windFrequencies, windDirections, nDirections = amaliaWind()
#             nDirections = 10
#             # windSpeeds = np.ones(nDirections)*10.
#             # windFrequencies = np.ones(nDirections)/nDirections
#             windSpeeds = np.random.rand(nDirections)*15.+2.
#             windFrequencies = np.random.rand(nDirections)
#             windFrequencies = windFrequencies/np.sum(windFrequencies)
#             # windDirections = np.linspace(0.,360.-360./nDirections,nDirections)
#             # windDirections = np.array([0.,90.])
#             windDirections = np.random.rand(nDirections)*360.
#
#         shearExp = 0.17
#
#         nGroups = 3
#
#         turbineX = xmin+np.random.rand(nTurbs)*(xmax-xmin)
#         turbineY = ymin+np.random.rand(nTurbs)*(ymax-ymin)
#         turbineZ = np.random.rand(nTurbs)*60.+73.2
#
#         """OpenMDAO"""
#
#         prob = Problem()
#         root = prob.root = Group()
#
#         for i in range(nGroups):
#             root.add('turbineH%s'%i, IndepVarComp('turbineH%s'%i, float(turbineZ[i])), promotes=['*'])
#         root.add('hGroups', hGroups(nTurbs), promotes=['*'])
#         root.add('AEPGroup', AEPGroup(nTurbs, nDirections=nDirections,
#                     use_rotor_components=use_rotor_components, datasize=datasize, differentiable=True,
#                     optimizingLayout=False, nSamples=0), promotes=['*'])
#         root.add('maxAEP', AEPobj(), promotes=['*'])
#
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
#         prob.driver = pyOptSparseDriver()
#         prob.driver.options['optimizer'] = 'SNOPT'
#
#         # --- Objective ---
#         prob.driver.add_objective('AEP', scaler=1.0E-1)
#
#         # --- Design Variables ---
#         prob.driver.add_desvar('turbineX', lower=np.ones(nTurbs)*min(turbineX), upper=np.ones(nTurbs)*max(turbineX), scaler=1.0E-3)
#         prob.driver.add_desvar('turbineY', lower=np.ones(nTurbs)*min(turbineY), upper=np.ones(nTurbs)*max(turbineY), scaler=1.0E-3)
#
#         for i in range(nGroups):
#             prob.driver.add_desvar('turbineH%s'%i, lower=73.2, upper=None, scaler=1.0E-2)
#
#         for direction_id in range(nDirections):
#             prob.driver.add_desvar('yaw%i' % direction_id, lower=-30.0, upper=30.0, scaler=1)
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
#         prob.setup(check=True)
#
#         prob['turbineX'] = turbineX
#         prob['turbineY'] = turbineY
#         prob['nGroups'] = nGroups
#         # for i in range(nDirections):
#         #     prob['yaw%s'%i] = yaw
#
#         prob['ratedPower'] = np.ones_like(turbineX)*5000. # in kw
#
#         prob['boundaryVertices'] = boundaryVertices
#         prob['boundaryNormals'] = boundaryNormals
#
#         # assign values to constant inputs (not design variables)
#         prob['rotorDiameter'] = rotorDiameter
#         prob['axialInduction'] = axialInduction
#         prob['generatorEfficiency'] = generatorEfficiency
#         prob['air_density'] = air_density
#         prob['windDirections'] = windDirections
#         prob['windFrequencies'] = windFrequencies
#         prob['Uref'] = windSpeeds
#         prob['Ct_in'] = Ct
#         prob['Cp_in'] = Cp
#         prob['floris_params:cos_spread'] = 1E12 # turns off cosine spread (just needs to be very large)
#
#         prob.run_once()
#
#         print 'AEP: ', prob['AEP']
#
#         print 'TurbineX: ', prob['turbineX']
#         print 'TurbineY: ', prob['turbineY']
#         print 'TurbineZ: ', prob['turbineZ']
#
#         J = prob.check_total_derivatives(out_stream=None)
#         self.J = J
#         self.nGroups = nGroups
#         self.nDirections = nDirections
#
#         print 'AEP: ', prob['AEP']
#         print 'hGroup: ', prob['hGroup']
#         print 'turbineZ: ', prob['turbineZ']
#
#         print 'Analytic X'
#         print self.J[('AEP', 'turbineX')]['J_rev']
#         print 'Finite Difference X'
#         print self.J[('AEP', 'turbineX')]['J_fd']
#
#         print 'Analytic Y'
#         print self.J[('AEP', 'turbineY')]['J_rev']
#         print 'Finite Difference Y'
#         print self.J[('AEP', 'turbineY')]['J_fd']
#
#
#         for i in range(nGroups):
#              print 'Analytic H%s'%i
#              print self.J[('AEP', 'turbineH%s'%i)]['J_rev']
#              print 'Finite Difference H%s'%i
#              print self.J[('AEP', 'turbineH%s'%i)]['J_fd']
#
#
#     def test_X(self):
#         np.testing.assert_allclose(self.J[('AEP', 'turbineX')]['J_rev'], self.J[('AEP', 'turbineX')]['J_fd'], 1e-2, 1e-2)
#
#     def test_Y(self):
#         np.testing.assert_allclose(self.J[('AEP', 'turbineY')]['J_rev'], self.J[('AEP', 'turbineY')]['J_fd'], 1e-4, 1e-4)
#
#     def testZ(self):
#         for i in range(self.nGroups):
#             np.testing.assert_allclose(self.J[('AEP', 'turbineH%s'%i)]['J_rev'], self.J[('AEP', 'turbineH%s'%i)]['J_fd'], 1e-6, 1e-6)
#
#     def testYaw(self):
#         for i in range(self.nDirections):
#             np.testing.assert_allclose(self.J[('AEP', 'yaw%s'%i)]['J_rev'], self.J[('AEP', 'yaw%s'%i)]['J_fd'], 1e-2, 1e-2)


"""6"""
# class TestTotalDerivatives(unittest.TestCase):
#
#     def setUp(self):
#         use_rotor_components = False
#
#
#         rotor_diameter = 126.4
#
#         nRows = 4
#         nTurbs = nRows**2
#         spacing = 2.0  # turbine grid spacing in diameters
#         points = np.linspace(start=spacing*rotor_diameter, stop=nRows*spacing*rotor_diameter, num=nRows)
#         xpoints, ypoints = np.meshgrid(points, points)
#         turbineX_bounds = np.ndarray.flatten(xpoints)
#         turbineY_bounds = np.ndarray.flatten(ypoints)
#         xmin = min(turbineX_bounds)
#         xmax = max(turbineX_bounds)
#         ymin = min(turbineY_bounds)
#         ymax = max(turbineY_bounds)
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
#             Cp[turbI] = 0.7737 * 4.0 * 1.0/3.0 * np.power((1. - 1.0/3.0), 2)
#             generatorEfficiency[turbI] = 1.0#0.944
#             yaw[turbI] = 0.     # deg.
#
#         minSpacing = 2.0
#
#         # generate boundary constraint
#         locations = np.zeros((len(turbineX_bounds),2))
#         for i in range(len(turbineX_bounds)):
#             locations[i][0] = turbineX_bounds[i]
#             locations[i][1] = turbineY_bounds[i]
#         print locations
#         boundaryVertices, boundaryNormals = calculate_boundary(locations)
#         nVertices = boundaryVertices.shape[0]
#
#         """Define wind flow"""
#         air_density = 1.1716    # kg/m^3
#
#         windData = "Amalia"
#
#         """Amalia Wind Arrays"""
#         if windData == "Amalia":
#             windSpeeds, windFrequencies, windDirections, nDirections = amaliaWind()
#             nDirections = 1
#             windSpeeds = np.ones(nDirections)*10.
#             windFrequencies = np.ones(nDirections)/nDirections
#             windDirections = np.linspace(0.,360.-360./nDirections,nDirections)
#             # windDirections = np.array([0.,90.])
#
#
#         """Define tower structural properties"""
#         # --- geometry ---
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
#         nPoints = 3
#         nFull = n
#         rhoAir = air_density
#
#         shearExp = 0.17
#
#         nGroups = 3
#
#         turbineX = xmin+np.random.rand(nTurbs)*(xmax-xmin)
#         turbineY = ymin+np.random.rand(nTurbs)*(ymax-ymin)
#         turbineZ = 73.2+np.random.rand(nTurbs)*60.
#         d_param = np.zeros((nTurbs,3))
#         t_param = np.zeros((nTurbs,3))
#         for i in range(nGroups):
#             d_param[i] = 3.6+np.random.rand(3)*(6.3-3.6)
#             t_param[i] = 0.01+np.random.rand(3)*(0.04)
#
#
#         """OpenMDAO"""
#
#         prob = Problem()
#         root = prob.root = Group()
#
#         # root.add('turbineZ', IndepVarComp('turbineZ', turbineZ), promotes=['*'])
#         for i in range(nGroups):
#             root.add('d_param%s'%i, IndepVarComp('d_param%s'%i, d_param[i]), promotes=['*'])
#             root.add('t_param%s'%i, IndepVarComp('t_param%s'%i, t_param[i]), promotes=['*'])
#             root.add('get_z_param%s'%i, get_z(nPoints))
#             root.add('get_z_full%s'%i, get_z(nFull))
#             root.add('Tower%s_max_thrust'%i, Tower(nPoints, nFull), promotes=['L_reinforced','m','mrhox','E','sigma_y','gamma_f','gamma_b','rhoAir','z0','zref','shearExp','rho'])
#             root.add('Tower%s_max_speed'%i, Tower(nPoints, nFull), promotes=['L_reinforced','m','mrhox','E','sigma_y','gamma_f','gamma_b','rhoAir','z0','zref','shearExp','rho'])
#             root.add('turbineH%s'%i, IndepVarComp('turbineH%s'%i, float(turbineZ[i])), promotes=['*'])
#         root.add('Zs', DeMUX(nTurbs))
#         root.add('hGroups', hGroups(nTurbs), promotes=['*'])
#         root.add('AEPGroup', AEPGroup(nTurbs, nDirections=nDirections,
#                     use_rotor_components=use_rotor_components, datasize=0, differentiable=True,
#                     optimizingLayout=False, nSamples=0), promotes=['*'])
#         root.add('COEGroup', COEGroup(nTurbs, nGroups), promotes=['*'])
#         root.add('maxAEP', AEPobj(), promotes=['*'])
#
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
#         root.connect('turbineZ', 'Zs.Array')
#         for i in range(nGroups):
#             root.connect('get_z_param%s.z_param'%i, 'Tower%s_max_thrust.z_param'%i)
#             root.connect('get_z_full%s.z_param'%i, 'Tower%s_max_thrust.z_full'%i)
#             root.connect('get_z_param%s.z_param'%i, 'Tower%s_max_speed.z_param'%i)
#             root.connect('get_z_full%s.z_param'%i, 'Tower%s_max_speed.z_full'%i)
#             root.connect('Zs.output%s'%i, 'get_z_param%s.turbineZ'%i)
#             root.connect('Zs.output%s'%i, 'get_z_full%s.turbineZ'%i)
#             root.connect('Zs.output%s'%i, 'Tower%s_max_thrust.L'%i)
#             root.connect('Zs.output%s'%i, 'Tower%s_max_speed.L'%i)
#
#             root.connect('get_z_param%s.z_param'%i, 'Tower%s_max_thrust.z_param'%i)
#             root.connect('get_z_full%s.z_param'%i, 'Tower%s_max_thrust.z_full'%i)
#
#             root.connect('Tower%s_max_thrust.mass'%i, 'mass%s'%i)
#
#             root.connect('d_param%s'%i, 'Tower%s_max_thrust.d_param'%i)
#             root.connect('t_param%s'%i, 'Tower%s_max_thrust.t_param'%i)
#             root.connect('d_param%s'%i, 'Tower%s_max_speed.d_param'%i)
#             root.connect('t_param%s'%i, 'Tower%s_max_speed.t_param'%i)
#
#             root.connect('Tower%s_max_speed.Mt'%i, 'Tower%s_max_speed.Mt'%i)
#             root.connect('Tower%s_max_speed.It'%i, 'Tower%s_max_speed.It'%i)
#
#         prob.driver = pyOptSparseDriver()
#         prob.driver.options['optimizer'] = 'SNOPT'
#
#         # --- Objective ---
#         prob.driver.add_objective('COE', scaler=1.0E-1)
#
#         # --- Design Variables ---
#         prob.driver.add_desvar('turbineX', lower=np.ones(nTurbs)*min(turbineX), upper=np.ones(nTurbs)*max(turbineX), scaler=1.0E-3)
#         prob.driver.add_desvar('turbineY', lower=np.ones(nTurbs)*min(turbineY), upper=np.ones(nTurbs)*max(turbineY), scaler=1.0E-3)
#         # prob.driver.add_desvar('turbineZ', lower=73.2, upper=np.ones(nTurbs)*max(turbineY), scaler=1.0E-2)
#
#         for i in range(nGroups):
#             prob.driver.add_desvar('turbineH%s'%i, lower=73.2, upper=None, scaler=1.0E-2)
#
#         for i in range(nGroups):
#             prob.driver.add_desvar('d_param%s'%i, lower=np.array([1.0, 1.0, 3.87]), upper=np.ones(nPoints)*6.3, scaler=1.0E-1)
#             prob.driver.add_desvar('t_param%s'%i, lower=np.ones(nPoints)*.001, upper=np.ones(nPoints)*0.05, scaler=1.0E1)
#
#         # for direction_id in range(nDirections):
#         #     prob.driver.add_desvar('yaw%i' % direction_id, lower=-30.0, upper=30.0, scaler=1)
#
#         # prob.driver.add_desvar('yaw1', lower=-30.0, upper=30.0, scaler=1)
#
#         # boundary constraint (convex hull)
#         prob.driver.add_constraint('boundaryDistances', lower=np.zeros(nVertices*nTurbs), scaler=1.0)
#         # spacing constraint
#         prob.driver.add_constraint('sc', lower=np.zeros(((nTurbs-1.)*nTurbs/2.)), scaler=1.0/rotor_diameter)
#
#
#         for i in range(nGroups):
#             prob.driver.add_constraint('Tower%s_max_thrust.shell_buckling'%i, upper=np.ones(nFull))
#             prob.driver.add_constraint('Tower%s_max_speed.shell_buckling'%i, upper=np.ones(nFull))
#             freq1p = 0.2  # 1P freq in Hz
#             prob.driver.add_constraint('Tower%s_max_thrust.freq'%i, lower=1.1*freq1p)
#             prob.driver.add_constraint('Tower%s_max_speed.freq'%i, lower=1.1*freq1p)
#
#         # ----------------------
#
#         prob.root.ln_solver.options['single_voi_relevance_reduction'] = True
#
#         prob.setup(check=True)
#
#         prob['nGroups'] = nGroups
#         # prob['randomize'] = True
#         prob['turbineX'] = turbineX
#         prob['turbineY'] = turbineY
#         # for i in range(nDirections):
#         #     prob['yaw%s'%i] = yaw
#
#         prob['ratedPower'] = np.ones_like(turbineX)*5000. # in kw
#
#         prob['boundaryVertices'] = boundaryVertices
#         prob['boundaryNormals'] = boundaryNormals
#
#         # assign values to constant inputs (not design variables)
#         prob['rotorDiameter'] = rotorDiameter
#         prob['axialInduction'] = axialInduction
#         prob['generatorEfficiency'] = generatorEfficiency
#         prob['air_density'] = air_density
#         prob['windDirections'] = windDirections
#         prob['windFrequencies'] = windFrequencies
#         prob['Uref'] = windSpeeds
#
#         if use_rotor_components == True:
#             prob['gen_params:windSpeedToCPCT_CP'] = NREL5MWCPCT['CP']
#             prob['gen_params:windSpeedToCPCT_CT'] = NREL5MWCPCT['CT']
#             prob['gen_params:windSpeedToCPCT_wind_speed'] = NREL5MWCPCT['wind_speed']
#         else:
#             prob['Ct_in'] = Ct
#             prob['Cp_in'] = Cp
#         prob['floris_params:cos_spread'] = 1E12 # turns off cosine spread (just needs to be very large)
#
#         prob['L_reinforced'] = L_reinforced
#         prob['rho'] = rho
#         prob['shearExp'] = shearExp
#         prob['E'] = E
#         prob['gamma_f'] = gamma_f
#         prob['gamma_b'] = gamma_b
#         prob['sigma_y'] = sigma_y
#         prob['m'] = m
#         prob['mrhox'] = mrhox
#         prob['zref'] = 90.
#         prob['z0'] = 0.
#
#         prob['turbineX'] = turbineX
#         prob['turbineY'] = turbineY
#
#         for i in range(nGroups):
#             prob['Tower%s_max_thrust.Fy'%i] = Fy1
#             prob['Tower%s_max_thrust.Fx'%i] = Fx1
#             prob['Tower%s_max_thrust.Fz'%i] = Fz1
#             prob['Tower%s_max_thrust.Mxx'%i] = Mxx1
#             prob['Tower%s_max_thrust.Myy'%i] = Myy1
#             prob['Tower%s_max_thrust.Vel'%i] = wind_Uref1
#             prob['Tower%s_max_thrust.Mt'%i] = m[0]
#             prob['Tower%s_max_thrust.It'%i] = mIzz[0]
#
#             prob['Tower%s_max_speed.Fy'%i] = Fy2
#             prob['Tower%s_max_speed.Fx'%i] = Fx2
#             prob['Tower%s_max_speed.Fz'%i] = Fz2
#             prob['Tower%s_max_speed.Mxx'%i] = Mxx2
#             prob['Tower%s_max_speed.Myy'%i] = Myy2
#             prob['Tower%s_max_speed.Vel'%i] = wind_Uref2
#
#
#         prob.run_once()
#
#         print 'AEP: ', prob['AEP']
#         print 'COE: ', prob['COE']
#         # print 'Cost: ', prob['cost']
#
#         print 'TurbineX: ', prob['turbineX']
#         print 'TurbineY: ', prob['turbineY']
#         print 'TurbineZ: ', prob['turbineZ']
#         print 'd_param: ', prob['d_param0']
#         print 't_param: ', prob['t_param0']
#
#         self.J = prob.check_total_derivatives(out_stream=None)
#         self.nGroups = nGroups
#
#         for i in range(nGroups):
#             print 'H%s fwd: '%i, self.J[('COE', 'turbineH%s'%i)]['J_fwd']
#             print 'H%s fd: '%i, self.J[('COE', 'turbineH%s'%i)]['J_fd']
#         for i in range(nGroups):
#             print 'd_param%s fwd: '%i, self.J[('COE', 'd_param%s'%i)]['J_fwd']
#             print 'd_param%s fd: '%i, self.J[('COE', 'd_param%s'%i)]['J_fd']
#         for i in range(nGroups):
#             print 't_param%s fwd: '%i, self.J[('COE', 't_param%s'%i)]['J_fwd']
#             print 't_param%s fd: '%i, self.J[('COE', 't_param%s'%i)]['J_fd']
#         print 'X fwd: ', self.J[('COE', 'turbineX')]['J_fwd']
#         print 'X fd: ', self.J[('COE', 'turbineX')]['J_fd']
#         print 'Y fwd: ', self.J[('COE', 'turbineY')]['J_fwd']
#         print 'Y fd: ', self.J[('COE', 'turbineY')]['J_fd']
#
#
#     def test_COE_H1(self):
#         for i in range(self.nGroups):
#             np.testing.assert_allclose(self.J[('COE', 'turbineH%s'%i)]['J_fwd'], self.J[('COE', 'turbineH%s'%i)]['J_fd'], 1e-6, 1e-6)
#
#     def test_COE_d_t(self):
#         for i in range(self.nGroups):
#             np.testing.assert_allclose(self.J[('COE', 'd_param%s'%i)]['J_fwd'], self.J[('COE', 'd_param%s'%i)]['J_fd'], 1e-6, 1e-6)
#             np.testing.assert_allclose(self.J[('COE', 't_param%s'%i)]['J_fwd'], self.J[('COE', 't_param%s'%i)]['J_fd'], 1e-6, 1e-6)
#
#     def test_COE_X_Y(self):
#         np.testing.assert_allclose(self.J[('COE', 'turbineX')]['J_fwd'], self.J[('COE', 'turbineX')]['J_fd'], 1e-6, 1e-6)
#         np.testing.assert_allclose(self.J[('COE', 'turbineY')]['J_fwd'], self.J[('COE', 'turbineY')]['J_fd'], 1e-6, 1e-6)






"""STILL CHECK"""


class TestTotalDerivatives_Constraints(unittest.TestCase):

    def setUp(self):
        use_rotor_components = False


        rotor_diameter = 126.4

        nRows = 2
        nTurbs = nRows**2
        spacing = 2.0  # turbine grid spacing in diameters
        points = np.linspace(start=spacing*rotor_diameter, stop=nRows*spacing*rotor_diameter, num=nRows)
        xpoints, ypoints = np.meshgrid(points, points)
        turbineX_bounds = np.ndarray.flatten(xpoints)
        turbineY_bounds = np.ndarray.flatten(ypoints)
        xmin = min(turbineX_bounds)
        xmax = max(turbineX_bounds)
        ymin = min(turbineY_bounds)
        ymax = max(turbineY_bounds)

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
        locations = np.zeros((len(turbineX_bounds),2))
        for i in range(len(turbineX_bounds)):
            locations[i][0] = turbineX_bounds[i]
            locations[i][1] = turbineY_bounds[i]
        print locations
        boundaryVertices, boundaryNormals = calculate_boundary(locations)
        nVertices = boundaryVertices.shape[0]

        """Define wind flow"""
        air_density = 1.1716    # kg/m^3

        windData = "Amalia"

        """Amalia Wind Arrays"""
        if windData == "Amalia":
            windSpeeds, windFrequencies, windDirections, nDirections = amaliaWind()
            nDirections = 1
            windSpeeds = np.ones(nDirections)*10.
            windFrequencies = np.ones(nDirections)/nDirections
            windDirections = np.linspace(0.,360.-360./nDirections,nDirections)
            # windDirections = np.array([0.,90.])


        """Define tower structural properties"""
        # --- geometry ---
        n = 15

        L_reinforced, Toweryaw, E, G, rho, sigma_y, kidx, kx, ky, kz, ktx, kty, ktz, nK, \
                    midx, m, mIxx, mIyy, mIzz, mIxy, mIxz, mIyz, mrhox, mrhoy, mrhoz, nMass, \
                    addGravityLoadForExtraMass, wind_zref, wind_z0, shearExp, wind_Uref1, \
                    plidx1, Fx1, Fy1, Fz1, Mxx1, Myy1, Mzz1, nPL, wind_Uref2, \
                    plidx2, Fx2, Fy2, Fz2, Mxx2, Myy2, Mzz2, gamma_f, gamma_m, gamma_n, \
                    gamma_b, z_DEL, M_DEL, nDEL, gamma_fatigue, life, m_SN, min_d_to_t, min_taper \
                    = setupTower(n)

        nPoints = 3
        nFull = n
        rhoAir = air_density

        shearExp = 0.17

        nGroups = 2

        turbineX = xmin+np.random.rand(nTurbs)*(xmax-xmin)
        turbineY = ymin+np.random.rand(nTurbs)*(ymax-ymin)
        turbineZ = 73.2+np.random.rand(nTurbs)*60.
        d_param = np.zeros((nTurbs,3))
        t_param = np.zeros((nTurbs,3))
        for i in range(nGroups):
            d_param[i] = 3.6+np.random.rand(3)*(6.3-3.6)
            t_param[i] = 0.01+np.random.rand(3)*(0.04)


        """OpenMDAO"""

        prob = Problem()
        root = prob.root = Group()

        # root.add('turbineZ', IndepVarComp('turbineZ', turbineZ), promotes=['*'])
        for i in range(nGroups):
            root.add('d_param%s'%i, IndepVarComp('d_param%s'%i, d_param[i]), promotes=['*'])
            root.add('t_param%s'%i, IndepVarComp('t_param%s'%i, t_param[i]), promotes=['*'])
            root.add('get_z_param%s'%i, get_z(nPoints))
            root.add('get_z_full%s'%i, get_z(nFull))
            root.add('Tower%s_max_thrust'%i, Tower(nPoints, nFull), promotes=['L_reinforced','m','mrhox','E','sigma_y','gamma_f','gamma_b','rhoAir','z0','zref','shearExp','rho'])
            root.add('Tower%s_max_speed'%i, Tower(nPoints, nFull), promotes=['L_reinforced','m','mrhox','E','sigma_y','gamma_f','gamma_b','rhoAir','z0','zref','shearExp','rho'])
            root.add('turbineH%s'%i, IndepVarComp('turbineH%s'%i, float(turbineZ[i])), promotes=['*'])
        root.add('Zs', DeMUX(nTurbs))
        root.add('hGroups', hGroups(nTurbs), promotes=['*'])
        root.add('AEPGroup', AEPGroup(nTurbs, nDirections=nDirections,
                    use_rotor_components=use_rotor_components, datasize=0, differentiable=True,
                    optimizingLayout=False, nSamples=0), promotes=['*'])
        root.add('COEGroup', COEGroup(nTurbs, nGroups), promotes=['*'])
        root.add('maxAEP', AEPobj(), promotes=['*'])

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

        root.connect('turbineZ', 'Zs.Array')
        for i in range(nGroups):
            root.connect('get_z_param%s.z_param'%i, 'Tower%s_max_thrust.z_param'%i)
            root.connect('get_z_full%s.z_param'%i, 'Tower%s_max_thrust.z_full'%i)
            root.connect('get_z_param%s.z_param'%i, 'Tower%s_max_speed.z_param'%i)
            root.connect('get_z_full%s.z_param'%i, 'Tower%s_max_speed.z_full'%i)
            root.connect('Zs.output%s'%i, 'get_z_param%s.turbineZ'%i)
            root.connect('Zs.output%s'%i, 'get_z_full%s.turbineZ'%i)
            root.connect('Zs.output%s'%i, 'Tower%s_max_thrust.L'%i)
            root.connect('Zs.output%s'%i, 'Tower%s_max_speed.L'%i)

            root.connect('get_z_param%s.z_param'%i, 'Tower%s_max_thrust.z_param'%i)
            root.connect('get_z_full%s.z_param'%i, 'Tower%s_max_thrust.z_full'%i)

            root.connect('Tower%s_max_thrust.mass'%i, 'mass%s'%i)

            root.connect('d_param%s'%i, 'Tower%s_max_thrust.d_param'%i)
            root.connect('t_param%s'%i, 'Tower%s_max_thrust.t_param'%i)
            root.connect('d_param%s'%i, 'Tower%s_max_speed.d_param'%i)
            root.connect('t_param%s'%i, 'Tower%s_max_speed.t_param'%i)

            root.connect('Tower%s_max_speed.Mt'%i, 'Tower%s_max_speed.Mt'%i)
            root.connect('Tower%s_max_speed.It'%i, 'Tower%s_max_speed.It'%i)

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'SNOPT'

        # --- Objective ---
        prob.driver.add_objective('COE', scaler=1.0E-1)

        # --- Design Variables ---
        prob.driver.add_desvar('turbineX', lower=np.ones(nTurbs)*min(turbineX), upper=np.ones(nTurbs)*max(turbineX), scaler=1.0E-3)
        prob.driver.add_desvar('turbineY', lower=np.ones(nTurbs)*min(turbineY), upper=np.ones(nTurbs)*max(turbineY), scaler=1.0E-3)
        # prob.driver.add_desvar('turbineZ', lower=73.2, upper=np.ones(nTurbs)*max(turbineY), scaler=1.0E-2)

        for i in range(nGroups):
            prob.driver.add_desvar('turbineH%s'%i, lower=73.2, upper=None, scaler=1.0E-2)

        for i in range(nGroups):
            prob.driver.add_desvar('d_param%s'%i, lower=np.array([1.0, 1.0, 3.87]), upper=np.ones(nPoints)*6.3, scaler=1.0E-1)
            prob.driver.add_desvar('t_param%s'%i, lower=np.ones(nPoints)*.001, upper=np.ones(nPoints)*0.05, scaler=1.0E1)

        # for direction_id in range(nDirections):
        #     prob.driver.add_desvar('yaw%i' % direction_id, lower=-30.0, upper=30.0, scaler=1)

        # prob.driver.add_desvar('yaw1', lower=-30.0, upper=30.0, scaler=1)

        # boundary constraint (convex hull)
        prob.driver.add_constraint('boundaryDistances', lower=np.zeros(nVertices*nTurbs), scaler=1.0)
        # spacing constraint
        prob.driver.add_constraint('sc', lower=np.zeros(((nTurbs-1.)*nTurbs/2.)), scaler=1.0/rotor_diameter)


        for i in range(nGroups):
            prob.driver.add_constraint('Tower%s_max_thrust.shell_buckling'%i, upper=np.ones(nFull))
            prob.driver.add_constraint('Tower%s_max_speed.shell_buckling'%i, upper=np.ones(nFull))
            freq1p = 0.2  # 1P freq in Hz
            prob.driver.add_constraint('Tower%s_max_thrust.freq'%i, lower=1.1*freq1p)
            prob.driver.add_constraint('Tower%s_max_speed.freq'%i, lower=1.1*freq1p)

        # ----------------------

        prob.root.ln_solver.options['single_voi_relevance_reduction'] = True

        prob.setup(check=True)

        prob['nGroups'] = nGroups
        # prob['randomize'] = True
        prob['turbineX'] = turbineX
        prob['turbineY'] = turbineY
        # for i in range(nDirections):
        #     prob['yaw%s'%i] = yaw

        prob['ratedPower'] = np.ones_like(turbineX)*5000. # in kw

        prob['boundaryVertices'] = boundaryVertices
        prob['boundaryNormals'] = boundaryNormals

        # assign values to constant inputs (not design variables)
        prob['rotorDiameter'] = rotorDiameter
        prob['axialInduction'] = axialInduction
        prob['generatorEfficiency'] = generatorEfficiency
        prob['air_density'] = air_density
        prob['windDirections'] = windDirections
        prob['windFrequencies'] = windFrequencies
        prob['Uref'] = windSpeeds

        if use_rotor_components == True:
            prob['gen_params:windSpeedToCPCT_CP'] = NREL5MWCPCT['CP']
            prob['gen_params:windSpeedToCPCT_CT'] = NREL5MWCPCT['CT']
            prob['gen_params:windSpeedToCPCT_wind_speed'] = NREL5MWCPCT['wind_speed']
        else:
            prob['Ct_in'] = Ct
            prob['Cp_in'] = Cp
        prob['floris_params:cos_spread'] = 1E12 # turns off cosine spread (just needs to be very large)

        prob['L_reinforced'] = L_reinforced
        prob['rho'] = rho
        prob['shearExp'] = shearExp
        prob['E'] = E
        prob['gamma_f'] = gamma_f
        prob['gamma_b'] = gamma_b
        prob['sigma_y'] = sigma_y
        prob['m'] = m
        prob['mrhox'] = mrhox
        prob['zref'] = 90.
        prob['z0'] = 0.

        prob['turbineX'] = turbineX
        prob['turbineY'] = turbineY

        for i in range(nGroups):
            prob['Tower%s_max_thrust.Fy'%i] = Fy1
            prob['Tower%s_max_thrust.Fx'%i] = Fx1
            prob['Tower%s_max_thrust.Fz'%i] = Fz1
            prob['Tower%s_max_thrust.Mxx'%i] = Mxx1
            prob['Tower%s_max_thrust.Myy'%i] = Myy1
            prob['Tower%s_max_thrust.Vel'%i] = wind_Uref1
            prob['Tower%s_max_thrust.Mt'%i] = m[0]
            prob['Tower%s_max_thrust.It'%i] = mIzz[0]

            prob['Tower%s_max_speed.Fy'%i] = Fy2
            prob['Tower%s_max_speed.Fx'%i] = Fx2
            prob['Tower%s_max_speed.Fz'%i] = Fz2
            prob['Tower%s_max_speed.Mxx'%i] = Mxx2
            prob['Tower%s_max_speed.Myy'%i] = Myy2
            prob['Tower%s_max_speed.Vel'%i] = wind_Uref2


        prob.run_once()

        print 'AEP: ', prob['AEP']
        print 'COE: ', prob['COE']
        print 'Z_full: ', prob['Tower0_max_thrust.z_full']

        self.J = prob.check_total_derivatives(out_stream=None)
        self.nGroups = nGroups

        print 'Spacing constraints:'
        print 'X'
        print 'analytic: '
        print self.J['sc', 'turbineX']['J_fwd']
        print 'fd: '
        print self.J['sc', 'turbineX']['J_fd']

        print 'Y'
        print 'wrt X: '
        print self.J['sc', 'turbineY']['J_fwd']
        print 'wrt Y: '
        print self.J['sc', 'turbineY']['J_fd']

        for j in range(nGroups):
            # for i in range(nGroups):
                # print 'Tower%s_max_speed FROM Tower%s fwd: '%(i,j), self.J[('Tower%s_max_speed.shell_buckling'%i, 'turbineH%s'%j)]['J_fwd']
                # print 'Tower%s_max_speed FROM Tower%s fd: '%(i,j), self.J[('Tower%s_max_speed.shell_buckling'%i, 'turbineH%s'%j)]['J_fd']
            for i in range(nGroups):
                print 'Tower%s_max_speed FROM Tower%s fwd: '%(i,j), self.J[('Tower%s_max_speed.freq'%i, 'turbineH%s'%j)]['J_fwd']
                print 'Tower%s_max_speed FROM Tower%s fd: '%(i,j), self.J[('Tower%s_max_speed.freq'%i, 'turbineH%s'%j)]['J_fd']
            # for i in range(nGroups):
                # print 'Tower%s_max_thrust FROM Tower%s fwd: '%(i,j), self.J[('Tower%s_max_thrust.shell_buckling'%i, 'turbineH%s'%j)]['J_fwd']
                # print 'Tower%s_max_thrust FROM Tower%s fd: '%(i,j), self.J[('Tower%s_max_thrust.shell_buckling'%i, 'turbineH%s'%j)]['J_fd']
            for i in range(nGroups):
                print 'Tower%s_max_thrust FROM Tower%s fwd: '%(i,j), self.J[('Tower%s_max_thrust.freq'%i, 'turbineH%s'%j)]['J_fwd']
                print 'Tower%s_max_thrust FROM Tower%s fd: '%(i,j), self.J[('Tower%s_max_thrust.freq'%i, 'turbineH%s'%j)]['J_fd']
        # print 'X fwd: ', self.J[('COE', 'turbineX')]['J_fwd']
        # print 'X fd: ', self.J[('COE', 'turbineX')]['J_fd']
        # print 'Y fwd: ', self.J[('COE', 'turbineY')]['J_fwd']
        # print 'Y fd: ', self.J[('COE', 'turbineY')]['J_fd']


    def test_buck_freq_H1(self):
        for j in range(self.nGroups):
            for i in range(self.nGroups):
                # np.testing.assert_allclose(self.J[('Tower%s_max_speed.shell_buckling'%i, 'turbineH%s'%j)]['J_fwd'], self.J[('Tower%s_max_speed.shell_buckling'%i, 'turbineH%s'%j)]['J_fd'], 1e-6, 1e-6)
                # np.testing.assert_allclose(self.J[('Tower%s_max_thrust.shell_buckling'%i, 'turbineH%s'%j)]['J_fwd'], self.J[('Tower%s_max_thrust.shell_buckling'%i, 'turbineH%s'%j)]['J_fd'], 1e-6, 1e-6)
                np.testing.assert_allclose(self.J[('Tower%s_max_speed.freq'%i, 'turbineH%s'%j)]['J_fwd'], self.J[('Tower%s_max_speed.freq'%i, 'turbineH%s'%j)]['J_fd'], 1e-6, 1e-6)
                np.testing.assert_allclose(self.J[('Tower%s_max_thrust.freq'%i, 'turbineH%s'%j)]['J_fwd'], self.J[('Tower%s_max_thrust.freq'%i, 'turbineH%s'%j)]['J_fd'], 1e-6, 1e-6)

    # def test_COE_d_t(self):
    #     for i in range(self.nGroups):
    #         np.testing.assert_allclose(self.J[('COE', 'd_param%s'%i)]['J_fwd'], self.J[('COE', 'd_param%s'%i)]['J_fd'], 1e-6, 1e-6)
    #         np.testing.assert_allclose(self.J[('COE', 't_param%s'%i)]['J_fwd'], self.J[('COE', 't_param%s'%i)]['J_fd'], 1e-6, 1e-6)
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
#         use_rotor_components = True
#
#         if use_rotor_components:
#             NREL5MWCPCT = pickle.load(open('doc/tune/NREL5MWCPCT_smooth_dict.p'))
#             # NREL5MWCPCT = pickle.load(open('doc/tune/NREL5MWCPCT_smooth_dict.p'))
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
#         spacing = 4.0  # turbine grid spacing in diameters
#         points = np.linspace(start=spacing*rotor_diameter, stop=nRows*spacing*rotor_diameter, num=nRows)
#         xpoints, ypoints = np.meshgrid(points, points)
#         turbineX_bounds = np.ndarray.flatten(xpoints)
#         turbineY_bounds = np.ndarray.flatten(ypoints)
#         xmin = min(turbineX_bounds)
#         xmax = max(turbineX_bounds)
#         ymin = min(turbineY_bounds)
#         ymax = max(turbineY_bounds)
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
#             Cp[turbI] = 0.7737 * 4.0 * 1.0/3.0 * np.power((1. - 1.0/3.0), 2)
#             generatorEfficiency[turbI] = 1.0#0.944
#             yaw[turbI] = 0.     # deg.
#
#         minSpacing = 2.0
#
#         # generate boundary constraint
#         locations = np.zeros((len(turbineX_bounds),2))
#         for i in range(len(turbineX_bounds)):
#             locations[i][0] = turbineX_bounds[i]
#             locations[i][1] = turbineY_bounds[i]
#         print locations
#         boundaryVertices, boundaryNormals = calculate_boundary(locations)
#         nVertices = boundaryVertices.shape[0]
#
#         """Define wind flow"""
#         air_density = 1.1716    # kg/m^3
#
#         windData = "Amalia"
#
#         """Amalia Wind Arrays"""
#         if windData == "Amalia":
#             windSpeeds, windFrequencies, windDirections, nDirections = amaliaWind()
#             nDirections = 2
#             windSpeeds = np.ones(nDirections)*10.
#             windFrequencies = np.ones(nDirections)/nDirections
#             # windDirections = np.linspace(0.,360.-360./nDirections,nDirections)
#             windDirections = np.array([0.,90.])
#
#
#         """Define tower structural properties"""
#         # --- geometry ---
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
#         nPoints = 3
#         nFull = n
#         rhoAir = air_density
#
#         shearExp = 0.17
#
#         numb = 1
#         COE = np.zeros(numb)
#         nGroups = 3
#
#         for j in range(numb):
#             turbineX = xmin+np.random.rand(nTurbs)*(xmax-xmin)
#             turbineY = ymin+np.random.rand(nTurbs)*(ymax-ymin)
#             turbineZ = 73.2+np.random.rand(nTurbs)*60.
#             d_param = np.zeros((nTurbs,3))
#             t_param = np.zeros((nTurbs,3))
#             for i in range(nGroups):
#                 d_param[i] = 3.6+np.random.rand(3)*(6.3-3.6)
#                 t_param[i] = 0.01+np.random.rand(3)*(0.04)
#
#
#             """OpenMDAO"""
#
#             prob = Problem()
#             root = prob.root = Group()
#
#             # root.add('turbineZ', IndepVarComp('turbineZ', turbineZ), promotes=['*'])
#             for i in range(nGroups):
#                 root.add('d_param%s'%i, IndepVarComp('d_param%s'%i, d_param[i]), promotes=['*'])
#                 root.add('t_param%s'%i, IndepVarComp('t_param%s'%i, t_param[i]), promotes=['*'])
#                 root.add('get_z_param%s'%i, get_z(nPoints))
#                 root.add('get_z_full%s'%i, get_z(nFull))
#                 root.add('Tower%s_max_thrust'%i, Tower(nPoints, nFull), promotes=['L_reinforced','m','mrhox','E','sigma_y','gamma_f','gamma_b','rhoAir','z0','zref','shearExp','rho'])
#                 root.add('Tower%s_max_speed'%i, Tower(nPoints, nFull), promotes=['L_reinforced','m','mrhox','E','sigma_y','gamma_f','gamma_b','rhoAir','z0','zref','shearExp','rho'])
#                 root.add('turbineH%s'%i, IndepVarComp('turbineH%s'%i, float(turbineZ[i])), promotes=['*'])
#             root.add('Zs', DeMUX(nTurbs))
#             root.add('hGroups', hGroups(nTurbs), promotes=['*'])
#             root.add('AEPGroup', AEPGroup(nTurbs, nDirections=nDirections,
#                         use_rotor_components=use_rotor_components, datasize=datasize, differentiable=True,
#                         optimizingLayout=False, nSamples=0), promotes=['*'])
#             root.add('COEGroup', COEGroup(nTurbs), promotes=['*'])
#             root.add('maxAEP', AEPobj(), promotes=['*'])
#
#             root.add('spacing_comp', SpacingComp(nTurbines=nTurbs), promotes=['*'])
#
#             # add constraint definitions
#             root.add('spacing_con', ExecComp('sc = wtSeparationSquared-(minSpacing*rotorDiameter[0])**2',
#                                          minSpacing=minSpacing, rotorDiameter=np.zeros(nTurbs),
#                                          sc=np.zeros(((nTurbs-1.)*nTurbs/2.)),
#                                          wtSeparationSquared=np.zeros(((nTurbs-1.)*nTurbs/2.))),
#                                          promotes=['*'])
#
#             if nVertices > 0:
#                 # add component that enforces a convex hull wind farm boundary
#                 root.add('boundary_con', BoundaryComp(nVertices=nVertices, nTurbines=nTurbs), promotes=['*'])
#
#             root.connect('turbineZ', 'Zs.Array')
#             for i in range(nGroups):
#                 root.connect('get_z_param%s.z_param'%i, 'Tower%s_max_thrust.z_param'%i)
#                 root.connect('get_z_full%s.z_param'%i, 'Tower%s_max_thrust.z_full'%i)
#                 root.connect('get_z_param%s.z_param'%i, 'Tower%s_max_speed.z_param'%i)
#                 root.connect('get_z_full%s.z_param'%i, 'Tower%s_max_speed.z_full'%i)
#                 root.connect('Zs.output%s'%i, 'get_z_param%s.turbineZ'%i)
#                 root.connect('Zs.output%s'%i, 'get_z_full%s.turbineZ'%i)
#                 root.connect('Zs.output%s'%i, 'Tower%s_max_thrust.L'%i)
#                 root.connect('Zs.output%s'%i, 'Tower%s_max_speed.L'%i)
#
#                 root.connect('get_z_param%s.z_param'%i, 'Tower%s_max_thrust.z_param'%i)
#                 root.connect('get_z_full%s.z_param'%i, 'Tower%s_max_thrust.z_full'%i)
#
#                 root.connect('Tower%s_max_thrust.mass'%i, 'mass%s'%i)
#
#                 root.connect('d_param%s'%i, 'Tower%s_max_thrust.d_param'%i)
#                 root.connect('t_param%s'%i, 'Tower%s_max_thrust.t_param'%i)
#                 root.connect('d_param%s'%i, 'Tower%s_max_speed.d_param'%i)
#                 root.connect('t_param%s'%i, 'Tower%s_max_speed.t_param'%i)
#
#                 root.connect('Tower%s_max_speed.Mt'%i, 'Tower%s_max_speed.Mt'%i)
#                 root.connect('Tower%s_max_speed.It'%i, 'Tower%s_max_speed.It'%i)
#
#             prob.driver = pyOptSparseDriver()
#             prob.driver.options['optimizer'] = 'SNOPT'
#
#             prob.driver.opt_settings['Function precision'] = 1.0E-8
#
#             # --- Objective ---
#             prob.driver.add_objective('COE', scaler=1.0E-1)
#
#             # --- Design Variables ---
#             prob.driver.add_desvar('turbineX', lower=np.ones(nTurbs)*min(turbineX), upper=np.ones(nTurbs)*max(turbineX), scaler=1.0E-3)
#             prob.driver.add_desvar('turbineY', lower=np.ones(nTurbs)*min(turbineY), upper=np.ones(nTurbs)*max(turbineY), scaler=1.0E-3)
#             # prob.driver.add_desvar('turbineZ', lower=73.2, upper=np.ones(nTurbs)*max(turbineY), scaler=1.0E-2)
#
#             for i in range(nGroups):
#                 prob.driver.add_desvar('turbineH%s'%i, lower=73.2, upper=None, scaler=1.0E-2)
#
#             for i in range(nGroups):
#                 prob.driver.add_desvar('d_param%s'%i, lower=np.array([1.0, 1.0, 3.87]), upper=np.ones(nPoints)*6.3, scaler=1.0E-1)
#                 prob.driver.add_desvar('t_param%s'%i, lower=np.ones(nPoints)*.001, upper=np.ones(nPoints)*0.05, scaler=1.0E1)
#
#             # for direction_id in range(nDirections):
#             #     prob.driver.add_desvar('yaw%i' % direction_id, lower=-30.0, upper=30.0, scaler=1)
#
#             # prob.driver.add_desvar('yaw1', lower=-30.0, upper=30.0, scaler=1)
#
#             # boundary constraint (convex hull)
#             prob.driver.add_constraint('boundaryDistances', lower=np.zeros(nVertices*nTurbs), scaler=1.0)
#             # spacing constraint
#             prob.driver.add_constraint('sc', lower=np.zeros(((nTurbs-1.)*nTurbs/2.)), scaler=1.0/rotor_diameter)
#
#
#             for i in range(nGroups):
#                 prob.driver.add_constraint('Tower%s_max_thrust.shell_buckling'%i, upper=np.ones(nFull))
#                 prob.driver.add_constraint('Tower%s_max_speed.shell_buckling'%i, upper=np.ones(nFull))
#                 freq1p = 0.2  # 1P freq in Hz
#                 prob.driver.add_constraint('Tower%s_max_thrust.freq'%i, lower=1.1*freq1p)
#                 prob.driver.add_constraint('Tower%s_max_speed.freq'%i, lower=1.1*freq1p)
#
#             # ----------------------
#
#             prob.root.ln_solver.options['single_voi_relevance_reduction'] = True
#
#             prob.setup(check=True)
#
#             prob['nGroups'] = nGroups
#             prob['randomize'] = True
#             prob['turbineX'] = turbineX
#             prob['turbineY'] = turbineY
#             # for i in range(nDirections):
#             #     prob['yaw%s'%i] = yaw
#
#             prob['ratedPower'] = np.ones_like(turbineX)*5000. # in kw
#
#             prob['boundaryVertices'] = boundaryVertices
#             prob['boundaryNormals'] = boundaryNormals
#
#             # assign values to constant inputs (not design variables)
#             prob['rotorDiameter'] = rotorDiameter
#             prob['axialInduction'] = axialInduction
#             prob['generatorEfficiency'] = generatorEfficiency
#             prob['air_density'] = air_density
#             prob['windDirections'] = windDirections
#             prob['windFrequencies'] = windFrequencies
#             prob['Uref'] = windSpeeds
#
#             if use_rotor_components == True:
#                 prob['gen_params:windSpeedToCPCT_CP'] = NREL5MWCPCT['CP']
#                 prob['gen_params:windSpeedToCPCT_CT'] = NREL5MWCPCT['CT']
#                 prob['gen_params:windSpeedToCPCT_wind_speed'] = NREL5MWCPCT['wind_speed']
#             else:
#                 prob['Ct_in'] = Ct
#                 prob['Cp_in'] = Cp
#             prob['floris_params:cos_spread'] = 1E12 # turns off cosine spread (just needs to be very large)
#
#             prob['L_reinforced'] = L_reinforced
#             prob['rho'] = rho
#             prob['shearExp'] = shearExp
#             prob['E'] = E
#             prob['gamma_f'] = gamma_f
#             prob['gamma_b'] = gamma_b
#             prob['sigma_y'] = sigma_y
#             prob['m'] = m
#             prob['mrhox'] = mrhox
#             prob['zref'] = 90.
#             prob['z0'] = 0.
#
#             prob['turbineX'] = turbineX
#             prob['turbineY'] = turbineY
#
#             for i in range(nGroups):
#                 prob['Tower%s_max_thrust.Fy'%i] = Fy1
#                 prob['Tower%s_max_thrust.Fx'%i] = Fx1
#                 prob['Tower%s_max_thrust.Fz'%i] = Fz1
#                 prob['Tower%s_max_thrust.Mxx'%i] = Mxx1
#                 prob['Tower%s_max_thrust.Myy'%i] = Myy1
#                 prob['Tower%s_max_thrust.Vel'%i] = wind_Uref1
#                 prob['Tower%s_max_thrust.Mt'%i] = m[0]
#                 prob['Tower%s_max_thrust.It'%i] = mIzz[0]
#
#                 prob['Tower%s_max_speed.Fy'%i] = Fy2
#                 prob['Tower%s_max_speed.Fx'%i] = Fx2
#                 prob['Tower%s_max_speed.Fz'%i] = Fz2
#                 prob['Tower%s_max_speed.Mxx'%i] = Mxx2
#                 prob['Tower%s_max_speed.Myy'%i] = Myy2
#                 prob['Tower%s_max_speed.Vel'%i] = wind_Uref2
#
#
#         prob.run_once()
#
#         # print 'Mass H1: ', prob['Tower1_max_speed.tower1.mass']
#         # print 'Mass H2: ', prob['Tower1_max_thrust.tower1.mass']
#         # print 'd: ', prob['d_param1']
#         # print 't: ', prob['t_param1']
#         self.J = prob.check_total_derivatives(out_stream=None)
#         # print self.J
#         # print '****************************************************************'
#         # print '^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^'
#         # # print 'H1 fwd: ', self.J[('COE', 'turbineH1')]['J_fwd']
#         # print 'H1 fd: ', self.J[('COE', 'turbineH1')]['J_fd']
#         # # print 'H2 fwd: ', self.J[('COE', 'turbineH2')]['J_fwd']
#         # print 'H2 fd: ', self.J[('COE', 'turbineH2')]['J_fd']
#         print 'X fwd: ', self.J[('COE', 'turbineX')]['J_fwd']
#         print 'X fd: ', self.J[('COE', 'turbineX')]['J_fd']
#         print 'Y fwd: ', self.J[('COE', 'turbineY')]['J_fwd']
#         print 'Y fd: ', self.J[('COE', 'turbineY')]['J_fd']
#         # print 'Z fwd: ', self.J[('COE', 'turbineZ')]['J_fwd']
#         # print 'Z fd: ', self.J[('COE', 'turbineZ')]['J_fd']
#         # # print 'd1 fwd: ', self.J[('COE', 'd_paramH1')]['J_fwd']
#         # print 'd1 fd: ', self.J[('COE', 'd_paramH1')]['J_fd']
#         # # print 'd2 fwd: ', self.J[('COE', 'd_paramH2')]['J_fwd']
#         # print 'd2 fd: ', self.J[('COE', 'd_paramH2')]['J_fd']
#         # # print 't1 fwd: ', self.J[('COE', 't_paramH1')]['J_fwd']
#         # print 't1 fd: ', self.J[('COE', 't_paramH1')]['J_fd']
#         # # print 't2 fwd: ', self.J[('COE', 't_paramH2')]['J_fwd']
#         # print 't2 fd: ', self.J[('COE', 't_paramH2')]['J_fd']
#         # print 'T1, stress: ', self.J[('COE', 'TowerH1.tower1.stress')]['J_fd']
#
#     # def test_COE_H1(self):
#     #     np.testing.assert_allclose(self.J[('COE', 'turbineZ')]['J_fwd'], self.J[('COE', 'turbineZ')]['J_fd'], 1e-6, 1e-6)
#
#     # def test_COE_d_t(self):
#     #     for i in range(25):
#     #         np.testing.assert_allclose(self.J[('COE', 'd_param%s'%i)]['J_fwd'], self.J[('COE', 'd_param%s'%i)]['J_fd'], 1e-6, 1e-6)
#     #         np.testing.assert_allclose(self.J[('COE', 't_param%s'%i)]['J_fwd'], self.J[('COE', 't_param%s'%i)]['J_fd'], 1e-6, 1e-6)
#
#     def test_COE_X_Y(self):
#         np.testing.assert_allclose(self.J[('COE', 'turbineX')]['J_fwd'], self.J[('COE', 'turbineX')]['J_fd'], 1e-6, 1e-6)
#         np.testing.assert_allclose(self.J[('COE', 'turbineY')]['J_fwd'], self.J[('COE', 'turbineY')]['J_fd'], 1e-6, 1e-6)
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
# class TestAEP(unittest.TestCase):
#
#
#     def setUp(self):
#
#         use_rotor_components = True
#
#         if use_rotor_components:
#             NREL5MWCPCT = pickle.load(open('../doc/tune/NREL5MWCPCT_smooth_dict.p'))
#             # print(NREL5MWCPCT)
#             # NREL5MWCPCT = pickle.Unpickler(open('./input_files/NREL5MWCPCT.p')).load()
#             datasize = NREL5MWCPCT['CP'].size
#         else:
#             datasize = 0
#
#         rotor_diameter = 126.4
#
#         # nRows = 1
#         # nTurbs = nRows**2
#         # spacing = 3   # turbine grid spacing in diameters
#         # points = np.linspace(start=spacing*rotor_diameter, stop=nRows*spacing*rotor_diameter, num=nRows)
#         # xpoints, ypoints = np.meshgrid(points, points)
#         # turbineX = np.ndarray.flatten(xpoints)
#         # turbineY = np.ndarray.flatten(ypoints)
#
#         nTurbs = 4
#         num = 100.
#         turbineX = np.array([0.,1.*num,2.*num,3.*num])
#         turbineY = np.array([0.,200.,400.,600.])
#         # nTurbs = 2
#         # num = 0.
#         # turbineX = np.array([0.,1.*num])
#         # turbineY = np.array([0.,200.])
#
#
#         turbineH1 = 125.5
#         turbineH2 = 135.
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
#         nDirections = 1
#         windSpeeds = np.ones(nDirections)*10.
#         windFrequencies = np.ones(nDirections)/nDirections
#         windDirections = np.linspace(0.,360.-360./nDirections,nDirections)
#         # windSpeeds = np.array([5.,10.])
#         # windDirections = np.array([0.,90.])
#
#         shearExp = 0.2
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
#         # root.deriv_options['form'] = 'central'
#         # root.deriv_options['step_size'] = 500.
#         # root.deriv_options['step_calc'] = 'relative'
#
#         root.add('turbineH1', IndepVarComp('turbineH1', turbineH1), promotes=['*'])
#         root.add('turbineH2', IndepVarComp('turbineH2', turbineH2), promotes=['*'])
#         root.add('Uref', IndepVarComp('Uref', windSpeeds), promotes=['*'])
#         root.add('getTurbineZ', getTurbineZ(nTurbs), promotes=['*'])
#         root.add('AEPGroup', AEPGroup(nTurbs, nDirections=nDirections,
#                     use_rotor_components=use_rotor_components, datasize=datasize, differentiable=True,
#                     optimizingLayout=False, nSamples=0), promotes=['*'])
#
#         prob.driver.add_objective('AEP', scaler=1.0E-8)
#         # prob.driver.add_objective('wtVelocity0', scaler=1.0)
#
#         prob.driver.add_desvar('turbineH1', lower=75., upper=None)
#         prob.driver.add_desvar('turbineH2', lower=75., upper=None)
#         prob.driver.add_desvar('turbineX', lower=75., upper=None)
#         prob.driver.add_desvar('turbineY', lower=75., upper=None)
#         prob.driver.add_desvar('Uref', lower=None, upper=None)
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
#         # self.J = prob.check_partial_derivatives(out_stream=sys.stdout)
#         self.J = prob.check_total_derivatives(out_stream=None)
#         print self.J
#         # print self.J
#         print 'H1'
#         print 'Finite Difference'
#         print self.J[('AEP', 'turbineH1')]['J_fd']
#         print 'Analytic'
#         print self.J[('AEP', 'turbineH1')]['J_fwd']
#
#         print 'H2'
#         print 'Finite Difference'
#         print self.J[('AEP', 'turbineH2')]['J_fd']
#         print 'Analytic'
#         print self.J[('AEP', 'turbineH2')]['J_fwd']
#
#         print 'X'
#         print 'Finite Difference'
#         print self.J[('AEP', 'turbineX')]['J_fd']
#         print 'Analytic'
#         print self.J[('AEP', 'turbineX')]['J_fwd']
#
#         print 'Y'
#         print 'Finite Difference'
#         print self.J[('AEP', 'turbineY')]['J_fd']
#         print 'Rev'
#         print self.J[('AEP', 'turbineY')]['J_rev']
#         print 'Analytic'
#         print self.J[('AEP', 'turbineY')]['J_fwd']
#
#         print 'Speeds?: ', prob['wtVelocity0']
#
#         # print 'Analytic'
#         # print self.J[('windSpeeds', 'turbineH1')]['J_fwd']
#         #
#         #
#         # print 'Analytic'
#         # print self.J[('windSpeeds', 'turbineH2')]['J_fwd']
#         # print 'Finite Difference'
#         # print self.J[('windSpeeds', 'turbineH2')]['J_fd']
#         #
#         # print 'Analytic'
#         # print self.J[('turbineZ', 'turbineH2')]['J_fwd']
#         # print 'Finite Difference'
#         # print self.J[('turbineZ', 'turbineH2')]['J_fd']
#
#     def testWRT_x_y(self):
#     #     np.testing.assert_allclose(self.J[('AEP', 'turbineX')]['J_fwd'], self.J[('AEP', 'turbineX')]['J_fd'], 1e-3, 1e-3)
#     #     np.testing.assert_allclose(self.J[('AEP', 'turbineY')]['J_fwd'], self.J[('AEP', 'turbineY')]['J_fd'], 1e-3, 1e-3)
#         np.testing.assert_allclose(self.J[('AEP', 'turbineX')]['J_fwd'], self.J[('AEP', 'turbineX')]['J_fd'], 1e-4, 1e-4)
#         np.testing.assert_allclose(self.J[('AEP', 'turbineY')]['J_fwd'], self.J[('AEP', 'turbineY')]['J_fd'], 1e-4, 1e-4)
#
#     def testWRT_H1_H2(self):
#     #     np.testing.assert_allclose(self.J[('AEP', 'turbineX')]['J_fwd'], self.J[('AEP', 'turbineX')]['J_fd'], 1e-3, 1e-3)
#     #     np.testing.assert_allclose(self.J[('AEP', 'turbineY')]['J_fwd'], self.J[('AEP', 'turbineY')]['J_fd'], 1e-3, 1e-3)
#         np.testing.assert_allclose(self.J[('AEP', 'turbineH1')]['J_fwd'], self.J[('AEP', 'turbineH1')]['J_fd'], 1e-4, 1e-4)
#         np.testing.assert_allclose(self.J[('AEP', 'turbineH2')]['J_fwd'], self.J[('AEP', 'turbineH2')]['J_fd'], 1e-4, 1e-4)

    # def test_Uref(self):
    #     np.testing.assert_allclose(self.J[('AEP', 'Uref')]['J_fwd'], self.J[('AEP', 'Uref')]['J_fd'], 1e-3, 1e-3)


# #
#


if __name__ == '__main__':
    unittest.main()
