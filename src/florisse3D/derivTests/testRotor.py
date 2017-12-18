import unittest
import numpy as np
from openmdao.api import Group, Problem, IndepVarComp, pyOptSparseDriver, view_tree, profile
from rotorse.rotor import *
from rotorse.rotoraerodefaults import *
from rotorse.rotoraero import *

# class test_Rotor(unittest.TestCase):
#
#     def setUp(self):
#
#         self.rtol = 1.E-6
#         self.atol = 1.E-6
#
#         import numpy as np
#         from openmdao.api import Group, Problem, IndepVarComp, pyOptSparseDriver, view_tree, profile
#         from FLORISSE3D.setupOptimization import *
#         from FLORISSE3D.simpleTower import Tower
#         from FLORISSE3D.GeneralWindFarmComponents import calculate_boundary, SpacingComp,\
#                     BoundaryComp, get_z, getTurbineZ, AEPobj, DeMUX, hGroups, randomStart,\
#                     getRotorDiameter, getRatedPower, DeMUX, Myy_estimate, bladeLengthComp, minHeight
#         from FLORISSE3D.COE import COEGroup
#         from FLORISSE3D.floris import AEPGroup
#         from FLORISSE3D.rotorComponents import getRating, freqConstraintGroup
#         import cPickle as pickle
#         from sys import argv
#         from rotorse.rotor import RotorSE
#         import os
#
#         from time import time
#
#         if __name__ == '__main__':
#             nDirections = amaliaWind({})
#
#             """setup the turbine locations"""
#
#             nGroups = 1
#
#             rotor_diameter = 126.4
#
#             nDirections = 1
#
#             nPoints = 3
#             nFull = 15
#
#             shearExp = 0.15
#             rotorDiameter = np.array([126.4, 70.,150.,155.,141.])
#             ratedPower = np.array([5000.,200.,2000.,3000.,3004.])
#
#             """OpenMDAO"""
#
#             start_setup = time()
#             prob = Problem()
#             root = prob.root = Group()
#
#             #Design Variables
#             for i in range(nGroups):
#                 root.add('ratedPower%s'%i, IndepVarComp('ratedPower%s'%i, float(ratedPower[i]), units='kW'), promotes=['*'])
#                 root.add('rotorDiameter%s'%i, IndepVarComp('rotorDiameter%s'%i, float(rotorDiameter[i])), promotes=['*'])
#
#
#             for i in range(nGroups):
#                 root.add('bladeLength%s'%i, bladeLengthComp()) #have derivatives
#                 topName = 'Rotor%s.'%i
#                 root.add('Rotor%s'%i, RotorSE(topName=topName, naero=17, nstr=38, npower=20)) #TODO check derivatives?
#                 root.add('split_I%s'%i, DeMUX(6)) #have derivatives
#                 root.add('Myy_estimate%s'%i, Myy_estimate()) #have derivatives
#
#             for i in range(nGroups):
#                 root.connect('ratedPower%s'%i, 'Rotor%s.control:ratedPower'%i) #TODO commented out line in RotorSE
#                 root.connect('Rotor%s.I_all_blades'%i, 'split_I%s.Array'%i)
#
#                 root.connect('rotorDiameter%s'%i, 'Myy_estimate%s.rotor_diameter'%i)
#
#                 root.connect('rotorDiameter%s'%i,'bladeLength%s.rotor_diameter'%i)
#                 root.connect('bladeLength%s.blade_length'%i,'Rotor%s.bladeLength'%i)
#
#             prob.driver = pyOptSparseDriver()
#             prob.driver.options['optimizer'] = 'SNOPT'
#             prob.driver.opt_settings['Major iterations limit'] = 1000
#             prob.driver.opt_settings['Major optimality tolerance'] = 1.1E-4
#             prob.driver.opt_settings['Major feasibility tolerance'] = 1.1E-4
#
#
#             root.Rotor0.deriv_options['type'] = 'fd'
#             root.Rotor0.deriv_options['form'] = 'central'
#             root.Rotor0.deriv_options['step_size'] = 1.E-4
#             root.Rotor0.deriv_options['step_calc'] = 'relative'
#
#             # prob.driver.add_objective('bladeLength.blade_length', scaler=0.1)
#             prob.driver.add_objective('Rotor0.mass.blade_mass', scaler=0.1)
#
#             for i in range(nGroups):
#                 prob.driver.add_desvar('rotorDiameter%s'%i, lower=10., upper=None, scaler=1.)
#                 prob.driver.add_desvar('ratedPower%s'%i, lower=0., upper=10000., scaler=1.)
#
#             prob.root.ln_solver.options['single_voi_relevance_reduction'] = True
#             prob.setup(check=True)
#
#             setupRotor(nGroups, prob)
#
#             for i in range(nGroups):
#                 prob['Rotor%s.afOptions'%i] = {}
#                 prob['Rotor%s.afOptions'%i]['GradientOptions'] = {}
#                 prob['Rotor%s.afOptions'%i]['GradientOptions']['ComputeGradient'] = False
#
#             prob.run_once()
#
#             print prob['Rotor0.mass.blade_mass']
#             # pass results to self for use with unit test
#             # self.J = prob.check_total_derivatives(out_stream=None)
#
#             # print 'wrt rotorDiameter'
#             # print 'FD: ', self.J['bladeLength.blade_length','rotorDiameter']['J_fd']
#             # print 'FWD: ', self.J['bladeLength.blade_length','rotorDiameter']['J_fwd']
#
#             # print 'wrt ratedPower'
#             # print 'FD: ', self.J['Rotor0.mass.blade_mass','ratedPower']['J_fd']
#             # print 'FWD: ', self.J['Rotor0.mass.blade_mass','ratedPower']['J_fwd']
#
#     def testAll(self):
#         np.testing.assert_allclose(self.J[('bladeLength.blade_length', 'rotorDiameter')]['J_fwd'], self.J[('bladeLength.blade_length', 'rotorDiameter')]['J_fd'], self.rtol, self.atol)

class test_Rotor(unittest.TestCase):

    def setUp(self):

        self.rtol = 1.E-6
        self.atol = 1.E-6

        """OpenMDAO"""

        start_setup = time()
        prob = Problem()
        root = prob.root = Group()

        #Design Variables
        for i in range(nGroups):
            root.add('ratedPower%s'%i, IndepVarComp('ratedPower%s'%i, float(ratedPower[i]), units='kW'), promotes=['*'])
            root.add('rotorDiameter%s'%i, IndepVarComp('rotorDiameter%s'%i, float(rotorDiameter[i])), promotes=['*'])


        for i in range(nGroups):
            root.add('RegulatedPowerCurveGroup', RegulatedPowerCurveGroup(20)) #TODO check derivatives?
            root.connect('bladeLength%s.blade_length'%i,'Rotor%s.bladeLength'%i)

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'SNOPT'
        prob.driver.opt_settings['Major iterations limit'] = 1000
        prob.driver.opt_settings['Major optimality tolerance'] = 1.1E-4
        prob.driver.opt_settings['Major feasibility tolerance'] = 1.1E-4


        root.Rotor0.deriv_options['type'] = 'fd'
        root.Rotor0.deriv_options['form'] = 'central'
        root.Rotor0.deriv_options['step_size'] = 1.E-4
        root.Rotor0.deriv_options['step_calc'] = 'relative'

        # prob.driver.add_objective('bladeLength.blade_length', scaler=0.1)
        prob.driver.add_objective('Rotor0.mass.blade_mass', scaler=0.1)

        for i in range(nGroups):
            prob.driver.add_desvar('rotorDiameter%s'%i, lower=10., upper=None, scaler=1.)
            prob.driver.add_desvar('ratedPower%s'%i, lower=0., upper=10000., scaler=1.)

        prob.root.ln_solver.options['single_voi_relevance_reduction'] = True
        prob.setup(check=True)

        setupRotor(nGroups, prob)

        for i in range(nGroups):
            prob['Rotor%s.afOptions'%i] = {}
            prob['Rotor%s.afOptions'%i]['GradientOptions'] = {}
            prob['Rotor%s.afOptions'%i]['GradientOptions']['ComputeGradient'] = False

        prob.run_once()

        print prob['Rotor0.mass.blade_mass']
        # pass results to self for use with unit test
        # self.J = prob.check_total_derivatives(out_stream=None)

        # print 'wrt rotorDiameter'
        # print 'FD: ', self.J['bladeLength.blade_length','rotorDiameter']['J_fd']
        # print 'FWD: ', self.J['bladeLength.blade_length','rotorDiameter']['J_fwd']

        # print 'wrt ratedPower'
        # print 'FD: ', self.J['Rotor0.mass.blade_mass','ratedPower']['J_fd']
        # print 'FWD: ', self.J['Rotor0.mass.blade_mass','ratedPower']['J_fwd']

    def testAll(self):
        np.testing.assert_allclose(self.J[('bladeLength.blade_length', 'rotorDiameter')]['J_fwd'], self.J[('bladeLength.blade_length', 'rotorDiameter')]['J_fd'], self.rtol, self.atol)

if __name__ == "__main__":
    unittest.main()
