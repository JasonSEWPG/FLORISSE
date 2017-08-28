import unittest
from openmdao.api import pyOptSparseDriver, ExecComp, IndepVarComp, Problem
from FLORISSE3D.GeneralWindFarmComponents import *
import numpy as np


# class TestOrganizeWindSpeeds(unittest.TestCase):
#
#
#     def setUp(self):
#         nTurbines = 9
#         speeds = np.array([10.,10.,5.,3.])
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
#


# class TestPowWind(unittest.TestCase):
#
#
#     def setUp(self):
#
#         self.rtol = 1E-5
#         self.atol = 1E-5
#
#         nDirections = 27
#         nTurbs = 15
#         Uref = np.random.rand(nDirections)*15.
#         turbineZ = np.random.rand(nTurbs)*100.*45.
#
#         prob = Problem()
#         root = prob.root = Group()
#
#         root.add('turbineZ', IndepVarComp('turbineZ', turbineZ), promotes=['*'])
#         root.add('PowWind', PowWind(nDirections, nTurbs), promotes=['*'])
#
#         prob.driver = pyOptSparseDriver()
#         prob.driver.options['optimizer'] = 'SNOPT'
#
#         prob.driver.add_objective('windSpeeds')
#         prob.driver.add_desvar('turbineZ')
#
#         prob.setup()
#
#         prob['Uref'] = Uref
#         prob['zref'] = 50.
#         prob['z0'] = 0.
#         prob['shearExp'] = float(np.random.rand(1)*0.22+0.08)
#
#         prob.run_once()
#
#         self.J = prob.check_total_derivatives(out_stream=None)
#
#     def test_mass(self):
#         np.testing.assert_allclose(self.J[('windSpeeds', 'turbineZ')]['J_fwd'], self.J[('windSpeeds', 'turbineZ')]['J_fd'], self.rtol, self.atol)



class TestWindDirectionPower(unittest.TestCase):


    def setUp(self):

        self.rtol = 1E-5
        self.atol = 1E-5

        nTurbs = 15
        rotorDiameter = np.random.rand(nTurbs)*100.+45.
        wtVelocity0 = np.random.rand(nTurbs)*15.+2.
        ratedPower = np.random.rand(nTurbs)*10000.+125.

        prob = Problem()
        root = prob.root = Group()

        root.add('rotorDiameter', IndepVarComp('rotorDiameter', rotorDiameter), promotes=['*'])
        root.add('wtVelocity0', IndepVarComp('wtVelocity0', wtVelocity0), promotes=['*'])
        root.add('ratedPower', IndepVarComp('ratedPower', ratedPower), promotes=['*'])
        root.add('WindDirectionPower', WindDirectionPower(nTurbs), promotes=['*'])

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'SNOPT'

        # prob.driver.add_objective('wtPower0')
        prob.driver.add_objective('dir_power0')

        prob.driver.add_desvar('rotorDiameter')
        prob.driver.add_desvar('wtVelocity0')
        prob.driver.add_desvar('ratedPower')

        prob.setup()

        prob.run_once()

        # print prob['dir_power0']
        # print prob['rotorDiameter']
        # print prob['ratedPower']
        # print prob['wtVelocity0']

        self.J = prob.check_total_derivatives(out_stream=None)

        print 'FWD'
        print self.J[('dir_power0', 'ratedPower')]['J_fwd']
        print 'FD'
        print self.J[('dir_power0', 'ratedPower')]['J_fd']


    def test_mass(self):
        np.testing.assert_allclose(self.J[('dir_power0', 'rotorDiameter')]['J_fwd'], self.J[('dir_power0', 'rotorDiameter')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J[('dir_power0', 'wtVelocity0')]['J_fwd'], self.J[('dir_power0', 'wtVelocity0')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J[('dir_power0', 'ratedPower')]['J_fwd'], self.J[('dir_power0', 'ratedPower')]['J_fd'], self.rtol, self.atol)

if __name__ == "__main__":
    unittest.main()
