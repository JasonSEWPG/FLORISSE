import unittest
from openmdao.api import pyOptSparseDriver, ExecComp, IndepVarComp, Problem
from FLORISSE3D.COE import *
import numpy as np

class test_rotorCostComponent(unittest.TestCase):

    def setUp(self):

        nTurbs = 15
        self.nTurbs = nTurbs
        self.rtol = 1E-3
        self.atol = 1E-3

        rotorCost = np.random.rand(nTurbs)*3000000.+500000.

        prob = Problem()
        root = prob.root = Group()

        for i in range(nTurbs):
            root.add('rotorCost%s'%i, IndepVarComp('rotorCost%s'%i, float(rotorCost[i])), promotes=['*'])
        root.add('rotorCostComponent', rotorCostComponent(nTurbs), promotes=['*'])

        # set up optimizer
        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'SNOPT'
        prob.driver.add_objective('rotorCost', scaler=1.E-1)

        # select design variables
        for i in range(nTurbs):
            prob.driver.add_desvar('rotorCost%s'%i, scaler=1E-6)

        # initialize problem
        prob.setup()

        # run problem
        prob.run_once()

        # pass results to self for use with unit test
        self.J = prob.check_total_derivatives(out_stream=None)

        print 'FWD:'
        print self.J[('rotorCost', 'rotorCost%s'%i)]['J_fwd']
        print 'REV:'
        print self.J[('rotorCost', 'rotorCost%s'%i)]['J_rev']
        print 'FD:'
        print self.J[('rotorCost', 'rotorCost%s'%i)]['J_fd']

    def test(self):
        for i in range(self.nTurbs):
            np.testing.assert_allclose(self.J[('rotorCost', 'rotorCost%s'%i)]['J_fwd'], self.J[('rotorCost', 'rotorCost%s'%i)]['J_fd'], self.rtol, self.atol)

#
# class test_nacelleCostComponent(unittest.TestCase):
#
#     def setUp(self):
#
#         nTurbs = 17
#         self.nTurbs = nTurbs
#         self.rtol = 1E-3
#         self.atol = 1E-3
#
#         nacelleCost = np.random.rand(nTurbs)*3000000.+500000.
#
#         prob = Problem()
#         root = prob.root = Group()
#
#         for i in range(nTurbs):
#             root.add('nacelleCost%s'%i, IndepVarComp('nacelleCost%s'%i, float(nacelleCost[i])), promotes=['*'])
#         root.add('nacelleCostComponent', nacelleCostComponent(nTurbs), promotes=['*'])
#
#         # set up optimizer
#         prob.driver = pyOptSparseDriver()
#         prob.driver.options['optimizer'] = 'SNOPT'
#         prob.driver.add_objective('nacelleCost', scaler=1.E-1)
#
#         # select design variables
#         for i in range(nTurbs):
#             prob.driver.add_desvar('nacelleCost%s'%i, scaler=1E-6)
#
#         # initialize problem
#         prob.setup()
#
#         # run problem
#         prob.run_once()
#
#         # pass results to self for use with unit test
#         self.J = prob.check_total_derivatives(out_stream=None)
#
#     def test(self):
#         for i in range(self.nTurbs):
#             np.testing.assert_allclose(self.J[('nacelleCost', 'nacelleCost%s'%i)]['J_fwd'], self.J[('nacelleCost', 'nacelleCost%s'%i)]['J_fd'], self.rtol, self.atol)
#
#
# class test_farmCost(unittest.TestCase):
#
#     def setUp(self):
#
#         nTurbs = 7
#         self.nTurbs = nTurbs
#         self.rtol = 1E-3
#         self.atol = 1E-3
#
#         #THE GRADIENTS ARE GOOD
#         #if you go too small on these though the small step size for FD
#         #gives an error (start running into machine epsilon I beleive)
#         mass = np.random.rand(nTurbs)*200.*100.
#         rotorCost = np.random.rand(nTurbs)*3000.+5000.
#         nacelleCost = np.random.rand(nTurbs)*3000.+5000.
#
#
#         prob = Problem()
#         root = prob.root = Group()
#
#         for i in range(nTurbs):
#             root.add('mass%s'%i, IndepVarComp('mass%s'%i, float(mass[i])), promotes=['*'])
#         root.add('rotorCost', IndepVarComp('rotorCost', rotorCost), promotes=['*'])
#         root.add('nacelleCost', IndepVarComp('nacelleCost', nacelleCost), promotes=['*'])
#
#         root.add('farmCost', farmCost(nTurbs), promotes=['*'])
#
#         # set up optimizer
#         prob.driver = pyOptSparseDriver()
#         prob.driver.options['optimizer'] = 'SNOPT'
#         prob.driver.add_objective('cost', scaler=1.E-7)
#
#         # select design variables
#         for i in range(nTurbs):
#             prob.driver.add_desvar('mass%s'%i, scaler=1E-7)
#         prob.driver.add_desvar('rotorCost', scaler=1E-7)
#         prob.driver.add_desvar('nacelleCost', scaler=1E-7)
#
#         # initialize problem
#         prob.setup()
#
#         # run problem
#         prob.run_once()
#
#         # pass results to self for use with unit test
#         self.J = prob.check_total_derivatives(out_stream=None)
#
#     def test(self):
#         for i in range(self.nTurbs):
#             np.testing.assert_allclose(self.J[('cost', 'mass%s'%i)]['J_fwd'], self.J[('cost', 'mass%s'%i)]['J_fd'], self.rtol, self.atol)
#         np.testing.assert_allclose(self.J[('cost', 'rotorCost')]['J_fwd'], self.J[('cost', 'rotorCost')]['J_fd'], self.rtol, self.atol)
#         np.testing.assert_allclose(self.J[('cost', 'nacelleCost')]['J_fwd'], self.J[('cost', 'nacelleCost')]['J_fd'], self.rtol, self.atol)
#
#
# class test_COEComponenet(unittest.TestCase):
#
#     def setUp(self):
#
#         nTurbs = 15
#         self.nTurbs = nTurbs
#         self.rtol = 1E-7
#         self.atol = 1E-7
#
#         cost = np.random.rand(1)*30000000.+10000000.
#         AEP = np.random.rand(1)*100000000.
#         BOS = np.random.rand(1)*10000000.+2000000.
#
#         prob = Problem()
#         root = prob.root = Group()
#
#         root.add('cost', IndepVarComp('cost', float(cost)), promotes=['*'])
#         root.add('AEP', IndepVarComp('AEP', float(AEP)), promotes=['*'])
#         root.add('BOS', IndepVarComp('BOS', float(BOS)), promotes=['*'])
#         root.add('COEComponent', COEComponent(nTurbs), promotes=['*'])
#
#         # set up optimizer
#         prob.driver = pyOptSparseDriver()
#         prob.driver.options['optimizer'] = 'SNOPT'
#         prob.driver.add_objective('COE', scaler=1.E-1)
#
#         # select design variables
#         prob.driver.add_desvar('cost', scaler=1E-6)
#         prob.driver.add_desvar('AEP', scaler=1E-6)
#         prob.driver.add_desvar('BOS', scaler=1E-6)
#
#         # initialize problem
#         prob.setup()
#
#         # run problem
#         prob.run_once()
#
#         # pass results to self for use with unit test
#         self.J = prob.check_total_derivatives(out_stream=None)
#
#     def test(self):
#         np.testing.assert_allclose(self.J[('COE', 'cost')]['J_fwd'], self.J[('COE', 'cost')]['J_fd'], self.rtol, self.atol)
#         np.testing.assert_allclose(self.J[('COE', 'AEP')]['J_fwd'], self.J[('COE', 'AEP')]['J_fd'], self.rtol, self.atol)
#         np.testing.assert_allclose(self.J[('COE', 'BOS')]['J_fwd'], self.J[('COE', 'BOS')]['J_fd'], self.rtol, self.atol)


if __name__ == "__main__":
    unittest.main()
