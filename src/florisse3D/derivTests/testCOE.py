import unittest
from openmdao.api import pyOptSparseDriver, ExecComp, IndepVarComp, Problem
from FLORISSE3D.COE import *
import numpy as np

# class test_rotorCostComponent(unittest.TestCase):
#
#     def setUp(self):
#
#         nTurbs = 15
#         nGroups = 2
#         groups = np.array([0,1,2,3])
#         self.nTurbs = nTurbs
#         self.nGroups = nGroups
#         self.rtol = 1E-3
#         self.atol = 1E-3
#
#         rotorCost = np.random.rand(nGroups)*3000000.+500000.
#         hGroup = np.zeros(nTurbs)
#         for i in range(nTurbs/nGroups):
#             for j in range(nGroups):
#                 hGroup[i*nGroups+j] = groups[j]
#
#         print hGroup
#
#         prob = Problem()
#         root = prob.root = Group()
#
#         for i in range(nGroups):
#             root.add('rotorCost%s'%i, IndepVarComp('rotorCost%s'%i, float(rotorCost[i])), promotes=['*'])
#         root.add('rotorCostComponent', rotorCostComponent(nTurbs, nGroups), promotes=['*'])
#
#         # set up optimizer
#         prob.driver = pyOptSparseDriver()
#         prob.driver.options['optimizer'] = 'SNOPT'
#         prob.driver.add_objective('rotorCost', scaler=1.E-1)
#
#         # select design variables
#         for i in range(nGroups):
#             prob.driver.add_desvar('rotorCost%s'%i, scaler=1E-6)
#
#         # initialize problem
#         prob.setup()
#
#         prob['hGroup'] = hGroup
#
#         # run problem
#         prob.run_once()
#
#         # pass results to self for use with unit test
#         self.J = prob.check_total_derivatives(out_stream=None)
#
#         for i in range(nGroups):
#             print 'FWD:'
#             print self.J[('rotorCost', 'rotorCost%s'%i)]['J_fwd']
#             print 'REV:'
#             print self.J[('rotorCost', 'rotorCost%s'%i)]['J_rev']
#             print 'FD:'
#             print self.J[('rotorCost', 'rotorCost%s'%i)]['J_fd']
#
#     def test(self):
#         for i in range(self.nGroups):
#             np.testing.assert_allclose(self.J[('rotorCost', 'rotorCost%s'%i)]['J_fwd'], self.J[('rotorCost', 'rotorCost%s'%i)]['J_fd'], self.rtol, self.atol)


# class test_nacelleCostComponent(unittest.TestCase):
#
#     def setUp(self):
#
#         nTurbs = 17
#         nGroups = 2
#         self.nGroups = nGroups
#         self.nTurbs = nTurbs
#         self.rtol = 1E-3
#         self.atol = 1E-3
#
#         groups = np.array([0,1,2,3,4,5])
#
#         nacelleCost = np.random.rand(nGroups)*3000000.+500000.
#
#         hGroup = np.zeros(nTurbs)
#         for i in range(nTurbs/nGroups):
#             for j in range(nGroups):
#                 hGroup[i*nGroups+j] = groups[j]
#
#         prob = Problem()
#         root = prob.root = Group()
#
#         for i in range(nGroups):
#             root.add('nacelleCost%s'%i, IndepVarComp('nacelleCost%s'%i, float(nacelleCost[i])), promotes=['*'])
#         root.add('nacelleCostComponent', nacelleCostComponent(nTurbs,nGroups), promotes=['*'])
#
#         # set up optimizer
#         prob.driver = pyOptSparseDriver()
#         prob.driver.options['optimizer'] = 'SNOPT'
#         prob.driver.add_objective('nacelleCost', scaler=1.E-1)
#
#         # select design variables
#         for i in range(nGroups):
#             prob.driver.add_desvar('nacelleCost%s'%i, scaler=1E-6)
#
#         # initialize problem
#         prob.setup()
#
#         prob['hGroup'] = hGroup
#
#         # run problem
#         prob.run_once()
#
#         # pass results to self for use with unit test
#         self.J = prob.check_total_derivatives(out_stream=None)
#
#         for i in range(nGroups):
#                 print 'FWD:'
#                 print self.J[('nacelleCost', 'nacelleCost%s'%i)]['J_fwd']
#                 print 'REV:'
#                 print self.J[('nacelleCost', 'nacelleCost%s'%i)]['J_rev']
#                 print 'FD:'
#                 print self.J[('nacelleCost', 'nacelleCost%s'%i)]['J_fd']
#
#     def test(self):
#         for i in range(self.nGroups):
#             np.testing.assert_allclose(self.J[('nacelleCost', 'nacelleCost%s'%i)]['J_fwd'], self.J[('nacelleCost', 'nacelleCost%s'%i)]['J_fd'], self.rtol, self.atol)


class test_farmCost(unittest.TestCase):

    def setUp(self):

        nTurbs = 10
        nGroups = 2
        groups = np.array([0,1,2,3,4])
        self.nGroups = nGroups
        self.nTurbs = nTurbs
        self.rtol = 1E-3
        self.atol = 1E-3

        hGroup = np.zeros(nTurbs)
        for i in range(nTurbs/nGroups):
            for j in range(nGroups):
                hGroup[i*nGroups+j] = groups[j]

        #I THINK THE GRADIENTS ARE GOOD
        #if you go too small on these though the small step size for FD
        #gives an error (start running into machine epsilon I beleive)
        mass = np.random.rand(nGroups)*200000.*10000.
        rotorCost = np.random.rand(nTurbs)*300000.+500000.
        nacelleCost = np.random.rand(nTurbs)*300000.+500000.


        prob = Problem()
        root = prob.root = Group()

        for i in range(nGroups):
            root.add('mass%s'%i, IndepVarComp('mass%s'%i, float(mass[i])), promotes=['*'])
        root.add('rotorCost', IndepVarComp('rotorCost', rotorCost), promotes=['*'])
        root.add('nacelleCost', IndepVarComp('nacelleCost', nacelleCost), promotes=['*'])

        root.add('farmCost', farmCost(nTurbs, nGroups), promotes=['*'])

        # set up optimizer
        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'SNOPT'
        prob.driver.add_objective('cost', scaler=1.E-7)
        root.deriv_options['form'] = 'central'
        root.deriv_options['step_size'] = 0.1
        root.deriv_options['step_calc'] = 'relative'

        # select design variables
        for i in range(nGroups):
            prob.driver.add_desvar('mass%s'%i, scaler=1E-7)
        prob.driver.add_desvar('rotorCost', scaler=1E-5)
        prob.driver.add_desvar('nacelleCost', scaler=1E-5)

        # initialize problem
        prob.setup()

        prob['hGroup'] = hGroup

        # run problem
        prob.run_once()

        # cost1 = prob['cost']
        # nacelleCost += step
        # prob['nacelleCost'] = nacelleCost
        # prob.run_once()
        # print (prob['cost']-cost1)/step

        # pass results to self for use with unit test
        self.J = prob.check_total_derivatives(out_stream=None)

    # def testMass(self):
    #     for i in range(self.nGroups):
    #         np.testing.assert_allclose(self.J[('cost', 'mass%s'%i)]['J_fwd'], self.J[('cost', 'mass%s'%i)]['J_fd'], self.rtol, self.atol)
    #
    # def testRotor(self):
    #     np.testing.assert_allclose(self.J[('cost', 'rotorCost')]['J_fwd'], self.J[('cost', 'rotorCost')]['J_fd'], self.rtol, self.atol)

    def testNacelle(self):
        np.testing.assert_allclose(self.J[('cost', 'nacelleCost')]['J_fwd'], self.J[('cost', 'nacelleCost')]['J_fd'], self.rtol, self.atol)



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
