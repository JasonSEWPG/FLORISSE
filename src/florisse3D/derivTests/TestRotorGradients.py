import numpy as np
from openmdao.api import Group, Problem, IndepVarComp, pyOptSparseDriver, view_tree, profile
from FLORISSE3D.SimpleRotorSE import SimpleRotorSE, create_rotor_functions
import unittest

from time import time

class TestSimpleRotorSE(unittest.TestCase):

    def setUp(self):
        interp_spline_ratedQ, interp_spline_blade_mass, interp_spline_Vrated, interp_spline_I1, \
            interp_spline_I2, interp_spline_I3, interp_spline_ratedT, interp_spline_extremeT = create_rotor_functions()

        self.rtol = 1E-4
        self.atol = 1E-4

        prob = Problem()
        root = prob.root = Group()

        rated_power = float(np.random.rand(1)*9500.+500.)
        rotor_diameter = float(np.random.rand(1)*120.+50.)
        # rated_power = 500.
        # rotor_diameter = 50.

        print rated_power
        print rotor_diameter


        root.add('rotorDiameter', IndepVarComp('rotorDiameter', float(rotor_diameter)), promotes=['*'])
        root.add('turbineRating', IndepVarComp('turbineRating', float(rated_power)), promotes=['*'])

        root.add('Rotor', SimpleRotorSE(interp_spline_ratedQ, interp_spline_blade_mass, interp_spline_Vrated, interp_spline_I1, interp_spline_I2, interp_spline_I3, interp_spline_ratedT, interp_spline_extremeT), promotes=['*'])

        #ratedT, ratedQ, blade_mass, Vrated, extremeT, or I
        obj = 'blade_mass'
        self.obj = obj

        prob.driver.add_objective(obj, scaler=1E-3)

        # select design variables
        prob.driver.add_desvar('turbineRating', scaler=1E-3)
        prob.driver.add_desvar('rotorDiameter', scaler=1E-2)

        # initialize problem
        prob.setup()

        # run problem
        prob.run_once()

        # pass results to self for use with unit test
        self.J = prob.check_total_derivatives(out_stream=None)

        print 'wrt rotorDiameter'
        print 'fwd'
        print self.J[(obj, 'rotorDiameter')]['J_fwd']
        print 'fd'
        print self.J[(obj, 'rotorDiameter')]['J_fd']

        print 'wrt turbineRating'
        print 'fwd'
        print self.J[(obj, 'turbineRating')]['J_fwd']
        print 'fd'
        print self.J[(obj, 'turbineRating')]['J_fd']


    def testObj(self):

        np.testing.assert_allclose(self.J[(self.obj, 'turbineRating')]['J_fwd'], self.J[(self.obj, 'turbineRating')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J[(self.obj, 'rotorDiameter')]['J_fwd'], self.J[(self.obj, 'rotorDiameter')]['J_fd'], self.rtol, self.atol)


if __name__ == "__main__":
    unittest.main()


# indep_list = ['turbineX', 'turbineY', 'yaw', 'rotorDiameter']
# unknown_list = ['dir_power0']
# self.J = prob.calc_gradient(indep_list, unknown_list, return_format='array')
# print self.J
