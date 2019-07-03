from __future__ import print_function, division, absolute_import

import unittest

import numpy as np

import openmdao.api as om
from florisse.floris import Floris


class TotalDerivTestsFlorisAEP(unittest.TestCase):

    def setUp(self):
        print('Test FLORIS TAPENADE derivatives')
        nTurbines = 3
        self.rtol = 1E-6
        self.atol = 1E-6

        np.random.seed(seed=10)

        turbineX = np.random.rand(nTurbines)*3000.
        turbineY = np.random.rand(nTurbines)*50.
        hubHeight = np.random.rand(nTurbines)*100.+50.

        # initialize input variable arrays
        rotorDiameter = np.ones(nTurbines)*np.random.random()*150.
        axialInduction = np.ones(nTurbines)*np.random.random()*(1./3.)
        Ct = np.ones(nTurbines)*np.random.random()
        yaw = np.random.rand(nTurbines)*60. - 30.

        # Define flow properties
        nDirections = 1
        wind_speed = float(np.random.rand(1)*10.)       # m/s
        air_density = 1.1716    # kg/m^3
        windDirections = np.random.rand(nDirections)*360.0
        windFrequencies = np.random.rand(nDirections)

        shearExp = 0.15
        z0 = 0.
        zref = 50.

        prob = om.Problem()
        model = prob.model

        ivc = om.IndepVarComp()
        ivc.add_output('turbineXw', turbineX)
        ivc.add_output('turbineYw', turbineY)
        ivc.add_output('yaw0', yaw)
        ivc.add_output('hubHeight', hubHeight)
        ivc.add_output('rotorDiameter', rotorDiameter)

        model.add_subsystem('desvar', ivc, promotes=['*'])


        model.add_subsystem('floris', Floris(nTurbines=nTurbines, differentiable=True, use_rotor_components=False,
                                             nSamples=0, verbose=False),promotes=['*'])

        # set up optimizer
        prob.driver = om.pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'SNOPT'
        model.add_objective('wtVelocity0', scaler=1.0)

        # select design variables
        model.add_design_var('turbineXw', scaler=1.0)
        model.add_design_var('turbineYw', scaler=1.0)
        model.add_design_var('hubHeight', scaler=1.0)
        model.add_design_var('yaw0', scaler=1.0)
        model.add_design_var('rotorDiameter', scaler=1.0)

        # initialize problem
        prob.setup()

        # assign values to constant inputs (not design variables)
        prob['turbineXw'] = turbineX
        prob['turbineYw'] = turbineY
        prob['hubHeight'] = hubHeight
        prob['yaw0'] = yaw
        prob['rotorDiameter'] = rotorDiameter
        prob['axialInduction'] = axialInduction
        prob['Ct'] = Ct
        prob['wind_speed'] = wind_speed
        prob['axialInduction'] = axialInduction
        prob['floris_params:shearExp'] = shearExp
        prob['floris_params:z_ref'] = zref
        prob['floris_params:z0'] = z0

        # run problem
        prob.run_model()

        print(prob['wtVelocity0'])

        # pass results to self for use with unit test
        self.J = prob.check_totals(out_stream=None)
        self.nDirections = nDirections

        print('Check derivatives')
        print('wrt turbineXw')
        print('FD: ', self.J[('floris.wtVelocity0', 'desvar.turbineXw')]['J_fd'])
        print('FWD: ', self.J[('floris.wtVelocity0', 'desvar.turbineXw')]['J_fwd'])

        print('wrt turbineYw')
        print('FD: ', self.J[('floris.wtVelocity0', 'desvar.turbineYw')]['J_fd'])
        print('FWD: ', self.J[('floris.wtVelocity0', 'desvar.turbineYw')]['J_fwd'])

        print('wrt hubHeight')
        print('FD: ', self.J[('floris.wtVelocity0', 'desvar.hubHeight')]['J_fd'])
        print('FWD: ', self.J[('floris.wtVelocity0', 'desvar.hubHeight')]['J_fwd'])

        print('wrt yaw0')
        print('FD: ', self.J[('floris.wtVelocity0', 'desvar.yaw0')]['J_fd'])
        print('FWD: ', self.J[('floris.wtVelocity0', 'desvar.yaw0')]['J_fwd'])

        print('wrt rotorDiameter')
        print('FD: ', self.J[('floris.wtVelocity0', 'desvar.rotorDiameter')]['J_fd'])
        print('FWD: ', self.J[('floris.wtVelocity0', 'desvar.rotorDiameter')]['J_fwd'])


    def testObj(self):

        np.testing.assert_allclose(self.J[('floris.wtVelocity0', 'desvar.turbineXw')]['J_fwd'],
                                   self.J[('floris.wtVelocity0', 'desvar.turbineXw')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J[('floris.wtVelocity0', 'desvar.turbineYw')]['J_fwd'],
                                   self.J[('floris.wtVelocity0', 'desvar.turbineYw')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J[('floris.wtVelocity0', 'desvar.hubHeight')]['J_fwd'],
                                   self.J[('floris.wtVelocity0', 'desvar.hubHeight')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J[('floris.wtVelocity0', 'desvar.yaw0')]['J_fwd'],
                                   self.J[('floris.wtVelocity0', 'desvar.yaw0')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J[('floris.wtVelocity0', 'desvar.rotorDiameter')]['J_fwd'],
                                   self.J[('floris.wtVelocity0', 'desvar.rotorDiameter')]['J_fd'], self.rtol, self.atol)


if __name__ == "__main__":
    unittest.main()
