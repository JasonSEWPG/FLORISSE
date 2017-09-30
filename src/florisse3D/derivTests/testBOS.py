import unittest
from openmdao.api import pyOptSparseDriver, ExecComp, IndepVarComp, Problem
from FLORISSE3D.BOS import *
import numpy as np


class test_transportationCost(unittest.TestCase):

    def setUp(self):

        nTurbs = 15
        self.nTurbs = nTurbs
        self.rtol = 1E-3
        self.atol = 1E-3

        transportDist = np.random.rand(1)*500.
        cost = np.random.rand(1)*100000000.

        prob = Problem()
        root = prob.root = Group()

        root.add('cost', IndepVarComp('cost', float(cost)), promotes=['*'])
        root.add('transportationCost', transportationCost(nTurbs), promotes=['*'])

        # set up optimizer
        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'SNOPT'
        prob.driver.add_objective('transportation_cost', scaler=1.E-1)

        # select design variables
        prob.driver.add_desvar('cost', scaler=1E-6)

        # initialize problem
        prob.setup()

        prob['transportDist'] = transportDist

        # run problem
        prob.run_once()

        # pass results to self for use with unit test
        self.J = prob.check_total_derivatives(out_stream=None)

    def test(self):
        np.testing.assert_allclose(self.J[('transportation_cost', 'cost')]['J_fwd'], self.J[('transportation_cost', 'cost')]['J_fd'], self.rtol, self.atol)


# TODO
class test_powerPerformanceCost(unittest.TestCase):
    #TODO this is a piecewise function...what to do?

    def setUp(self):

        nTurbs = 15
        self.nTurbs = nTurbs
        self.rtol = 1E-8
        self.atol = 1E-8

        turbineZ = np.random.rand(nTurbs)*75.+45.

        prob = Problem()
        root = prob.root = Group()

        root.add('turbineZ', IndepVarComp('turbineZ', turbineZ), promotes=['*'])
        root.add('powerPerformanceCost', powerPerformanceCost(nTurbs), promotes=['*'])

        # set up optimizer
        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'SNOPT'
        prob.driver.add_objective('power_performance_cost', scaler=1.E-5)

        # select design variables
        prob.driver.add_desvar('turbineZ')

        # initialize problem
        prob.setup()

        # run problem
        prob.run_once()

        # pass results to self for use with unit test
        self.J = prob.check_total_derivatives(out_stream=None)

    def test(self):
        np.testing.assert_allclose(self.J[('power_performance_cost', 'turbineZ')]['J_fwd'], self.J[('power_performance_cost', 'turbineZ')]['J_fd'], self.rtol, self.atol)


class test_accessRoadCost(unittest.TestCase):

    def setUp(self):

        nTurbs = 15
        self.nTurbs = nTurbs
        self.rtol = 1E-4
        self.atol = 1E-4

        rotorDiameter = np.random.rand(nTurbs)*150.+20.

        prob = Problem()
        root = prob.root = Group()

        root.add('rotorDiameter', IndepVarComp('rotorDiameter', rotorDiameter), promotes=['*'])
        root.add('accessRoadCost', accessRoadCost(nTurbs), promotes=['*'])

        # set up optimizer
        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'SNOPT'
        prob.driver.add_objective('access_road_cost')

        # select design variables
        prob.driver.add_desvar('rotorDiameter')

        # initialize problem
        prob.setup()

        # run problem
        prob.run_once()

        # pass results to self for use with unit test
        self.J = prob.check_total_derivatives(out_stream=None)

    def test(self):
        np.testing.assert_allclose(self.J[('access_road_cost', 'rotorDiameter')]['J_fwd'], self.J[('access_road_cost', 'rotorDiameter')]['J_fd'], self.rtol, self.atol)


class test_foundationCost(unittest.TestCase):

    def setUp(self):

        nTurbs = 15
        self.nTurbs = nTurbs
        self.rtol = 1E-3
        self.atol = 1E-3

        rotorDiameter = np.random.rand(nTurbs)*150.+20.
        turbineZ = np.random.rand(nTurbs)*120.+45.
        ratedPower = np.random.rand(nTurbs)*10000.+125.

        prob = Problem()
        root = prob.root = Group()

        root.add('rotorDiameter', IndepVarComp('rotorDiameter', rotorDiameter), promotes=['*'])
        root.add('turbineZ', IndepVarComp('turbineZ', turbineZ), promotes=['*'])
        root.add('ratedPower', IndepVarComp('ratedPower', ratedPower), promotes=['*'])
        root.add('foundationCost', foundationCost(nTurbs), promotes=['*'])

        # set up optimizer
        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'SNOPT'
        prob.driver.add_objective('foundation_cost')

        # select design variables
        prob.driver.add_desvar('rotorDiameter')
        prob.driver.add_desvar('turbineZ')
        prob.driver.add_desvar('ratedPower')

        # initialize problem
        prob.setup()

        # run problem
        prob.run_once()

        # pass results to self for use with unit test
        self.J = prob.check_total_derivatives(out_stream=None)

    def test(self):
        np.testing.assert_allclose(self.J[('foundation_cost', 'rotorDiameter')]['J_fwd'], self.J[('foundation_cost', 'rotorDiameter')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J[('foundation_cost', 'turbineZ')]['J_fwd'], self.J[('foundation_cost', 'turbineZ')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J[('foundation_cost', 'ratedPower')]['J_fwd'], self.J[('foundation_cost', 'ratedPower')]['J_fd'], self.rtol, self.atol)


class test_erectionCost(unittest.TestCase):

    def setUp(self):

        nTurbs = 15
        self.nTurbs = nTurbs
        self.rtol = 1E-4
        self.atol = 1E-4

        turbineZ = np.random.rand(nTurbs)*120.+45.
        ratedPower = np.random.rand(nTurbs)*10000.+125.

        prob = Problem()
        root = prob.root = Group()

        root.add('turbineZ', IndepVarComp('turbineZ', turbineZ), promotes=['*'])
        root.add('ratedPower', IndepVarComp('ratedPower', ratedPower), promotes=['*'])
        root.add('erectionCost', erectionCost(nTurbs), promotes=['*'])

        # set up optimizer
        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'SNOPT'
        prob.driver.add_objective('erection_cost')

        # select design variables
        prob.driver.add_desvar('turbineZ')
        prob.driver.add_desvar('ratedPower')

        # initialize problem
        prob.setup()

        # run problem
        prob.run_once()

        # pass results to self for use with unit test
        self.J = prob.check_total_derivatives(out_stream=None)

    def test(self):
        np.testing.assert_allclose(self.J[('erection_cost', 'turbineZ')]['J_fwd'], self.J[('erection_cost', 'turbineZ')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J[('erection_cost', 'ratedPower')]['J_fwd'], self.J[('erection_cost', 'ratedPower')]['J_fd'], self.rtol, self.atol)


class test_electircalMaterialsCost(unittest.TestCase):

    def setUp(self):

        nTurbs = 15
        self.nTurbs = nTurbs
        self.rtol = 1E-4
        self.atol = 1E-4

        rotorDiameter = np.random.rand(nTurbs)*150.+20.

        prob = Problem()
        root = prob.root = Group()

        root.add('rotorDiameter', IndepVarComp('rotorDiameter', rotorDiameter), promotes=['*'])
        root.add('electircalMaterialsCost', electircalMaterialsCost(nTurbs), promotes=['*'])

        # set up optimizer
        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'SNOPT'
        prob.driver.add_objective('electrical_materials_cost')

        # select design variables
        prob.driver.add_desvar('rotorDiameter')

        # initialize problem
        prob.setup()

        # run problem
        prob.run_once()

        # pass results to self for use with unit test
        self.J = prob.check_total_derivatives(out_stream=None)

    def test(self):
        np.testing.assert_allclose(self.J[('electrical_materials_cost', 'rotorDiameter')]['J_fwd'], self.J[('electrical_materials_cost', 'rotorDiameter')]['J_fd'], self.rtol, self.atol)


class test_electircalInstallationCost(unittest.TestCase):

    def setUp(self):

        nTurbs = 15
        self.nTurbs = nTurbs
        self.rtol = 1E-5
        self.atol = 1E-5

        rotorDiameter = np.random.rand(nTurbs)*150.+20.

        prob = Problem()
        root = prob.root = Group()

        root.add('rotorDiameter', IndepVarComp('rotorDiameter', rotorDiameter), promotes=['*'])
        root.add('electircalInstallationCost', electircalInstallationCost(nTurbs), promotes=['*'])

        # set up optimizer
        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'SNOPT'
        prob.driver.add_objective('electrical_installation_cost')

        # select design variables
        prob.driver.add_desvar('rotorDiameter')

        # initialize problem
        prob.setup()

        # run problem
        prob.run_once()

        # pass results to self for use with unit test
        self.J = prob.check_total_derivatives(out_stream=None)

    def test(self):
        np.testing.assert_allclose(self.J[('electrical_installation_cost', 'rotorDiameter')]['J_fwd'], self.J[('electrical_installation_cost', 'rotorDiameter')]['J_fd'], self.rtol, self.atol)


class test_insuranceCost_insurance_cost(unittest.TestCase):

    def setUp(self):

        nTurbs = 15
        self.nTurbs = nTurbs
        self.rtol = 1E-3
        self.atol = 1E-3

        ratedPower = np.random.rand(nTurbs)*10000.+250.
        cost = np.random.rand(1)*50000000.
        foundation_cost = np.random.rand(1)*5000000.

        prob = Problem()
        root = prob.root = Group()

        root.add('ratedPower', IndepVarComp('ratedPower', ratedPower), promotes=['*'])
        root.add('cost', IndepVarComp('cost', float(cost)), promotes=['*'])
        root.add('foundation_cost', IndepVarComp('foundation_cost', float(foundation_cost)), promotes=['*'])
        root.add('insuranceCost', insuranceCost(nTurbs), promotes=['*'])

        # set up optimizer
        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'SNOPT'
        prob.driver.add_objective('insurance_cost')

        # select design variables
        prob.driver.add_desvar('ratedPower')
        prob.driver.add_desvar('cost')
        prob.driver.add_desvar('foundation_cost')

        # initialize problem
        prob.setup()

        # run problem
        prob.run_once()

        # pass results to self for use with unit test
        self.J = prob.check_total_derivatives(out_stream=None)

    def test(self):
        np.testing.assert_allclose(self.J[('insurance_cost', 'ratedPower')]['J_fwd'], self.J[('insurance_cost', 'ratedPower')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J[('insurance_cost', 'cost')]['J_fwd'], self.J[('insurance_cost', 'cost')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J[('insurance_cost', 'foundation_cost')]['J_fwd'], self.J[('insurance_cost', 'foundation_cost')]['J_fd'], self.rtol, self.atol)


class test_insuranceCost_alpha_insurance(unittest.TestCase):

    def setUp(self):

        nTurbs = 15
        self.nTurbs = nTurbs
        self.rtol = 1E-3
        self.atol = 1E-3

        ratedPower = np.random.rand(nTurbs)*10000.+250.
        cost = np.random.rand(1)*50000000.
        foundation_cost = np.random.rand(1)*5000000.

        prob = Problem()
        root = prob.root = Group()

        root.add('ratedPower', IndepVarComp('ratedPower', ratedPower), promotes=['*'])
        root.add('cost', IndepVarComp('cost', float(cost)), promotes=['*'])
        root.add('foundation_cost', IndepVarComp('foundation_cost', float(foundation_cost)), promotes=['*'])
        root.add('insuranceCost', insuranceCost(nTurbs), promotes=['*'])

        # set up optimizer
        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'SNOPT'
        prob.driver.add_objective('alpha_insurance')

        # select design variables
        prob.driver.add_desvar('ratedPower')
        prob.driver.add_desvar('cost')
        prob.driver.add_desvar('foundation_cost')

        # initialize problem
        prob.setup()

        # run problem
        prob.run_once()

        # pass results to self for use with unit test
        self.J = prob.check_total_derivatives(out_stream=None)

    def test(self):
        np.testing.assert_allclose(self.J[('alpha_insurance', 'ratedPower')]['J_fwd'], self.J[('alpha_insurance', 'ratedPower')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J[('alpha_insurance', 'cost')]['J_fwd'], self.J[('alpha_insurance', 'cost')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J[('alpha_insurance', 'foundation_cost')]['J_fwd'], self.J[('alpha_insurance', 'foundation_cost')]['J_fd'], self.rtol, self.atol)



class test_markupCost_markup_cost(unittest.TestCase):

    def setUp(self):

        nTurbs = 15
        self.nTurbs = nTurbs
        self.rtol = 1E-3
        self.atol = 1E-3

        transportation_cost = np.random.rand(1)*5000000.

        prob = Problem()
        root = prob.root = Group()

        root.add('transportation_cost', IndepVarComp('transportation_cost', float(transportation_cost)), promotes=['*'])
        root.add('markupCost', markupCost(nTurbs), promotes=['*'])

        # set up optimizer
        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'SNOPT'
        prob.driver.add_objective('markup_cost')

        # select design variables
        prob.driver.add_desvar('transportation_cost')

        # initialize problem
        prob.setup()

        # run problem
        prob.run_once()

        # pass results to self for use with unit test
        self.J = prob.check_total_derivatives(out_stream=None)

    def test(self):
        np.testing.assert_allclose(self.J[('markup_cost', 'transportation_cost')]['J_fwd'], self.J[('markup_cost', 'transportation_cost')]['J_fd'], self.rtol, self.atol)


class test_BOScalc(unittest.TestCase):

    def setUp(self):

        nTurbs = 15
        self.nTurbs = nTurbs
        self.rtol = 1E-3
        self.atol = 1E-3

        transportation_cost = np.random.rand(1)*50000.
        power_performance_cost = np.random.rand(1)*50000.
        access_road_cost = np.random.rand(1)*50000.
        foundation_cost = np.random.rand(1)*50000.
        erection_cost = np.random.rand(1)*50000.
        electrical_materials_cost = np.random.rand(1)*50000.
        electrical_installation_cost = np.random.rand(1)*50000.
        insurance_cost = np.random.rand(1)*50000.
        markup_cost = np.random.rand(1)*50000.
        alpha_insurance = 1.2
        alpha_markup = 1.5
        cost = np.random.rand(1)*50000.

        prob = Problem()
        root = prob.root = Group()

        root.add('transportation_cost', IndepVarComp('transportation_cost', float(transportation_cost)), promotes=['*'])
        root.add('power_performance_cost', IndepVarComp('power_performance_cost', float(power_performance_cost)), promotes=['*'])
        root.add('access_road_cost', IndepVarComp('access_road_cost', float(access_road_cost)), promotes=['*'])
        root.add('foundation_cost', IndepVarComp('foundation_cost', float(foundation_cost)), promotes=['*'])
        root.add('erection_cost', IndepVarComp('erection_cost', float(erection_cost)), promotes=['*'])
        root.add('electrical_materials_cost', IndepVarComp('electrical_materials_cost', float(electrical_materials_cost)), promotes=['*'])
        root.add('electrical_installation_cost', IndepVarComp('electrical_installation_cost', float(electrical_installation_cost)), promotes=['*'])
        root.add('insurance_cost', IndepVarComp('insurance_cost', float(insurance_cost)), promotes=['*'])
        root.add('markup_cost', IndepVarComp('markup_cost', float(markup_cost)), promotes=['*'])
        root.add('alpha_insurance', IndepVarComp('alpha_insurance', float(alpha_insurance)), promotes=['*'])
        root.add('alpha_markup', IndepVarComp('alpha_markup', float(alpha_markup)), promotes=['*'])
        root.add('cost', IndepVarComp('cost', float(cost)), promotes=['*'])

        root.add('BOScalc', BOScalc(nTurbs), promotes=['*'])

        # set up optimizer
        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'SNOPT'
        prob.driver.add_objective('BOS')

        # select design variables
        prob.driver.add_desvar('transportation_cost')
        prob.driver.add_desvar('power_performance_cost')
        prob.driver.add_desvar('access_road_cost')
        prob.driver.add_desvar('foundation_cost')
        prob.driver.add_desvar('erection_cost')
        prob.driver.add_desvar('electrical_materials_cost')
        prob.driver.add_desvar('electrical_installation_cost')
        prob.driver.add_desvar('insurance_cost')
        prob.driver.add_desvar('markup_cost')
        prob.driver.add_desvar('cost')

        # initialize problem
        prob.setup()

        # run problem
        prob.run_once()

        # pass results to self for use with unit test
        self.J = prob.check_total_derivatives(out_stream=None)

    def test(self):
        np.testing.assert_allclose(self.J[('BOS', 'transportation_cost')]['J_fwd'], self.J[('BOS', 'transportation_cost')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J[('BOS', 'power_performance_cost')]['J_fwd'], self.J[('BOS', 'power_performance_cost')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J[('BOS', 'access_road_cost')]['J_fwd'], self.J[('BOS', 'access_road_cost')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J[('BOS', 'foundation_cost')]['J_fwd'], self.J[('BOS', 'foundation_cost')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J[('BOS', 'erection_cost')]['J_fwd'], self.J[('BOS', 'erection_cost')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J[('BOS', 'electrical_materials_cost')]['J_fwd'], self.J[('BOS', 'electrical_materials_cost')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J[('BOS', 'electrical_installation_cost')]['J_fwd'], self.J[('BOS', 'electrical_installation_cost')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J[('BOS', 'insurance_cost')]['J_fwd'], self.J[('BOS', 'insurance_cost')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J[('BOS', 'markup_cost')]['J_fwd'], self.J[('BOS', 'markup_cost')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J[('BOS', 'cost')]['J_fwd'], self.J[('BOS', 'cost')]['J_fd'], self.rtol, self.atol)




if __name__ == "__main__":
    unittest.main()
