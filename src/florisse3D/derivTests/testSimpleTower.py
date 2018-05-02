import unittest
from openmdao.api import pyOptSparseDriver, ExecComp, IndepVarComp, Problem
from FLORISSE3D.simpleTower import *
import numpy as np


# class TestTopMassAdder(unittest.TestCase):
#
#
#     def setUp(self):
#
#         self.rtol = 1E-5
#         self.atol = 1E-5
#
#         rotor_mass = float(np.random.rand(1)*250000.)
#         nacelle_mass = float(np.random.rand(1)*250000.)
#
#         prob = Problem()
#         root = prob.root = Group()
#
#         root.add('rotor_mass', IndepVarComp('rotor_mass', float(rotor_mass)), promotes=['*'])
#         root.add('nacelle_mass', IndepVarComp('nacelle_mass', float(nacelle_mass)), promotes=['*'])
#         root.add('topMassAdder', topMassAdder(), promotes=['*'])
#
#         prob.driver = pyOptSparseDriver()
#         prob.driver.options['optimizer'] = 'SNOPT'
#
#         prob.driver.add_objective('m')
#
#         prob.driver.add_desvar('rotor_mass')
#         prob.driver.add_desvar('nacelle_mass')
#
#         prob.setup()
#         prob.run_once()
#
#         self.J = prob.check_total_derivatives(out_stream=None)
#
#         print 'FWD rotor'
#         print self.J[('m', 'rotor_mass')]['J_fwd']
#         print 'FD rotor'
#         print self.J[('m', 'rotor_mass')]['J_fd']
#
#         print 'FWD nacelle'
#         print self.J[('m', 'nacelle_mass')]['J_fwd']
#         print 'FD nacelle'
#         print self.J[('m', 'nacelle_mass')]['J_fd']
#
#
#     def test(self):
#         np.testing.assert_allclose(self.J[('m', 'rotor_mass')]['J_fwd'], self.J[('m', 'rotor_mass')]['J_fd'], self.rtol, self.atol)
#         np.testing.assert_allclose(self.J[('m', 'nacelle_mass')]['J_fwd'], self.J[('m', 'nacelle_mass')]['J_fd'], self.rtol, self.atol)


# class TestCalcMass(unittest.TestCase):
#
#
#     def setUp(self):
#
#         self.rtol = 1E-5
#         self.atol = 1E-5
#
#         nFull = 15
#         H = float(np.random.rand(1)*100.+35.)
#         z_full = np.linspace(0.,H,nFull)
#
#         dlow = float(np.random.rand(1)*2.+1.)
#         dhigh = float(np.random.rand(1)*2.+3.)
#         d_full = np.linspace(dlow,dhigh,nFull)
#
#         tlow = float(np.random.rand(1)*0.01+0.006)
#         thigh = float(np.random.rand(1)*0.04+0.016)
#         t_full = np.linspace(tlow,thigh,nFull)
#
#         rho = np.ones(nFull)*8050.
#
#         prob = Problem()
#         root = prob.root = Group()
#
#         root.add('z_full', IndepVarComp('z_full', z_full), promotes=['*'])
#         root.add('d_full', IndepVarComp('d_full', d_full), promotes=['*'])
#         root.add('t_full', IndepVarComp('t_full', t_full), promotes=['*'])
#         root.add('calcMass', calcMass(nFull), promotes=['*'])
#
#         prob.driver = pyOptSparseDriver()
#         prob.driver.options['optimizer'] = 'SNOPT'
#
#         prob.driver.add_objective('mass')
#
#         prob.driver.add_desvar('z_full')
#         prob.driver.add_desvar('d_full')
#         prob.driver.add_desvar('t_full')
#
#         prob.setup()
#
#         prob['rho'] = rho
#
#         prob.run_once()
#
#         self.J = prob.check_total_derivatives(out_stream=None)
#
#         print 'FWD z'
#         print self.J[('mass', 'z_full')]['J_fwd']
#         print 'FD z'
#         print self.J[('mass', 'z_full')]['J_fd']
#
#         print 'FWD d'
#         print self.J[('mass', 'd_full')]['J_fwd']
#         print 'FD d'
#         print self.J[('mass', 'd_full')]['J_fd']
#
#         print 'FWD t'
#         print self.J[('mass', 't_full')]['J_fwd']
#         print 'FD t'
#         print self.J[('mass', 't_full')]['J_fd']
#
#
#     def test(self):
#         np.testing.assert_allclose(self.J[('mass', 'z_full')]['J_fwd'], self.J[('mass', 'z_full')]['J_fd'], self.rtol, self.atol)
#         np.testing.assert_allclose(self.J[('mass', 'd_full')]['J_fwd'], self.J[('mass', 'd_full')]['J_fd'], self.rtol, self.atol)
#         np.testing.assert_allclose(self.J[('mass', 't_full')]['J_fwd'], self.J[('mass', 't_full')]['J_fd'], self.rtol, self.atol)


class TestTowerDiscretization(unittest.TestCase):


    def setUp(self):

        self.rtol = 1E-5
        self.atol = 1E-5

        nPoints = 3
        nFull = 15
        H = float(np.random.rand(1)*100.+35.)
        z_param = np.linspace(0.,H,nPoints)
        z_full = np.linspace(0.,H,nFull)

        dlow = float(np.random.rand(1)*2.+1.)
        dhigh = float(np.random.rand(1)*2.+3.)
        d_param = np.linspace(dlow,dhigh,nPoints)

        tlow = float(np.random.rand(1)*0.01+0.006)
        thigh = float(np.random.rand(1)*0.04+0.016)
        t_param = np.linspace(tlow,thigh,nPoints)

        prob = Problem()
        root = prob.root = Group()

        root.add('z_param', IndepVarComp('z_param', z_param), promotes=['*'])
        root.add('z_full', IndepVarComp('z_full', z_full), promotes=['*'])
        root.add('d_param', IndepVarComp('d_param', d_param), promotes=['*'])
        root.add('t_param', IndepVarComp('t_param', t_param), promotes=['*'])
        root.add('TowerDiscretization', TowerDiscretization(nPoints, nFull), promotes=['*'])

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'SNOPT'

        # prob.driver.add_objective('d_full')
        prob.driver.add_objective('t_full')

        prob.driver.add_desvar('z_param')
        prob.driver.add_desvar('z_full')
        prob.driver.add_desvar('d_param')
        prob.driver.add_desvar('t_param')

        prob.setup()

        prob.run_once()

        self.J = prob.check_total_derivatives(out_stream=None)

        # print 'D_FULL'
        # print 'FWD z full'
        # print self.J[('d_full', 'z_full')]['J_fwd']
        # print 'FD z full'
        # print self.J[('d_full', 'z_full')]['J_fd']
        #
        # print 'FWD z param'
        # print self.J[('d_full', 'z_param')]['J_fwd']
        # print 'FD z param'
        # print self.J[('d_full', 'z_param')]['J_fd']
        #
        # print 'FWD d'
        # print self.J[('d_full', 'd_param')]['J_fwd']
        # print 'FD d'
        # print self.J[('d_full', 'd_param')]['J_fd']
        #
        # print 'FWD t'
        # print self.J[('d_full', 't_param')]['J_fwd']
        # print 'FD t'
        # print self.J[('d_full', 't_param')]['J_fd']

        print 'T_FULL'
        print 'FWD z full'
        print self.J[('t_full', 'z_full')]['J_fwd']
        print 'FD z full'
        print self.J[('t_full', 'z_full')]['J_fd']

        print 'FWD z param'
        print self.J[('t_full', 'z_param')]['J_fwd']
        print 'FD z param'
        print self.J[('t_full', 'z_param')]['J_fd']

        print 'FWD d'
        print self.J[('t_full', 'd_param')]['J_fwd']
        print 'FD d'
        print self.J[('t_full', 'd_param')]['J_fd']

        print 'FWD t'
        print self.J[('t_full', 't_param')]['J_fwd']
        print 'FD t'
        print self.J[('t_full', 't_param')]['J_fd']


    def test(self):
        # np.testing.assert_allclose(self.J[('d_full', 'z_full')]['J_fwd'], self.J[('d_full', 'z_full')]['J_fd'], self.rtol, self.atol)
        # np.testing.assert_allclose(self.J[('d_full', 'z_param')]['J_fwd'], self.J[('d_full', 'z_param')]['J_fd'], self.rtol, self.atol)
        # np.testing.assert_allclose(self.J[('d_full', 'd_param')]['J_fwd'], self.J[('d_full', 'd_param')]['J_fd'], self.rtol, self.atol)
        # np.testing.assert_allclose(self.J[('d_full', 't_param')]['J_fwd'], self.J[('d_full', 't_param')]['J_fd'], self.rtol, self.atol)

        np.testing.assert_allclose(self.J[('t_full', 'z_full')]['J_fwd'], self.J[('t_full', 'z_full')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J[('t_full', 'z_param')]['J_fwd'], self.J[('t_full', 'z_param')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J[('t_full', 'd_param')]['J_fwd'], self.J[('t_full', 'd_param')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J[('t_full', 't_param')]['J_fwd'], self.J[('t_full', 't_param')]['J_fd'], self.rtol, self.atol)


# class testPowWindTower(unittest.TestCase):
#
#
#     def setUp(self):
#
#         self.rtol = 1E-3
#         self.atol = 1E-3
#
#         nFull = 5
#         H = float(np.random.rand(1)*100.+35.)
#         z_full = np.linspace(0.,H,nFull)
#
#         Vel = float(np.random.rand(1)*15.+8.)
#         zref = 50.
#         z0 = 0.
#         shearExp = float(np.random.rand(1)*0.22+0.08)
#
#         prob = Problem()
#         root = prob.root = Group()
#
#         root.add('z_full', IndepVarComp('z_full', z_full), promotes=['*'])
#         root.add('powWindTower', powWindTower(nFull), promotes=['*'])
#
#         prob.driver = pyOptSparseDriver()
#         prob.driver.options['optimizer'] = 'SNOPT'
#
#         prob.driver.add_objective('towerSpeeds')
#
#         prob.driver.add_desvar('z_full')
#
#         prob.setup()
#
#         prob['Vel'] = Vel
#         prob['zref'] = zref
#         prob['z0'] = z0
#         prob['shearExp'] = shearExp
#
#         prob.run_once()
#
#         self.J = prob.check_total_derivatives(out_stream=None)
#
#         print 'Tower Speeds: ', prob['towerSpeeds']
#         print 'FWD z'
#         print self.J[('towerSpeeds', 'z_full')]['J_fwd']
#         print 'FD z'
#         print self.J[('towerSpeeds', 'z_full')]['J_fd']
#
#
#     def test(self):
#         np.testing.assert_allclose(self.J[('towerSpeeds', 'z_full')]['J_fwd'], self.J[('towerSpeeds', 'z_full')]['J_fd'], self.rtol, self.atol)


# class testDynamic_q(unittest.TestCase):
#
#
#     def setUp(self):
#
#         self.rtol = 1E-5
#         self.atol = 1E-5
#
#         nFull = 5
#         V = float(np.random.rand(1)*15.+5.)
#         towerSpeeds = np.linspace(0.,V,nFull)
#
#         prob = Problem()
#         root = prob.root = Group()
#
#         root.add('towerSpeeds', IndepVarComp('towerSpeeds', towerSpeeds), promotes=['*'])
#         root.add('dynamic_q', dynamic_q(nFull), promotes=['*'])
#
#         prob.driver = pyOptSparseDriver()
#         prob.driver.options['optimizer'] = 'SNOPT'
#
#         prob.driver.add_objective('q_dyn')
#
#         prob.driver.add_desvar('towerSpeeds')
#
#         prob.setup()
#
#         prob.run_once()
#
#         self.J = prob.check_total_derivatives(out_stream=None)
#
#         print 'FWD z'
#         print self.J[('q_dyn', 'towerSpeeds')]['J_fwd']
#         print 'FD z'
#         print self.J[('q_dyn', 'towerSpeeds')]['J_fd']
#
#
#     def test(self):
#         np.testing.assert_allclose(self.J[('q_dyn', 'towerSpeeds')]['J_fwd'], self.J[('q_dyn', 'towerSpeeds')]['J_fd'], self.rtol, self.atol)


# class testHoopStressEurocode(unittest.TestCase):
#
#
#     def setUp(self):
#
#         self.rtol = 1E-3
#         self.atol = 1E-3
#
#         nFull = 15
#
#         dlow = float(np.random.rand(1)*2.+1.)
#         dhigh = float(np.random.rand(1)*2.+3.)
#         d_full = np.linspace(dlow,dhigh,nFull)
#
#         tlow = float(np.random.rand(1)*0.01+0.006)
#         thigh = float(np.random.rand(1)*0.04+0.016)
#         t_full = np.linspace(tlow,thigh,nFull)
#
#         L_reinforced = np.ones(nFull)*30.
#         q_dyn = np.random.rand(nFull)*300.+50.
#
#         prob = Problem()
#         root = prob.root = Group()
#
#         root.add('d_full', IndepVarComp('d_full', d_full), promotes=['*'])
#         root.add('t_full', IndepVarComp('t_full', t_full), promotes=['*'])
#         root.add('q_dyn', IndepVarComp('q_dyn', q_dyn), promotes=['*'])
#         root.add('hoopStressEurocode', hoopStressEurocode(nFull), promotes=['*'])
#
#         prob.driver = pyOptSparseDriver()
#         prob.driver.options['optimizer'] = 'SNOPT'
#
#         prob.driver.add_objective('hoop_stress')
#
#         prob.driver.add_desvar('d_full')
#         prob.driver.add_desvar('t_full')
#         prob.driver.add_desvar('q_dyn')
#
#         prob.setup()
#
#         prob['L_reinforced'] = L_reinforced
#
#         prob.run_once()
#
#         self.J = prob.check_total_derivatives(out_stream=None)
#
#         print 'FWD d'
#         print self.J[('hoop_stress', 'd_full')]['J_fwd']
#         print 'FD d'
#         print self.J[('hoop_stress', 'd_full')]['J_fd']
#
#         print 'FWD t'
#         print self.J[('hoop_stress', 't_full')]['J_fwd']
#         print 'FD t'
#         print self.J[('hoop_stress', 't_full')]['J_fd']
#
#         print 'FWD q'
#         print self.J[('hoop_stress', 'q_dyn')]['J_fwd']
#         print 'FD q'
#         print self.J[('hoop_stress', 'q_dyn')]['J_fd']
#
#
#     def test(self):
#         np.testing.assert_allclose(self.J[('hoop_stress', 'd_full')]['J_fwd'], self.J[('hoop_stress', 'd_full')]['J_fd'], self.rtol, self.atol)
#         np.testing.assert_allclose(self.J[('hoop_stress', 't_full')]['J_fwd'], self.J[('hoop_stress', 't_full')]['J_fd'], self.rtol, self.atol)
#         np.testing.assert_allclose(self.J[('hoop_stress', 'q_dyn')]['J_fwd'], self.J[('hoop_stress', 'q_dyn')]['J_fd'], self.rtol, self.atol)


# class testAxial_and_shear_SHEAR(unittest.TestCase):
#
#
#     def setUp(self):
#
#         self.rtol = 1E-3
#         self.atol = 1E-3
#
#         nFull = 5
#
#         m = np.random.rand(1)*200000.+100000.
#         Fz = float(-9.81*m)
#
#         dlow = float(np.random.rand(1)*2.+1.)
#         dhigh = float(np.random.rand(1)*2.+3.)
#         d_full = np.linspace(dlow,dhigh,nFull)
#
#         tlow = float(np.random.rand(1)*0.01+0.006)
#         thigh = float(np.random.rand(1)*0.04+0.016)
#         t_full = np.linspace(tlow,thigh,nFull)
#
#         H = float(np.random.rand(1)*100.+35.)
#         z_full = np.linspace(0.,H,nFull)
#
#         Fx = float(np.random.rand(1)*1000000.+300000.)
#
#         qx = np.random.rand(nFull)*10000.
#         Mxx = -1.*float(np.random.rand(1))*1000000.-642674
#         Myy = -1.*float(np.random.rand(1))*1000000.-1642674
#
#         rho = np.ones(nFull)*8050.
#
#         mrhox = np.array([-1.13197635])
#
#         prob = Problem()
#         root = prob.root = Group()
#
#         root.add('Fx', IndepVarComp('Fx', Fx), promotes=['*'])
#         root.add('Fy', IndepVarComp('Fy', 0.), promotes=['*'])
#         root.add('Fz', IndepVarComp('Fz', Fz), promotes=['*'])
#         # root.add('Mxx', IndepVarComp('Mxx', Mxx), promotes=['*'])
#         root.add('Myy', IndepVarComp('Myy', Myy), promotes=['*'])
#         root.add('m', IndepVarComp('m', m), promotes=['*'])
#         root.add('qx', IndepVarComp('qx', qx), promotes=['*'])
#         # root.add('qy', IndepVarComp('qy', 0.), promotes=['*'])
#
#         root.add('z_full', IndepVarComp('z_full', z_full), promotes=['*'])
#         root.add('d_full', IndepVarComp('d_full', d_full), promotes=['*'])
#         root.add('t_full', IndepVarComp('t_full', t_full), promotes=['*'])
#         root.add('axial_and_shear', axial_and_shear(nFull), promotes=['*'])
#
#         prob.driver = pyOptSparseDriver()
#         prob.driver.options['optimizer'] = 'SNOPT'
#
#         prob.driver.add_objective('shear_stress')
#
#         prob.driver.add_desvar('z_full')
#         prob.driver.add_desvar('d_full')
#         prob.driver.add_desvar('t_full')
#
#         prob.driver.add_desvar('Fx')
#         prob.driver.add_desvar('Fy')
#         prob.driver.add_desvar('Fz')
#         # prob.driver.add_desvar('Mxx')
#         prob.driver.add_desvar('Myy')
#         prob.driver.add_desvar('m')
#         prob.driver.add_desvar('qx')
#         # prob.driver.add_desvar('qy')
#
#         prob.setup()
#
#         prob['rho'] = rho
#         prob['mrhox'] = mrhox
#
#         prob.run_once()
#
#         self.J = prob.check_total_derivatives(out_stream=None)
#
#         print 'FWD z'
#         print self.J[('shear_stress', 'z_full')]['J_fwd']
#         print 'FD z'
#         print self.J[('shear_stress', 'z_full')]['J_fd']
#
#         print 'FWD d'
#         print self.J[('shear_stress', 'd_full')]['J_fwd']
#         print 'FD d'
#         print self.J[('shear_stress', 'd_full')]['J_fd']
#
#         print 'FWD t'
#         print self.J[('shear_stress', 't_full')]['J_fwd']
#         print 'FD t'
#         print self.J[('shear_stress', 't_full')]['J_fd']
#
#         print 'FWD Fx'
#         print self.J[('shear_stress', 'Fx')]['J_fwd']
#         print 'FD Fx'
#         print self.J[('shear_stress', 'Fx')]['J_fd']
#
#         print 'FWD Fy'
#         print self.J[('shear_stress', 'Fy')]['J_fwd']
#         print 'FD Fy'
#         print self.J[('shear_stress', 'Fy')]['J_fd']
#
#         print 'FWD Fz'
#         print self.J[('shear_stress', 'Fz')]['J_fwd']
#         print 'FD Fz'
#         print self.J[('shear_stress', 'Fz')]['J_fd']
#
#         # print 'FWD Mxx'
#         # print self.J[('shear_stress', 'Mxx')]['J_fwd']
#         # print 'FD Mxx'
#         # print self.J[('shear_stress', 'Mxx')]['J_fd']
#
#         print 'FWD Myy'
#         print self.J[('shear_stress', 'Myy')]['J_fwd']
#         print 'FD Myy'
#         print self.J[('shear_stress', 'Myy')]['J_fd']
#
#         print 'FWD qx'
#         print self.J[('shear_stress', 'qx')]['J_fwd']
#         print 'FD qx'
#         print self.J[('shear_stress', 'qx')]['J_fd']
#
#         # print 'FWD qy'
#         # print self.J[('shear_stress', 'qy')]['J_fwd']
#         # print 'FD qy'
#         # print self.J[('shear_stress', 'qy')]['J_fd']
#
#         print 'FWD m'
#         print self.J[('shear_stress', 'm')]['J_fwd']
#         print 'FD m'
#         print self.J[('shear_stress', 'm')]['J_fd']
#
#
#     def test(self):
#         np.testing.assert_allclose(self.J[('shear_stress', 'z_full')]['J_fwd'], self.J[('shear_stress', 'z_full')]['J_fd'], self.rtol, self.atol)
#         np.testing.assert_allclose(self.J[('shear_stress', 'd_full')]['J_fwd'], self.J[('shear_stress', 'd_full')]['J_fd'], self.rtol, self.atol)
#         np.testing.assert_allclose(self.J[('shear_stress', 't_full')]['J_fwd'], self.J[('shear_stress', 't_full')]['J_fd'], self.rtol, self.atol)
#         np.testing.assert_allclose(self.J[('shear_stress', 'Fx')]['J_fwd'], self.J[('shear_stress', 'Fx')]['J_fd'], self.rtol, self.atol)
#         np.testing.assert_allclose(self.J[('shear_stress', 'Fz')]['J_fwd'], self.J[('shear_stress', 'Fz')]['J_fd'], self.rtol, self.atol)
#         # np.testing.assert_allclose(self.J[('shear_stress', 'Mxx')]['J_fwd'], self.J[('shear_stress', 'Mxx')]['J_fd'], self.rtol, self.atol)
#         np.testing.assert_allclose(self.J[('shear_stress', 'Myy')]['J_fwd'], self.J[('shear_stress', 'Myy')]['J_fd'], self.rtol, self.atol)
#         np.testing.assert_allclose(self.J[('shear_stress', 'qx')]['J_fwd'], self.J[('shear_stress', 'qx')]['J_fd'], self.rtol, self.atol)
#         np.testing.assert_allclose(self.J[('shear_stress', 'm')]['J_fwd'], self.J[('shear_stress', 'm')]['J_fd'], self.rtol, self.atol)


# class testAxial_and_shear_AXIAL(unittest.TestCase):
#
#
#     def setUp(self):
#
#         self.rtol = 1E-3
#         self.atol = 1E-3
#
#         nFull = 8
#
#         m = np.random.rand(1)*200000.+100000.
#         Fz = float(-9.81*m)
#
#         dlow = float(np.random.rand(1)*2.+1.)
#         dhigh = float(np.random.rand(1)*2.+3.)
#         d_full = np.linspace(dlow,dhigh,nFull)
#
#         tlow = float(np.random.rand(1)*0.01+0.006)
#         thigh = float(np.random.rand(1)*0.04+0.016)
#         t_full = np.linspace(tlow,thigh,nFull)
#
#         H = float(np.random.rand(1)*100.+35.)
#         z_full = np.linspace(0.,H,nFull)
#
#         Fx = float(np.random.rand(1)*1000000.+300000.)
#
#         qx = np.random.rand(nFull)*10000.
#         Mxx = -1.*float(np.random.rand(1))*1000000.-642674
#         Myy = -1.*float(np.random.rand(1))*1000000.-1642674
#
#         rho = np.ones(nFull)*8050.
#
#         mrhox = np.array([-1.13197635])
#
#         prob = Problem()
#         root = prob.root = Group()
#
#         root.add('Fx', IndepVarComp('Fx', Fx), promotes=['*'])
#         root.add('Fz', IndepVarComp('Fz', Fz), promotes=['*'])
#         root.add('Mxx', IndepVarComp('Mxx', Mxx), promotes=['*'])
#         root.add('Myy', IndepVarComp('Myy', Myy), promotes=['*'])
#         root.add('m', IndepVarComp('m', m), promotes=['*'])
#         root.add('qx', IndepVarComp('qx', qx), promotes=['*'])
#
#         root.add('z_full', IndepVarComp('z_full', z_full), promotes=['*'])
#         root.add('d_full', IndepVarComp('d_full', d_full), promotes=['*'])
#         root.add('t_full', IndepVarComp('t_full', t_full), promotes=['*'])
#         root.add('axial_and_shear', axial_and_shear(nFull), promotes=['*'])
#         root.Myy.deriv_options['step_size'] = 100.
#
#         prob.driver = pyOptSparseDriver()
#         prob.driver.options['optimizer'] = 'SNOPT'
#
#         prob.driver.add_objective('axial_stress')
#
#         prob.driver.add_desvar('z_full')
#         prob.driver.add_desvar('d_full')
#         prob.driver.add_desvar('t_full')
#
#         prob.driver.add_desvar('Fx')
#         prob.driver.add_desvar('Fz')
#         prob.driver.add_desvar('Mxx')
#         prob.driver.add_desvar('Myy')
#         prob.driver.add_desvar('m')
#         prob.driver.add_desvar('qx')
#
#         prob.setup()
#
#         prob['rho'] = rho
#         prob['mrhox'] = mrhox
#
#         prob.run_once()
#
#         self.J = prob.check_total_derivatives(out_stream=None)
#
#         print 'FWD z'
#         print self.J[('axial_stress', 'z_full')]['J_fwd']
#         print 'FD t'
#         print self.J[('axial_stress', 'z_full')]['J_fd']
#
#         print 'FWD d'
#         print self.J[('axial_stress', 'd_full')]['J_fwd']
#         print 'FD d'
#         print self.J[('axial_stress', 'd_full')]['J_fd']
#
#         print 'FWD t'
#         print self.J[('axial_stress', 't_full')]['J_fwd']
#         print 'FD t'
#         print self.J[('axial_stress', 't_full')]['J_fd']
#         #
#         print 'FWD Fx'
#         print self.J[('axial_stress', 'Fx')]['J_fwd']
#         print 'FD Fx'
#         print self.J[('axial_stress', 'Fx')]['J_fd']
#
#         print 'FWD Fz'
#         print self.J[('axial_stress', 'Fz')]['J_fwd']
#         print 'FD Fz'
#         print self.J[('axial_stress', 'Fz')]['J_fd']
#
#         #TODO THERE ARE DIFFERENT, but I think the analytic gradients are still good
#         print 'FWD Mxx'
#         print self.J[('axial_stress', 'Mxx')]['J_fwd']
#         print 'FD Mxx'
#         print self.J[('axial_stress', 'Mxx')]['J_fd']
#
#         #TODO fail but look good!
#         print 'FWD Myy'
#         print self.J[('axial_stress', 'Myy')]['J_fwd']
#         print 'FD Myy'
#         print self.J[('axial_stress', 'Myy')]['J_fd']
#
#         #TODO sometimes fail but look good!
#         print 'FWD qx'
#         print self.J[('axial_stress', 'qx')]['J_fwd']
#         print 'FD qx'
#         print self.J[('axial_stress', 'qx')]['J_fd']
#
#         #TODO fail but look good!
#         print 'FWD m'
#         print self.J[('axial_stress', 'm')]['J_fwd']
#         print 'FD m'
#         print self.J[('axial_stress', 'm')]['J_fd']
#
#
#     def test(self):
#         np.testing.assert_allclose(self.J[('axial_stress', 'z_full')]['J_fwd'], self.J[('axial_stress', 'z_full')]['J_fd'], self.rtol, self.atol)
#         np.testing.assert_allclose(self.J[('axial_stress', 'd_full')]['J_fwd'], self.J[('axial_stress', 'd_full')]['J_fd'], self.rtol, self.atol)
#         np.testing.assert_allclose(self.J[('axial_stress', 't_full')]['J_fwd'], self.J[('axial_stress', 't_full')]['J_fd'], self.rtol, self.atol)
#         np.testing.assert_allclose(self.J[('axial_stress', 'Fx')]['J_fwd'], self.J[('axial_stress', 'Fx')]['J_fd'], self.rtol, self.atol)
#         np.testing.assert_allclose(self.J[('axial_stress', 'Fz')]['J_fwd'], self.J[('axial_stress', 'Fz')]['J_fd'], self.rtol, self.atol)
#         np.testing.assert_allclose(self.J[('axial_stress', 'Mxx')]['J_fwd'], self.J[('axial_stress', 'Mxx')]['J_fd'], self.rtol, self.atol)
#         np.testing.assert_allclose(self.J[('axial_stress', 'Myy')]['J_fwd'], self.J[('axial_stress', 'Myy')]['J_fd'], self.rtol, self.atol)
#         np.testing.assert_allclose(self.J[('axial_stress', 'qx')]['J_fwd'], self.J[('axial_stress', 'qx')]['J_fd'], self.rtol, self.atol)
#         np.testing.assert_allclose(self.J[('axial_stress', 'm')]['J_fwd'], self.J[('axial_stress', 'm')]['J_fd'], self.rtol, self.atol)

# class testShellBuckling(unittest.TestCase):
#
#
#     def setUp(self):
#
#         self.rtol = 1E-3
#         self.atol = 1E-3
#
#         nFull = 5
#
#         dlow = float(np.random.rand(1)*2.+1.)
#         dhigh = float(np.random.rand(1)*2.+3.)
#         d_full = np.linspace(dlow,dhigh,nFull)
#
#         tlow = float(np.random.rand(1)*0.01+0.006)
#         thigh = float(np.random.rand(1)*0.04+0.016)
#         t_full = np.linspace(tlow,thigh,nFull)
#
#         axial_stress = -1.*np.random.rand(nFull)*1.e+08-4.25095523e+07
#         shear_stress = np.random.rand(nFull)*3.e+06+4.e+06
#         hoop_stress = -1.*np.random.rand(nFull)*3.e+05-4.e+05
#
#         L_reinforced = np.ones(nFull)*30.
#         E = 210.e9*np.ones(nFull)
#         sigma_y = 450.0e6*np.ones(nFull)
#
#
#         prob = Problem()
#         root = prob.root = Group()
#
#         root.add('d_full', IndepVarComp('d_full', d_full), promotes=['*'])
#         root.add('t_full', IndepVarComp('t_full', t_full), promotes=['*'])
#         root.add('axial_stress', IndepVarComp('axial_stress', axial_stress), promotes=['*'])
#         root.add('shear_stress', IndepVarComp('shear_stress', shear_stress), promotes=['*'])
#         root.add('hoop_stress', IndepVarComp('hoop_stress', hoop_stress), promotes=['*'])
#         root.add('shellBuckling', shellBuckling(nFull), promotes=['*'])
#
#         prob.driver = pyOptSparseDriver()
#         prob.driver.options['optimizer'] = 'SNOPT'
#
#         prob.driver.add_objective('shell_buckling')
#
#         prob.driver.add_desvar('d_full')
#         prob.driver.add_desvar('t_full')
#         prob.driver.add_desvar('axial_stress')
#         prob.driver.add_desvar('shear_stress')
#         prob.driver.add_desvar('hoop_stress')
#
#         prob.setup()
#
#         prob['L_reinforced'] = L_reinforced
#         prob['E'] = E
#         prob['sigma_y'] = sigma_y
#
#         prob.run_once()
#
#         self.J = prob.check_total_derivatives(out_stream=None)
#
#         print 'FWD d'
#         print self.J[('shell_buckling', 'd_full')]['J_fwd']
#         print 'FD d'
#         print self.J[('shell_buckling', 'd_full')]['J_fd']
#
#         print 'FWD t'
#         print self.J[('shell_buckling', 't_full')]['J_fwd']
#         print 'FD t'
#         print self.J[('shell_buckling', 't_full')]['J_fd']
#
#         print 'FWD axial'
#         print self.J[('shell_buckling', 'axial_stress')]['J_fwd']
#         print 'FD axial'
#         print self.J[('shell_buckling', 'axial_stress')]['J_fd']
#
#         print 'FWD shear'
#         print self.J[('shell_buckling', 'shear_stress')]['J_fwd']
#         print 'FD shear'
#         print self.J[('shell_buckling', 'shear_stress')]['J_fd']
#
#         print 'FWD hoop'
#         print self.J[('shell_buckling', 'hoop_stress')]['J_fwd']
#         print 'FD hoop'
#         print self.J[('shell_buckling', 'hoop_stress')]['J_fd']
#
#     def test(self):
#         np.testing.assert_allclose(self.J[('shell_buckling', 'd_full')]['J_fwd'], self.J[('shell_buckling', 'd_full')]['J_fd'], self.rtol, self.atol)
#         np.testing.assert_allclose(self.J[('shell_buckling', 't_full')]['J_fwd'], self.J[('shell_buckling', 't_full')]['J_fd'], self.rtol, self.atol)
#         np.testing.assert_allclose(self.J[('shell_buckling', 'axial_stress')]['J_fwd'], self.J[('shell_buckling', 'axial_stress')]['J_fd'], self.rtol, self.atol)
#         np.testing.assert_allclose(self.J[('shell_buckling', 'shear_stress')]['J_fwd'], self.J[('shell_buckling', 'shear_stress')]['J_fd'], self.rtol, self.atol)
#         np.testing.assert_allclose(self.J[('shell_buckling', 'hoop_stress')]['J_fwd'], self.J[('shell_buckling', 'hoop_stress')]['J_fd'], self.rtol, self.atol)


# class testAverageI(unittest.TestCase):
#
#
#     def setUp(self):
#
#         self.rtol = 1E-3
#         self.atol = 1E-3
#
#         nFull = 15
#
#         dlow = float(np.random.rand(1)*2.+1.)
#         dhigh = float(np.random.rand(1)*2.+3.)
#         d_full = np.linspace(dlow,dhigh,nFull)
#
#         tlow = float(np.random.rand(1)*0.01+0.006)
#         thigh = float(np.random.rand(1)*0.04+0.016)
#         t_full = np.linspace(tlow,thigh,nFull)
#
#         prob = Problem()
#         root = prob.root = Group()
#
#         root.add('d_full', IndepVarComp('d_full', d_full), promotes=['*'])
#         root.add('t_full', IndepVarComp('t_full', t_full), promotes=['*'])
#         root.add('averageI', averageI(nFull), promotes=['*'])
#
#         prob.driver = pyOptSparseDriver()
#         prob.driver.options['optimizer'] = 'SNOPT'
#
#         prob.driver.add_objective('I')
#
#         prob.driver.add_desvar('d_full')
#         prob.driver.add_desvar('t_full')
#
#         prob.setup()
#
#         prob.run_once()
#
#         self.J = prob.check_total_derivatives(out_stream=None)
#
#         print 'FWD d'
#         print self.J[('I', 'd_full')]['J_fwd']
#         print 'FD d'
#         print self.J[('I', 'd_full')]['J_fd']
#
#         print 'FWD t'
#         print self.J[('I', 't_full')]['J_fwd']
#         print 'FD t'
#         print self.J[('I', 't_full')]['J_fd']
#
#     def test(self):
#         np.testing.assert_allclose(self.J[('I', 'd_full')]['J_fwd'], self.J[('I', 'd_full')]['J_fd'], self.rtol, self.atol)
#         np.testing.assert_allclose(self.J[('I', 't_full')]['J_fwd'], self.J[('I', 't_full')]['J_fd'], self.rtol, self.atol)


# class testM_L(unittest.TestCase):
#
#
#     def setUp(self):
#
#         self.rtol = 1E-3
#         self.atol = 1E-3
#
#         nFull = 15
#
#         L = float(np.random.rand(1)*100.+35.)
#         mass = float(np.random.rand(1)*200000.+100000.)
#
#         prob = Problem()
#         root = prob.root = Group()
#
#         root.add('L', IndepVarComp('L', L), promotes=['*'])
#         root.add('mass', IndepVarComp('mass', mass), promotes=['*'])
#         root.add('m_L', m_L(), promotes=['*'])
#
#         prob.driver = pyOptSparseDriver()
#         prob.driver.options['optimizer'] = 'SNOPT'
#
#         prob.driver.add_objective('m_L')
#
#         prob.driver.add_desvar('L')
#         prob.driver.add_desvar('mass')
#
#         prob.setup()
#
#         prob.run_once()
#
#         self.J = prob.check_total_derivatives(out_stream=None)
#
#         print 'FWD L'
#         print self.J[('m_L', 'L')]['J_fwd']
#         print 'FD d'
#         print self.J[('m_L', 'L')]['J_fd']
#
#         print 'FWD mass'
#         print self.J[('m_L', 'mass')]['J_fwd']
#         print 'FD t'
#         print self.J[('m_L', 'mass')]['J_fd']
#
#     def test(self):
#         np.testing.assert_allclose(self.J[('m_L', 'L')]['J_fwd'], self.J[('m_L', 'L')]['J_fd'], self.rtol, self.atol)
#         np.testing.assert_allclose(self.J[('m_L', 'mass')]['J_fwd'], self.J[('m_L', 'mass')]['J_fd'], self.rtol, self.atol)


#TODO test additional gradients when done
class testFreq(unittest.TestCase):


    def setUp(self):

        self.rtol = 1E-3
        self.atol = 1E-3

        nFull = 15

        L = float(np.random.rand(1)*100.+35.)
        Mt = np.random.rand(1)*200000.+100000.
        m_L = float(np.random.rand(1)*6000.+2000.)
        I = float(np.random.rand(1)*0.25+0.875)
        E = 210.e9*np.ones(nFull)
        It = float(np.random.rand(1)*1000.+4692.)

        prob = Problem()
        root = prob.root = Group()

        root.add('L', IndepVarComp('L', L), promotes=['*'])
        root.add('m_L', IndepVarComp('m_L', m_L), promotes=['*'])
        root.add('I', IndepVarComp('I', I), promotes=['*'])
        root.add('Mt', IndepVarComp('Mt', Mt), promotes=['*'])
        root.add('It', IndepVarComp('It', It), promotes=['*'])
        root.add('freq', freq(nFull), promotes=['*'])
        # root.It.deriv_options['step_size'] = 1000.

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'SNOPT'

        prob.driver.add_objective('freq')

        prob.driver.add_desvar('L')
        prob.driver.add_desvar('m_L')
        prob.driver.add_desvar('I')
        prob.driver.add_desvar('Mt')
        prob.driver.add_desvar('It')


        prob.setup()

        prob['E'] = E
        prob.run_once()

        self.J = prob.check_total_derivatives(out_stream=None)

        # f1 = prob['freq']
        # # prob['Mt'] += 1.
        # prob['It'] += 10.
        # prob.run_once()
        # print 'freq der wrt Mt: ', (prob['freq'] - f1)/10.


        # print 'FWD L'
        # print self.J[('freq', 'L')]['J_fwd']
        # print 'FD t'
        # print self.J[('freq', 'L')]['J_fd']
        #
        # print 'FWD m_L'
        # print self.J[('freq', 'm_L')]['J_fwd']
        # print 'FD d'
        # print self.J[('freq', 'm_L')]['J_fd']
        #
        # print 'FWD I'
        # print self.J[('freq', 'I')]['J_fwd']
        # print 'FD t'
        # print self.J[('freq', 'I')]['J_fd']

        print 'FWD Mt'
        print self.J[('freq', 'Mt')]['J_fwd']
        print 'FD Mt'
        print self.J[('freq', 'Mt')]['J_fd']

        print 'FWD It'
        print self.J[('freq', 'It')]['J_fwd']
        print 'FD It'
        print self.J[('freq', 'It')]['J_fd']

    def test(self):
        # np.testing.assert_allclose(self.J[('freq', 'L')]['J_fwd'], self.J[('freq', 'L')]['J_fd'], self.rtol, self.atol)
        # np.testing.assert_allclose(self.J[('freq', 'm_L')]['J_fwd'], self.J[('freq', 'm_L')]['J_fd'], self.rtol, self.atol)
        # np.testing.assert_allclose(self.J[('freq', 'I')]['J_fwd'], self.J[('freq', 'I')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J[('freq', 'Mt')]['J_fwd'], self.J[('freq', 'Mt')]['J_fd'], self.rtol, self.atol)
        np.testing.assert_allclose(self.J[('freq', 'It')]['J_fwd'], self.J[('freq', 'It')]['J_fd'], self.rtol, self.atol)




if __name__ == "__main__":
    unittest.main()
