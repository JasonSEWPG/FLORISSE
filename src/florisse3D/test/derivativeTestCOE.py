import unittest
from openmdao.api import pyOptSparseDriver, ExecComp, IndepVarComp, Problem

from FLORISSE3D.COE import *
from FLORISSE3D.floris import AEPGroup
from FLORISSE3D.GeneralWindFarmComponents import calculate_boundary, SpacingComp,\
            BoundaryComp, get_z, get_z_DEL, getTurbineZ, AEPobj, speedFreq, actualSpeeds
from towerse.tower import TowerSE
from setupOptimization import *



import cPickle as pickle

# Good and Fast
# class TestCost(unittest.TestCase):
#
#
#     def setUp(self):
#         nTurbines = 5
#         mass1 = 50000.
#         mass2 = 100000.
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
#         print self.J[('cost', 'mass2')]['J_fwd']
#         print 'Finite Difference'
#         print self.J[('cost', 'mass2')]['J_fd']
#
#     def test_mass(self):
#         np.testing.assert_allclose(self.J[('cost', 'mass1')]['J_fwd'], self.J[('cost', 'mass1')]['J_fd'], 1e-3, 1e-3)
#         np.testing.assert_allclose(self.J[('cost', 'mass2')]['J_fwd'], self.J[('cost', 'mass2')]['J_fd'], 1e-3, 1e-3)
#

class TotalDerivTestsCOE(unittest.TestCase):

    def setUp(self):
        use_rotor_components = True

        if use_rotor_components:
            NREL5MWCPCT = pickle.load(open('../../../doc/tune/NREL5MWCPCT_smooth_dict.p'))
            # print(NREL5MWCPCT)
            # NREL5MWCPCT = pickle.Unpickler(open('./input_files/NREL5MWCPCT.p')).load()
            datasize = NREL5MWCPCT['CP'].size
        else:
            datasize = 0

        nTurbines = 4
        nTurbs = nTurbines
        self.tol = 1E-6

        # np.random.seed(seed=10)

        turbineX = np.random.rand(nTurbines)*500.
        turbineY = np.random.rand(nTurbines)*500.
        # turbineH1 = np.random.rand(1)*50.+75.
        # turbineH2 = np.random.rand(1)*50.+75.
        turbineH1 = 73.
        turbineH2 = 122.
        H1_H2 = np.array([0,1,0,1])

        minSpacing = 2.

        # generate boundary constraint
        locations = np.zeros((len(turbineX),2))
        for i in range(len(turbineX)):
            locations[i][0] = turbineX[i]
            locations[i][1] = turbineY[i]
        print locations
        boundaryVertices, boundaryNormals = calculate_boundary(locations)
        nVertices = boundaryVertices.shape[0]


        # initialize input variable arrays
        rotorDiameter = np.ones(nTurbines)*np.random.random()*150.
        axialInduction = np.ones(nTurbines)*np.random.random()*(1./3.)
        Ct = np.ones(nTurbines)*np.random.random()
        Cp = np.ones(nTurbines)*np.random.random()
        generatorEfficiency = np.ones(nTurbines)*np.random.random()
        yaw = np.random.rand(nTurbines)*60. - 30.

        # Define flow properties
        nDirections = 50
        windSpeeds = np.random.rand(nDirections)*20        # m/s
        air_density = 1.1716    # kg/m^3
        windDirections = np.random.rand(nDirections)*360.0
        windFrequencies = np.random.rand(nDirections)
        tot = np.sum(windFrequencies)
        windFrequencies = windFrequencies/tot

        nIntegrationPoints = 1 #Number of points in wind effective wind speed integral

        # --- geometry ----
        d_paramH1 = np.array([6.,5.,4.])
        t_paramH1 = np.array([.2,.1,.05])
        d_paramH2 = np.array([5.,4.,3.])
        t_paramH2 = np.array([.2,.1,.04])
        n = 15

        L_reinforced, Toweryaw, E, G, rho, sigma_y, kidx, kx, ky, kz, ktx, kty, ktz, nK, \
                    midx, m, mIxx, mIyy, mIzz, mIxy, mIxz, mIyz, mrhox, mrhoy, mrhoz, nMass, \
                    addGravityLoadForExtraMass, wind_zref, wind_z0, shearExp, wind_Uref1, \
                    plidx1, Fx1, Fy1, Fz1, Mxx1, Myy1, Mzz1, nPL, wind_Uref2, \
                    plidx2, Fx2, Fy2, Fz2, Mxx2, Myy2, Mzz2, gamma_f, gamma_m, gamma_n, \
                    gamma_b, z_DEL, M_DEL, nDEL, gamma_fatigue, life, m_SN, min_d_to_t, min_taper \
                    = setupTower(n)

        nPoints = len(d_paramH1)
        nFull = n
        wind = 'PowerWind'

        shearExp = 0.15

        # set up problem

        prob = Problem()
        root = prob.root = Group()

        # root.deriv_options['type'] = 'fd'
        # root.deriv_options['form'] = 'central'
        # root.deriv_options['step_size'] = 1.E-4
        # root.deriv_options['step_type'] = 'relative'


        root.add('turbineH1', IndepVarComp('turbineH1', turbineH1), promotes=['*'])
        root.add('turbineH2', IndepVarComp('turbineH2', turbineH2), promotes=['*'])
        root.add('d_paramH1', IndepVarComp('d_paramH1', d_paramH1), promotes=['*'])
        root.add('t_paramH1', IndepVarComp('t_paramH1', t_paramH1), promotes=['*'])
        root.add('d_paramH2', IndepVarComp('d_paramH2', d_paramH2), promotes=['*'])
        root.add('t_paramH2', IndepVarComp('t_paramH2', t_paramH2), promotes=['*'])
        #This component takes turbineH1, turbineH2, and H1_H2 and gives back an array
        #of turbineZ
        root.add('getTurbineZ', getTurbineZ(nTurbs), promotes=['*'])
        #These components adjust the parameterized z locations for TowerSE calculations
        #with respect to turbineZ
        root.add('get_z_paramH1', get_z(nPoints))
        root.add('get_z_paramH2', get_z(nPoints))
        root.add('get_z_fullH1', get_z(n))
        root.add('get_z_fullH2', get_z(n))
        root.add('AEPGroup', AEPGroup(nTurbs, nDirections=nDirections,
                    use_rotor_components=use_rotor_components, datasize=datasize, differentiable=True,
                    optimizingLayout=False, nSamples=0), promotes=['*'])
        root.add('COEGroup', COEGroup(nTurbs), promotes=['*'])
        root.add('maxAEP', AEPobj(), promotes=['*'])

        #For Constraints
        root.add('TowerH1', TowerSE(nPoints, nFull, nK, nMass, nPL, nDEL, wind=wind), promotes=['L_reinforced',
                            'E','G','sigma_y','kidx','kx','ky','kz','ktx','kty',
                            'ktz','midx','m','mIxx','mIyy','mIzz','mIxy','mIxz','mIyz',
                            'mrhox','mrhoy','mrhoz','addGravityLoadForExtraMass','g',
                            'gamma_f','gamma_m','gamma_n','gamma_b','life','m_SN',
                            'gc.min_d_to_t','gc.min_taper','M_DEL','gamma_fatigue'])
        root.add('TowerH2', TowerSE(nPoints, nFull, nK, nMass, nPL, nDEL, wind=wind), promotes=['L_reinforced',
                            'E','G','sigma_y','kidx','kx','ky','kz','ktx','kty',
                            'ktz','midx','m','mIxx','mIyy','mIzz','mIxy','mIxz','mIyz',
                            'mrhox','mrhoy','mrhoz','addGravityLoadForExtraMass','g',
                            'gamma_f','gamma_m','gamma_n','gamma_b','life','m_SN',
                            'gc.min_d_to_t','gc.min_taper','M_DEL','gamma_fatigue'])
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

        root.connect('turbineH1', 'get_z_paramH1.turbineZ')
        root.connect('turbineH2', 'get_z_paramH2.turbineZ')
        root.connect('turbineH1', 'get_z_fullH1.turbineZ')
        root.connect('turbineH2', 'get_z_fullH2.turbineZ')
        root.connect('get_z_paramH1.z_param', 'TowerH1.z_param')
        root.connect('get_z_fullH1.z_param', 'TowerH1.z_full')
        root.connect('get_z_paramH2.z_param', 'TowerH2.z_param')
        root.connect('get_z_fullH2.z_param', 'TowerH2.z_full')
        root.connect('TowerH1.tower1.mass', 'mass1')
        root.connect('TowerH2.tower1.mass', 'mass2')
        root.connect('d_paramH1', 'TowerH1.d_param')
        root.connect('d_paramH2', 'TowerH2.d_param')
        root.connect('t_paramH1', 'TowerH1.t_param')
        root.connect('t_paramH2', 'TowerH2.t_param')

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'SNOPT'
        prob.driver.opt_settings['Function precision'] = 1E-8

        prob.driver.add_objective('COE', scaler=1.0E-1)

        # # --- Design Variables ---
        prob.driver.add_desvar('turbineH1', lower=72., upper=160., scaler=1.0E-1)
        prob.driver.add_desvar('turbineH2', lower=72., upper=160., scaler=1.0E-1)
        prob.driver.add_desvar('turbineX', lower=np.ones(nTurbs)*min(turbineX), upper=np.ones(nTurbs)*max(turbineX), scaler=1.0E-3)
        prob.driver.add_desvar('turbineY', lower=np.ones(nTurbs)*min(turbineY), upper=np.ones(nTurbs)*max(turbineY), scaler=1.0E-3)
        prob.driver.add_desvar('d_paramH1', lower=np.array([1.0, 1.0, d_paramH1[nPoints-1]]), upper=np.ones(nPoints)*6.3, scaler=1.0E-1)
        prob.driver.add_desvar('t_paramH1', lower=np.ones(nPoints)*.001, upper=np.ones(nPoints)*0.05, scaler=1.0E1)
        prob.driver.add_desvar('d_paramH2', lower=np.array([1.0, 1.0, d_paramH1[nPoints-1]]), upper=np.ones(nPoints)*6.3, scaler=1.0E-1)
        prob.driver.add_desvar('t_paramH2', lower=np.ones(nPoints)*.001, upper=np.ones(nPoints)*0.05, scaler=1.0E1)

        # --- Constraints ---
        #TowerH1 structure
        prob.driver.add_constraint('TowerH1.tower1.stress', upper=1.0)
        prob.driver.add_constraint('TowerH1.tower2.stress', upper=1.0)
        prob.driver.add_constraint('TowerH1.tower1.global_buckling', upper=1.0)
        prob.driver.add_constraint('TowerH1.tower2.global_buckling', upper=1.0)
        prob.driver.add_constraint('TowerH1.tower1.shell_buckling', upper=1.0)
        prob.driver.add_constraint('TowerH1.tower2.shell_buckling', upper=1.0)
        prob.driver.add_constraint('TowerH1.tower1.damage', upper=1.0)
        prob.driver.add_constraint('TowerH1.gc.weldability', upper=0.0)
        prob.driver.add_constraint('TowerH1.gc.manufacturability', upper=0.0)
        freq1p = 0.2  # 1P freq in Hz
        prob.driver.add_constraint('TowerH1.tower1.f1', lower=1.1*freq1p)

        #TowerH2 structure
        prob.driver.add_constraint('TowerH2.tower1.stress', upper=1.0)
        prob.driver.add_constraint('TowerH2.tower2.stress', upper=1.0)
        prob.driver.add_constraint('TowerH2.tower1.global_buckling', upper=1.0)
        prob.driver.add_constraint('TowerH2.tower2.global_buckling', upper=1.0)
        prob.driver.add_constraint('TowerH2.tower1.shell_buckling', upper=1.0)
        prob.driver.add_constraint('TowerH2.tower2.shell_buckling', upper=1.0)
        prob.driver.add_constraint('TowerH2.gc.weldability', upper=0.0)
        prob.driver.add_constraint('TowerH2.gc.manufacturability', upper=0.0)
        freq1p = 0.2  # 1P freq in Hz
        prob.driver.add_constraint('TowerH2.tower1.f1', lower=1.1*freq1p)

        # boundary constraint (convex hull)
        prob.driver.add_constraint('boundaryDistances', lower=np.zeros(nVertices*nTurbs), scaler=1.0)
        # spacing constraint
        prob.driver.add_constraint('sc', lower=np.zeros(((nTurbs-1.)*nTurbs/2.)), scaler=1.0/62.4)

        prob.root.ln_solver.options['single_voi_relevance_reduction'] = True

        # initialize problem
        prob.setup()

        if wind == "PowerWind":
            prob['TowerH1.wind1.shearExp'] = shearExp
            prob['TowerH1.wind2.shearExp'] = shearExp
            prob['TowerH2.wind1.shearExp'] = shearExp
            prob['TowerH2.wind2.shearExp'] = shearExp
            prob['shearExp'] = shearExp
        prob['turbineH1'] = turbineH1
        prob['turbineH2'] = turbineH2
        prob['H1_H2'] = H1_H2
        # prob['diameter'] = rotor_diameter

        prob['turbineX'] = turbineX
        prob['turbineY'] = turbineY
        prob['yaw0'] = yaw
        prob['ratedPower'] = np.ones_like(turbineX)*5000 # in kw

        prob['boundaryVertices'] = boundaryVertices
        prob['boundaryNormals'] = boundaryNormals

        # assign values to constant inputs (not design variables)
        prob['nIntegrationPoints'] = nIntegrationPoints
        prob['rotorDiameter'] = rotorDiameter
        prob['axialInduction'] = axialInduction
        prob['generatorEfficiency'] = generatorEfficiency
        prob['air_density'] = air_density
        prob['windDirections'] = np.array([windDirections])
        prob['windFrequencies'] = np.array([windFrequencies])
        prob['Uref'] = windSpeeds
        if use_rotor_components == True:
            prob['gen_params:windSpeedToCPCT_CP'] = NREL5MWCPCT['CP']
            prob['gen_params:windSpeedToCPCT_CT'] = NREL5MWCPCT['CT']
            prob['gen_params:windSpeedToCPCT_wind_speed'] = NREL5MWCPCT['wind_speed']
        else:
            prob['Ct_in'] = Ct
            prob['Cp_in'] = Cp
        prob['floris_params:cos_spread'] = 1E12 # turns off cosine spread (just needs to be very large)
        prob['zref'] = wind_zref
        prob['z0'] = wind_z0

        """tower structural properties"""
        # --- geometry ----
        # prob['d_param'] = d_param
        # prob['t_param'] = t_param

        prob['L_reinforced'] = L_reinforced
        prob['TowerH1.yaw'] = Toweryaw
        prob['TowerH2.yaw'] = Toweryaw

        # --- material props ---
        prob['E'] = E
        prob['G'] = G
        prob['TowerH1.tower1.rho'] = rho
        prob['TowerH2.tower1.rho'] = rho
        prob['sigma_y'] = sigma_y

        # --- spring reaction data.  Use float('inf') for rigid constraints. ---
        prob['kidx'] = kidx
        prob['kx'] = kx
        prob['ky'] = ky
        prob['kz'] = kz
        prob['ktx'] = ktx
        prob['kty'] = kty
        prob['ktz'] = ktz

        # --- extra mass ----
        prob['midx'] = midx
        prob['m'] = m
        prob['mIxx'] = mIxx
        prob['mIyy'] = mIyy
        prob['mIzz'] = mIzz
        prob['mIxy'] = mIxy
        prob['mIxz'] = mIxz
        prob['mIyz'] = mIyz
        prob['mrhox'] = mrhox
        prob['mrhoy'] = mrhoy
        prob['mrhoz'] = mrhoz
        prob['addGravityLoadForExtraMass'] = addGravityLoadForExtraMass
        # -----------

        # --- wind ---
        prob['TowerH1.zref'] = wind_zref
        prob['TowerH2.zref'] = wind_zref
        prob['TowerH1.z0'] = wind_z0
        prob['TowerH2.z0'] = wind_z0
        # ---------------

        # # --- loading case 1: max Thrust ---
        prob['TowerH1.wind1.Uref'] = wind_Uref1
        prob['TowerH1.tower1.plidx'] = plidx1
        prob['TowerH1.tower1.Fx'] = Fx1
        prob['TowerH1.tower1.Fy'] = Fy1
        prob['TowerH1.tower1.Fz'] = Fz1
        prob['TowerH1.tower1.Mxx'] = Mxx1
        prob['TowerH1.tower1.Myy'] = Myy1
        prob['TowerH1.tower1.Mzz'] = Mzz1
        #
        prob['TowerH2.wind1.Uref'] = wind_Uref1
        prob['TowerH2.tower1.plidx'] = plidx1
        prob['TowerH2.tower1.Fx'] = Fx1
        prob['TowerH2.tower1.Fy'] = Fy1
        prob['TowerH2.tower1.Fz'] = Fz1
        prob['TowerH2.tower1.Mxx'] = Mxx1
        prob['TowerH2.tower1.Myy'] = Myy1
        prob['TowerH2.tower1.Mzz'] = Mzz1
        # # ---------------

        # # --- loading case 2: max Wind Speed ---
        prob['TowerH1.wind2.Uref'] = wind_Uref2
        prob['TowerH1.tower2.plidx'] = plidx2
        prob['TowerH1.tower2.Fx'] = Fx2
        prob['TowerH1.tower2.Fy'] = Fy2
        prob['TowerH1.tower2.Fz'] = Fz2
        prob['TowerH1.tower2.Mxx'] = Mxx2
        prob['TowerH1.tower2.Myy'] = Myy2
        prob['TowerH1.tower2.Mzz'] = Mzz2
        #
        prob['TowerH2.wind2.Uref'] = wind_Uref2
        prob['TowerH2.tower2.plidx'] = plidx2
        prob['TowerH2.tower2.Fx'] = Fx2
        prob['TowerH2.tower2.Fy'] = Fy2
        prob['TowerH2.tower2.Fz'] = Fz2
        prob['TowerH2.tower2.Mxx'] = Mxx2
        prob['TowerH2.tower2.Myy'] = Myy2
        prob['TowerH2.tower2.Mzz'] = Mzz2
        # # # ---------------

        # --- safety factors ---
        prob['gamma_f'] = gamma_f
        prob['gamma_m'] = gamma_m
        prob['gamma_n'] = gamma_n
        prob['gamma_b'] = gamma_b
        # ---------------

        # --- fatigue ---
        prob['gamma_fatigue'] = gamma_fatigue
        prob['life'] = life
        prob['m_SN'] = m_SN
        # ---------------

        # --- constraints ---
        prob['gc.min_d_to_t'] = min_d_to_t
        prob['gc.min_taper'] = min_taper
        # ---------------

        # run problem
        prob.run_once()

        # pass results to self for use with unit test
        self.J = prob.check_total_derivatives(out_stream=None)
        self.nDirections = nDirections

        print "**************************************"
        print 'Finite Difference'
        print self.J['COE','d_paramH1']['J_fd']
        print 'Analytic'
        print self.J['COE','d_paramH1']['J_fwd']

    def testObj(self):

        np.testing.assert_allclose(self.J[('COE', 'd_paramH1')]['J_fwd'], self.J[('COE', 'd_paramH1')]['J_fd'], self.tol)

    # def testCon(self):
    #
    #     np.testing.assert_allclose(self.J[('sc', 'turbineX')]['rel error'], self.J[('sc', 'turbineX')]['rel error'], self.rtol, self.atol)
    #     np.testing.assert_allclose(self.J[('sc', 'turbineY')]['rel error'], self.J[('sc', 'turbineY')]['rel error'], self.rtol, self.atol)
    #     """for dir in np.arange(0, self.nDirections):
    #         np.testing.assert_allclose(self.J[('sc', 'yaw%i' % dir)]['rel error'], self.J[('sc', 'yaw%i' % dir)]['rel error'], self.rtol, self.atol)"""

if __name__ == "__main__":
    unittest.main()
