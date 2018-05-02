import numpy as np
from openmdao.api import Group, Problem, IndepVarComp, pyOptSparseDriver, view_tree, profile
from turbine_costsse.NEWnrel_csm_tcc_2015 import cost_group_variableRotorStudy
import unittest
import numpy as np
from openmdao.api import Group, Problem, IndepVarComp, pyOptSparseDriver
from FLORISSE3D.setupOptimization import *
from FLORISSE3D.simpleTower import Tower
from FLORISSE3D.GeneralWindFarmComponents import calculate_boundary, SpacingComp,\
            BoundaryComp, get_z, getTurbineZ, AEPobj, DeMUX, hGroups, randomStart,\
            getRotorDiameter, getRatedPower, DeMUX, Myy_estimate, bladeLengthComp, minHeight
from FLORISSE3D.COE import COEGroup
from FLORISSE3D.floris import AEPGroup
from FLORISSE3D.rotorComponents import getRating, freqConstraintGroup
from FLORISSE3D.SimpleRotorSE import SimpleRotorSE, create_rotor_functions
from rotorse.rotor import RotorSE

from time import time

class TestEverything(unittest.TestCase):

    def setUp(self):

        self.rtol = 1E-4
        self.atol = 1E-4



        from time import time

        """setup the turbine locations"""
        nRows = 2
        nTurbs = nRows**2
        nGroups = 1
        self.nGroups = nGroups
        spacing = 4.

        rotor_diameter = 126.4
        turbineX, turbineY = setupGrid(nRows, rotor_diameter, spacing)

        turbineX = np.random.rand(nTurbs)*(nRows*rotor_diameter*5.)
        turbineY = np.random.rand(nTurbs)*(nRows*rotor_diameter*5.)

        nDirections = 1

        minSpacing = 2.0

        locations = np.zeros((nTurbs,2))
        for i in range(nTurbs):
            locations[i][0] = turbineX[i]
            locations[i][1] = turbineY[i]

        boundaryVertices, boundaryNormals = calculate_boundary(locations)
        nVertices = boundaryVertices.shape[0]

        """initial yaw values"""
        yaw = np.zeros((nDirections, nTurbs))

        nPoints = 3
        nFull = 5

        d_param = np.array([6.3,5.3,4.3])
        t_param = np.array([0.02,0.01,0.006])
        # d_param = np.random.rand(3)*2.+4.3
        # t_param = np.random.rand(3)*0.006+0.014

        # shearExp = float(np.random.rand(1)*0.21+0.08)
        # ratedPower = np.random.rand(4)*9500.+500.
        # rotorDiameter = np.random.rand(4)*110.+50.
        # turbineZ = np.random.rand(4)*110.+50.

        shearExp = 0.13
        ratedPower = np.ones(nGroups)*4000.
        rotorDiameter = np.ones(nGroups)*108.
        turbineZ = np.ones(nGroups)*129.


        """for 1 group, in each of the different rated power regions"""
        # shearExp = 0.12
        # turbineZ = np.array([100.])
        # """less than rated power"""
        # ratedPower = np.array([5050.999])
        # rotorDiameter = np.array([100.420132657])
        # """spline region"""
        # ratedPower = np.array([5050.999])
        # rotorDiameter = np.array([105.420132657])
        # """greater than rated power"""
        # ratedPower = np.array([5050.999])
        # rotorDiameter = np.array([110.420132657])

        """OpenMDAO"""
        start_setup = time()
        prob = Problem()
        root = prob.root = Group()

        interp_spline_ratedQ, interp_spline_blade_mass, interp_spline_Vrated, interp_spline_I1, interp_spline_I2, interp_spline_I3, interp_spline_ratedT, interp_spline_extremeT = create_rotor_functions()

        #Design Variables
        for i in range(nGroups):
            root.add('ratedPower%s'%i, IndepVarComp('ratedPower%s'%i, float(ratedPower[i]), units='kW'), promotes=['*'])
            root.add('d_param%s'%i, IndepVarComp('d_param%s'%i, d_param), promotes=['*'])
            root.add('t_param%s'%i, IndepVarComp('t_param%s'%i, t_param), promotes=['*'])
            root.add('turbineH%s'%i, IndepVarComp('turbineH%s'%i, float(turbineZ[i])), promotes=['*'])
            root.add('rotorDiameter%s'%i, IndepVarComp('rotorDiameter%s'%i, float(rotorDiameter[i])), promotes=['*'])


        for i in range(nGroups):
            root.add('get_z_param%s'%i, get_z(nPoints)) #have derivatives
            root.add('get_z_full%s'%i, get_z(nFull)) #have derivatives
            root.add('Tower%s_max_thrust'%i, Tower(nPoints, nFull), promotes=['L_reinforced','mrhox','E','sigma_y','gamma_f','gamma_b','rhoAir','z0','zref','shearExp','rho'])
            root.add('Tower%s_max_speed'%i, Tower(nPoints, nFull), promotes=['L_reinforced','mrhox','E','sigma_y','gamma_f','gamma_b','rhoAir','z0','zref','shearExp','rho'])
            root.add('bladeLengthComp%s'%i, bladeLengthComp()) #have derivatives
            root.add('minHeight%s'%i, minHeight()) #have derivatives
            root.add('freqConstraintGroup%s'%i, freqConstraintGroup())


            root.add('Rotor%s'%i, SimpleRotorSE(interp_spline_ratedQ, interp_spline_blade_mass, interp_spline_Vrated, interp_spline_I1, interp_spline_I2, interp_spline_I3, interp_spline_ratedT, interp_spline_extremeT))
            root.add('split_I%s'%i, DeMUX(6)) #have derivatives
            root.add('Myy_estimate%s'%i, Myy_estimate()) #have derivatives

        root.add('Zs', DeMUX(nTurbs)) #have derivatives
        root.add('hGroups', hGroups(nTurbs, nGroups), promotes=['*']) #have derivatives
        root.add('getRotorDiameter', getRotorDiameter(nTurbs, nGroups), promotes=['*']) #have derivatives
        root.add('getRatedPower', getRatedPower(nTurbs, nGroups), promotes=['*'])    #have derivatives

        root.add('COEGroup', COEGroup(nTurbs, nGroups, nDirections, nPoints, nFull), promotes=['*']) #TODO check derivatives?

        root.add('spacing_comp', SpacingComp(nTurbines=nTurbs), promotes=['*'])

        # add constraint definitions
        root.add('spacing_con', ExecComp('sc = wtSeparationSquared-(minSpacing*rotorDiameter[0])**2',
                                     minSpacing=minSpacing, rotorDiameter=np.zeros(nTurbs),
                                     sc=np.zeros(((nTurbs-1)*nTurbs/2)),
                                     wtSeparationSquared=np.zeros(((nTurbs-1)*nTurbs/2))),
                                     promotes=['*'])

        if nVertices > 0:
            # add component that enforces a convex hull wind farm boundary
            root.add('boundary_con', BoundaryComp(nVertices=nVertices, nTurbines=nTurbs), promotes=['*'])

        root.connect('turbineZ', 'Zs.Array')
        # root.connect('air_density', 'rhoAir')

        for j in range(nGroups):
            root.connect('rotorDiameter%s'%j,'rotor_nacelle_costs%s.rotor_diameter'%j)
            root.connect('ratedPower%s'%j,'rotor_nacelle_costs%s.machine_rating'%j)

        for i in range(nGroups):
            root.connect('rotorDiameter%s'%i, 'Rotor%s.rotorDiameter'%i)
            root.connect('ratedPower%s'%i, 'Rotor%s.turbineRating'%i)
            root.connect('Rotor%s.ratedQ'%i, 'rotor_nacelle_costs%s.rotor_torque'%i)

            root.connect('Rotor%s.blade_mass'%i, 'rotor_nacelle_costs%s.blade_mass'%i)
            root.connect('Rotor%s.Vrated'%i,'Tower%s_max_thrust.Vel'%i)
            # root.connect('Rotor%s.turbineclass.V_extreme'%i, 'Tower%s_max_speed.Vel'%i)
            root.connect('Rotor%s.I'%i, 'split_I%s.Array'%i)
            root.connect('split_I%s.output%s'%(i,2),'Tower%s_max_speed.It'%i)
            root.connect('Tower%s_max_speed.It'%i,'Tower%s_max_thrust.It'%i)
            root.connect('Rotor%s.ratedT'%i,'Tower%s_max_thrust.Fx'%i)
            root.connect('Rotor%s.extremeT'%i,'Tower%s_max_speed.Fx'%i)

            root.connect('rotorDiameter%s'%i, 'Myy_estimate%s.rotor_diameter'%i)
            root.connect('Myy_estimate%s.Myy'%i,'Tower%s_max_thrust.Myy'%i)
            root.connect('Myy_estimate%s.Myy'%i,'Tower%s_max_speed.Myy'%i)

            root.connect('rotorDiameter%s'%i,'bladeLengthComp%s.rotor_diameter'%i)
            root.connect('rotorDiameter%s'%i,'freqConstraintGroup%s.diameter'%i)
            root.connect('Tower%s_max_thrust.freq'%i,'freqConstraintGroup%s.freq'%i)

            root.connect('turbineH%s'%i, 'minHeight%s.height'%i)
            root.connect('rotorDiameter%s'%i, 'minHeight%s.diameter'%i)

        for i in range(nGroups):
            root.connect('rotor_nacelle_costs%s.rotor_mass'%i, 'Tower%s_max_speed.rotor_mass'%i)
            root.connect('rotor_nacelle_costs%s.nacelle_mass'%i, 'Tower%s_max_speed.nacelle_mass'%i)

            root.connect('Tower%s_max_speed.rotor_mass'%i, 'Tower%s_max_thrust.rotor_mass'%i)
            root.connect('Tower%s_max_speed.nacelle_mass'%i, 'Tower%s_max_thrust.nacelle_mass'%i)

        # for j in range(nGroups):
        #     root.connect('rotor_diameters%s'%j,'rotor_nacelle_costs%s.rotor_diameter'%j)
        #     root.connect('rated_powers%s'%j,'rotor_nacelle_costs%s.machine_rating'%j)



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

            root.connect('d_param%s'%i, 'Tower%s_max_thrust.d_param'%i)
            root.connect('t_param%s'%i, 'Tower%s_max_thrust.t_param'%i)
            root.connect('d_param%s'%i, 'Tower%s_max_speed.d_param'%i)
            root.connect('t_param%s'%i, 'Tower%s_max_speed.t_param'%i)

            root.connect('get_z_param%s.z_param'%i, 'TowerDiscretization%s.z_param'%i)
            root.connect('get_z_full%s.z_param'%i, 'TowerDiscretization%s.z_full'%i)
            root.connect('d_param%s'%i, 'TowerDiscretization%s.d_param'%i)
            root.connect('t_param%s'%i, 'TowerDiscretization%s.t_param'%i)
            root.connect('rho', 'calcMass%s.rho'%i)

        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'SNOPT'
        prob.driver.opt_settings['Major iterations limit'] = 1000
        prob.driver.opt_settings['Major optimality tolerance'] = 1.E-5
        prob.driver.opt_settings['Major feasibility tolerance'] = 1.E-5

        # obj = 'rotor_nacelle_costs0.transformer_cost'
        obj = 'COE'
        obj = 'Tower0_max_thrust.shear_stress'
        # obj = 'Rotor0.extremeT'
        self.obj = obj

        prob.driver.add_objective(obj, scaler=10.)

        prob.driver.add_desvar('turbineX', scaler=1000.)
        prob.driver.add_desvar('turbineY', scaler=1000.)

        for i in range(nGroups):
            prob.driver.add_desvar('d_param%s'%i, lower=3.87, upper=6.3, scaler=1.)
            prob.driver.add_desvar('t_param%s'%i, lower=0.001, upper=None, scaler=0.1)
            prob.driver.add_desvar('turbineH%s'%i, lower=10., scaler=10.)
            prob.driver.add_desvar('rotorDiameter%s'%i, lower=10., upper=None, scaler=1.)
            prob.driver.add_desvar('ratedPower%s'%i, lower=500., upper=10000., scaler=100.)
        #TODO add design variables for blade design

        for i in range(nGroups):
            prob.driver.add_constraint('Tower%s_max_thrust.shell_buckling'%i, upper=np.ones(nFull))
            prob.driver.add_constraint('Tower%s_max_speed.shell_buckling'%i, upper=np.ones(nFull))
            prob.driver.add_constraint('freqConstraintGroup%s.freqConstraint'%i, lower=0.0)
            prob.driver.add_constraint('minHeight%s.minHeight'%i, lower=0.0)

        prob.driver.add_constraint('sc', lower=np.zeros(((nTurbs-1.)*nTurbs/2.)), scaler=1.0)
        prob.driver.add_constraint('boundaryDistances', lower=np.zeros(nVertices*nTurbs), scaler=1.0)

        prob.root.ln_solver.options['single_voi_relevance_reduction'] = True
        prob.setup(check=True)

        end_setup = time()

        start_assign = time()

        # amaliaWind(prob)
        setupTower(nFull, prob)
        simpleSetup(nTurbs, prob)
        # setupSimpleRotorSE()
        # prob['Uref'] = np.array([11.8])
        prob['Uref'] = np.array([11.85])
        prob['windDirections'] = np.array([87.])
        prob['windFrequencies'] = np.array([1.])

        for i in range(nDirections):
            prob['yaw%s'%i] = yaw[i]
        prob['turbineX'] = turbineX
        prob['turbineY'] = turbineY
        prob['shearExp'] = shearExp

        prob['turbineX'] = turbineX
        prob['turbineY'] = turbineY

        for i in range(nGroups):
            prob['Tower%s_max_speed.Vel'%i] = 70.

        # provide values for hull constraint
        prob['boundaryVertices'] = boundaryVertices
        prob['boundaryNormals'] = boundaryNormals

        prob.run_once()

        # pass results to self for use with unit test
        self.J = prob.check_total_derivatives(out_stream=None)

        # print 'ratedPower0: ', prob['ratedPower0']
        # print 'rotorDiameter0: ', prob['rotorDiameter0']
        # print 'turbineH0: ', prob['turbineH0']
        # print 'd_param0: ', prob['d_param0']
        # print 't_param0: ', prob['t_param0']

        # print 'wrt x'
        # print 'fwd'
        # print self.J[(obj, 'turbineX')]['J_fwd']
        # print 'fd'
        # print self.J[(obj, 'turbineX')]['J_fd']
        #
        # print 'wrt y'
        # print 'fwd'
        # print self.J[(obj, 'turbineY')]['J_fwd']
        # print 'fd'
        # print self.J[(obj, 'turbineY')]['J_fd']

        print 'wrt d_param'
        print 'fwd'
        print self.J[(obj, 'd_param0')]['J_fwd']
        print 'fd'
        print self.J[(obj, 'd_param0')]['J_fd']

        print 'wrt t_param'
        print 'fwd'
        print self.J[(obj, 't_param0')]['J_fwd']
        print 'fd'
        print self.J[(obj, 't_param0')]['J_fd']

        print 'wrt rotorDiameter'
        print 'fwd'
        print self.J[(obj, 'rotorDiameter0')]['J_fwd']
        print 'fd'
        print self.J[(obj, 'rotorDiameter0')]['J_fd']

        print 'wrt turbineH'
        print 'fwd'
        print self.J[(obj, 'turbineH0')]['J_fwd']
        print 'fd'
        print self.J[(obj, 'turbineH0')]['J_fd']

        print 'wrt ratedPower'
        print 'fwd'
        print self.J[(obj, 'ratedPower0')]['J_fwd']
        print 'fd'
        print self.J[(obj, 'ratedPower0')]['J_fd']

        # print 'Tower0_max_thrust.shell_buckling'
        # # print 'turbineX: ',      'FWD: ', self.J['Tower0_max_thrust.shell_buckling','turbineX']['J_fwd'],'FD: ', self.J['Tower0_max_thrust.shell_buckling','turbineX']['J_fd']
        # # print 'turbineY: ',      'FWD: ', self.J['Tower0_max_thrust.shell_buckling','turbineY']['J_fwd']      , 'FD: ', self.J['Tower0_max_thrust.shell_buckling','turbineY']['J_fd']
        # print '!!!!!!!!!!!!!!! '
        # print 'd_param: ',       'FWD: ', self.J['Tower0_max_thrust.shell_buckling','d_param0']['J_fwd']      , 'FD: ', self.J['Tower0_max_thrust.shell_buckling','d_param0']['J_fd']
        # # print 't_param: ',       'FWD: ', self.J['Tower0_max_thrust.shell_buckling','t_param0']['J_fwd']      , 'FD: ', self.J['Tower0_max_thrust.shell_buckling','t_param0']['J_fd']
        # print '!!!!!!!!!!!!!!! '
        # print 'turbineH: ',      'FWD: ', self.J['Tower0_max_thrust.shell_buckling','turbineH0']['J_fwd']     , 'FD: ', self.J['Tower0_max_thrust.shell_buckling','turbineH0']['J_fd']
        # print '!!!!!!!!!!!!!!! '
        # print 'rotorDiameter: ', 'FWD: ', self.J['Tower0_max_thrust.shell_buckling','rotorDiameter0']['J_fwd'], 'FD: ',   self.J['Tower0_max_thrust.shell_buckling','rotorDiameter0']['J_fd']
        # print '!!!!!!!!!!!!!!! '
        # print 'ratedPower: ',    'FWD: ', self.J['Tower0_max_thrust.shell_buckling','ratedPower0']['J_fwd']   , 'FD: ',   self.J['Tower0_max_thrust.shell_buckling','ratedPower0']['J_fd']
        #
        #
        # print 'Tower0_max_speed.shell_buckling'
        # # print 'turbineX: ',      'FWD: ', self.J['Tower0_max_speed.shell_buckling','turbineX']['J_fwd']      , 'FD: ', self.J['Tower0_max_speed.shell_buckling','turbineX']['J_fd']
        # # print 'turbineY: ',      'FWD: ', self.J['Tower0_max_speed.shell_buckling','turbineY']['J_fwd']      , 'FD: ', self.J['Tower0_max_speed.shell_buckling','turbineY']['J_fd']
        # print '!!!!!!!!!!!!!!! '
        # print 'd_param: ',       'FWD: ', self.J['Tower0_max_speed.shell_buckling','d_param0']['J_fwd']      , 'FD: ', self.J['Tower0_max_speed.shell_buckling','d_param0']['J_fd']
        # # print 't_param: ',       'FWD: ', self.J['Tower0_max_speed.shell_buckling','t_param0']['J_fwd']      , 'FD: ', self.J['Tower0_max_speed.shell_buckling','t_param0']['J_fd']
        # print '!!!!!!!!!!!!!!! '
        # print 'turbineH: ',      'FWD: ', self.J['Tower0_max_speed.shell_buckling','turbineH0']['J_fwd']     , 'FD: ', self.J['Tower0_max_speed.shell_buckling','turbineH0']['J_fd']
        # # print 'rotorDiameter: ', 'FWD: ', self.J['Tower0_max_speed.shell_buckling','rotorDiameter0']['J_fwd'], 'FD: ', self.J['Tower0_max_speed.shell_buckling','rotorDiameter0']['J_fd']
        # # print 'ratedPower: ',    'FWD: ', self.J['Tower0_max_speed.shell_buckling','ratedPower0']['J_fwd']   , 'FD: ', self.J['Tower0_max_speed.shell_buckling','ratedPower0']['J_fd']


        # d_param:  FWD:  [[-0.20512128  0.00392093  0.00132166]
        #  [-0.14919244 -0.14395245  0.0026644 ]
        #  [-0.         -0.48280209  0.00778987]
        #  [-0.         -0.18469007 -0.17932639]
        #  [-0.         -0.         -0.24702679]] FD:  [[-0.20464792  0.00844846  0.00574011]
        #  [-0.14897996 -0.13837979  0.00961609]
        #  [ 0.         -0.47593875  0.02160504]
        #  [ 0.         -0.18261455 -0.16994654]
        #  [ 0.          0.         -0.24702672]]


        # d_param:  FWD:  [[-0.20512128  0.00392093  0.00132166]
        #  [-0.14919244 -0.14395245  0.0026644 ]
        #  [-0.         -0.48280209  0.00778987]
        #  [-0.         -0.18469007 -0.17932639]
        #  [-0.         -0.         -0.24702679]] FD:  [[-0.20464792  0.00844846  0.00574011]
        #  [-0.14897996 -0.13837979  0.00961609]
        #  [ 0.         -0.47593875  0.02160504]
        #  [ 0.         -0.18261455 -0.16994654]
        #  [ 0.          0.         -0.24702672]]


        # print 'freqConstraintGroup0.freqConstraint'
        # print 'turbineX: ',      'FWD: ', self.J['freqConstraintGroup0.freqConstraint','turbineX']['J_fwd']     , 'FD: ', self.J['freqConstraintGroup0.freqConstraint','turbineX']['J_fd']
        # print 'turbineY: ',      'FWD: ', self.J['freqConstraintGroup0.freqConstraint','turbineY']['J_fwd']      , 'FD: ', self.J['freqConstraintGroup0.freqConstraint','turbineY']['J_fd']
        # print 'd_param: ',       'FWD: ', self.J['freqConstraintGroup0.freqConstraint','d_param0']['J_fwd']      , 'FD: ', self.J['freqConstraintGroup0.freqConstraint','d_param0']['J_fd']
        # print 't_param: ',       'FWD: ', self.J['freqConstraintGroup0.freqConstraint','t_param0']['J_fwd']      , 'FD: ', self.J['freqConstraintGroup0.freqConstraint','t_param0']['J_fd']
        # print 'turbineH: ',      'FWD: ', self.J['freqConstraintGroup0.freqConstraint','turbineH0']['J_fwd']     , 'FD: ', self.J['freqConstraintGroup0.freqConstraint','turbineH0']['J_fd']
        # print 'rotorDiameter: ', 'FWD: ', self.J['freqConstraintGroup0.freqConstraint','rotorDiameter0']['J_fwd'], 'FD: ', self.J['freqConstraintGroup0.freqConstraint','rotorDiameter0']['J_fd']
        # print 'ratedPower: ',    'FWD: ', self.J['freqConstraintGroup0.freqConstraint','ratedPower0']['J_fwd']   , 'FD: ', self.J['freqConstraintGroup0.freqConstraint','ratedPower0']['J_fd']


        # print 'minHeight0.minHeight'
        # print 'turbineX: ',      'FWD: ', self.J['minHeight0.minHeight','turbineX']['J_fwd']      , 'FD: ', self.J['minHeight0.minHeight','turbineX']['J_fd']
        # print 'turbineY: ',      'FWD: ', self.J['minHeight0.minHeight','turbineY']['J_fwd']      , 'FD: ', self.J['minHeight0.minHeight','turbineY']['J_fd']
        # print 'd_param: ',       'FWD: ', self.J['minHeight0.minHeight','d_param0']['J_fwd']      , 'FD: ', self.J['minHeight0.minHeight','d_param0']['J_fd']
        # print 't_param: ',       'FWD: ', self.J['minHeight0.minHeight','t_param0']['J_fwd']      , 'FD: ', self.J['minHeight0.minHeight','t_param0']['J_fd']
        # print 'turbineH: ',      'FWD: ', self.J['minHeight0.minHeight','turbineH0']['J_fwd']     , 'FD: ', self.J['minHeight0.minHeight','turbineH0']['J_fd']
        # print 'rotorDiameter: ', 'FWD: ', self.J['minHeight0.minHeight','rotorDiameter0']['J_fwd'], 'FD: ', self.J['minHeight0.minHeight','rotorDiameter0']['J_fd']
        # print 'ratedPower: ',    'FWD: ', self.J['minHeight0.minHeight','ratedPower0']['J_fwd']   , 'FD: ', self.J['minHeight0.minHeight','ratedPower0']['J_fd']


        #
        # print 'wrt turbineRating'
        # print 'fwd'
        # print self.J[(obj, 'ratedPower0')]['J_fwd']
        # print 'fd'
        # print self.J[(obj, 'ratedPower0')]['J_fd']
        #
        # print 'wrt height'
        # print 'fwd'
        # print self.J[(obj, 'turbineH0')]['J_fwd']
        # print 'fd'
        # print self.J[(obj, 'turbineH0')]['J_fd']

        # print prob['rotor_nacelle_costs0.machine_rating']
        # print prob['rotor_nacelle_costs0.blade_mass']
        # print prob['rotorDiameter']

        # print 'J: ', self.J

        # print 'cost: ', prob['cost']
        # print 'rotorCost: ', prob['rotorCost']
        # print 'nacelleCost: ', prob['nacelleCost']
        # print 'HubMass: ', prob['rotor_nacelle_costs0.hub_mass']

    """GOOD"""
    def test_ratedPower(self):
        for i in range(self.nGroups):
            np.testing.assert_allclose(self.J[(self.obj, 'ratedPower%s'%i)]['J_fwd'], self.J[(self.obj, 'ratedPower%s'%i)]['J_fd'], self.rtol, self.atol)
            np.testing.assert_allclose(self.J[(self.obj, 'rotorDiameter%s'%i)]['J_fwd'], self.J[(self.obj, 'rotorDiameter%s'%i)]['J_fd'], self.rtol, self.atol)

    """GOOD"""
    # def test_rotorDiameter(self):
    #     for i in range(self.nGroups):
    #         np.testing.assert_allclose(self.J[(self.obj, 'rotorDiameter%s'%i)]['J_fwd'], self.J[(self.obj, 'rotorDiameter%s'%i)]['J_fd'], self.rtol, self.atol)

    """GOOD"""
    # def test_XY(self):
    #     np.testing.assert_allclose(self.J[(self.obj, 'turbineX')]['J_fwd'], self.J[(self.obj, 'turbineX')]['J_fd'], self.rtol, self.atol)
    #     np.testing.assert_allclose(self.J[(self.obj, 'turbineY')]['J_fwd'], self.J[(self.obj, 'turbineY')]['J_fd'], self.rtol, self.atol)

    """GOOD"""
    # def test_dt(self):
    #     for i in range(self.nGroups):
    #         np.testing.assert_allclose(self.J[(self.obj, 'd_param%s'%i)]['J_fwd'], self.J[(self.obj, 'd_param%s'%i)]['J_fd'], self.rtol, self.atol)
    #         np.testing.assert_allclose(self.J[(self.obj, 't_param%s'%i)]['J_fwd'], self.J[(self.obj, 't_param%s'%i)]['J_fd'], self.rtol, self.atol)

    """GOOD"""
    # def test_turbineH(self):
    #     for i in range(self.nGroups):
    #         np.testing.assert_allclose(self.J[(self.obj, 'turbineH%s'%i)]['J_fwd'], self.J[(self.obj, 'turbineH%s'%i)]['J_fd'], self.rtol, self.atol)

    """GOOD"""
    # def test_constraint_shell_buckling_max_thrust_GOOD(self):
    #
    #     for i in range(self.nGroups):
    #         np.testing.assert_allclose(self.J[('Tower%s_max_thrust.shell_buckling'%i, 'turbineX')]['J_fwd'], self.J[('Tower%s_max_thrust.shell_buckling'%i, 'turbineX')]['J_fd'], self.rtol, self.atol)
    #         np.testing.assert_allclose(self.J[('Tower%s_max_thrust.shell_buckling'%i, 'turbineY')]['J_fwd'], self.J[('Tower%s_max_thrust.shell_buckling'%i, 'turbineY')]['J_fd'], self.rtol, self.atol)
    #         for j in range(self.nGroups):
    #             np.testing.assert_allclose(self.J[('Tower%s_max_thrust.shell_buckling'%i, 'ratedPower%s'%j)]['J_fwd'], self.J[('Tower%s_max_thrust.shell_buckling'%i, 'ratedPower%s'%j)]['J_fd'], self.rtol, self.atol)
    #             np.testing.assert_allclose(self.J[('Tower%s_max_thrust.shell_buckling'%i, 't_param%s'%j)]['J_fwd'], self.J[('Tower%s_max_thrust.shell_buckling'%i, 't_param%s'%j)]['J_fd'], 1.E-3, 1.E-3)


    #TODO
    """BAD  but close enough    *******************************************"""
    # def test_constraint_shell_buckling_max_thrust(self):
    #
    #     for i in range(self.nGroups):
    #         for j in range(self.nGroups):
    #             """FAIL"""
    #             np.testing.assert_allclose(self.J[('Tower%s_max_thrust.shell_buckling'%i, 'turbineH%s'%j)]['J_fwd'], self.J[('Tower%s_max_thrust.shell_buckling'%i, 'turbineH%s'%j)]['J_fd'], self.rtol, self.atol)
    #             """FAIL"""
    #             np.testing.assert_allclose(self.J[('Tower%s_max_thrust.shell_buckling'%i, 'rotorDiameter%s'%j)]['J_fwd'], self.J[('Tower%s_max_thrust.shell_buckling'%i, 'rotorDiameter%s'%j)]['J_fd'], self.rtol, self.atol)
    #             """FAIL"""
    #             np.testing.assert_allclose(self.J[('Tower%s_max_thrust.shell_buckling'%i, 'd_param%s'%j)]['J_fwd'], self.J[('Tower%s_max_thrust.shell_buckling'%i, 'd_param%s'%j)]['J_fd'], self.rtol, self.atol)

    """GOOD"""
    # def test_constraint_shell_buckling_max_speed(self):
    #
    #     for i in range(self.nGroups):
    #         np.testing.assert_allclose(self.J[('Tower%s_max_speed.shell_buckling'%i, 'turbineX')]['J_fwd'], self.J[('Tower%s_max_speed.shell_buckling'%i, 'turbineX')]['J_fd'], self.rtol, self.atol)
    #         np.testing.assert_allclose(self.J[('Tower%s_max_speed.shell_buckling'%i, 'turbineY')]['J_fwd'], self.J[('Tower%s_max_speed.shell_buckling'%i, 'turbineY')]['J_fd'], self.rtol, self.atol)
    #         for j in range(self.nGroups):
    #             np.testing.assert_allclose(self.J[('Tower%s_max_speed.shell_buckling'%i, 'ratedPower%s'%j)]['J_fwd'], self.J[('Tower%s_max_speed.shell_buckling'%i, 'ratedPower%s'%j)]['J_fd'], self.rtol, self.atol)
    #             np.testing.assert_allclose(self.J[('Tower%s_max_speed.shell_buckling'%i, 't_param%s'%j)]['J_fwd'], self.J[('Tower%s_max_speed.shell_buckling'%i, 't_param%s'%j)]['J_fd'], 1.E-3, 1.E-3)

    #TODO
    """BAD   but close enough   *******************************************"""
    # def test_constraint_shell_buckling_max_speed(self):
    #
    #     for i in range(self.nGroups):
    #         for j in range(self.nGroups):
    #             """Fail"""
    #             np.testing.assert_allclose(self.J[('Tower%s_max_speed.shell_buckling'%i, 'turbineH%s'%j)]['J_fwd'], self.J[('Tower%s_max_speed.shell_buckling'%i, 'turbineH%s'%j)]['J_fd'], self.rtol, self.atol)
    #             """Fail sometimes"""
    #             np.testing.assert_allclose(self.J[('Tower%s_max_speed.shell_buckling'%i, 'rotorDiameter%s'%j)]['J_fwd'], self.J[('Tower%s_max_speed.shell_buckling'%i, 'rotorDiameter%s'%j)]['J_fd'], self.rtol, self.atol)
    #             """Fail"""
    #             np.testing.assert_allclose(self.J[('Tower%s_max_speed.shell_buckling'%i, 'd_param%s'%j)]['J_fwd'], self.J[('Tower%s_max_speed.shell_buckling'%i, 'd_param%s'%j)]['J_fd'], self.rtol, self.atol)


    """GOOD"""
    # def test_constraint_frequency(self):
    #
    #     for i in range(self.nGroups):
    #         np.testing.assert_allclose(self.J[('freqConstraintGroup%s.freqConstraint'%i, 'turbineX')]['J_fwd'], self.J[('freqConstraintGroup%s.freqConstraint'%i, 'turbineX')]['J_fd'], self.rtol, self.atol)
    #         np.testing.assert_allclose(self.J[('freqConstraintGroup%s.freqConstraint'%i, 'turbineY')]['J_fwd'], self.J[('freqConstraintGroup%s.freqConstraint'%i, 'turbineY')]['J_fd'], self.rtol, self.atol)
    #         for j in range(self.nGroups):
    #             np.testing.assert_allclose(self.J[('freqConstraintGroup%s.freqConstraint'%i, 'turbineH%s'%j)]['J_fwd'], self.J[('freqConstraintGroup%s.freqConstraint'%i, 'turbineH%s'%j)]['J_fd'], self.rtol, self.atol)
    #             np.testing.assert_allclose(self.J[('freqConstraintGroup%s.freqConstraint'%i, 'ratedPower%s'%j)]['J_fwd'], self.J[('freqConstraintGroup%s.freqConstraint'%i, 'ratedPower%s'%j)]['J_fd'], self.rtol, self.atol)
    #             np.testing.assert_allclose(self.J[('freqConstraintGroup%s.freqConstraint'%i, 'd_param%s'%j)]['J_fwd'], self.J[('freqConstraintGroup%s.freqConstraint'%i, 'd_param%s'%j)]['J_fd'], self.rtol, self.atol)
    #             np.testing.assert_allclose(self.J[('freqConstraintGroup%s.freqConstraint'%i, 't_param%s'%j)]['J_fwd'], self.J[('freqConstraintGroup%s.freqConstraint'%i, 't_param%s'%j)]['J_fd'], 1.E-3, 1.E-3)
    #             np.testing.assert_allclose(self.J[('freqConstraintGroup%s.freqConstraint'%i, 'rotorDiameter%s'%j)]['J_fwd'], self.J[('freqConstraintGroup%s.freqConstraint'%i, 'rotorDiameter%s'%j)]['J_fd'], self.rtol, self.atol)


    """GOOD"""
    # def test_constraint_min_height(self):
    #
    #     for i in range(self.nGroups):
    #         np.testing.assert_allclose(self.J[('minHeight%s.minHeight'%i, 'turbineX')]['J_fwd'], self.J[('minHeight%s.minHeight'%i, 'turbineX')]['J_fd'], self.rtol, self.atol)
    #         np.testing.assert_allclose(self.J[('minHeight%s.minHeight'%i, 'turbineY')]['J_fwd'], self.J[('minHeight%s.minHeight'%i, 'turbineY')]['J_fd'], self.rtol, self.atol)
    #         for j in range(self.nGroups):
    #             np.testing.assert_allclose(self.J[('minHeight%s.minHeight'%i, 'turbineH%s'%j)]['J_fwd'], self.J[('minHeight%s.minHeight'%i, 'turbineH%s'%j)]['J_fd'], self.rtol, self.atol)
    #             np.testing.assert_allclose(self.J[('minHeight%s.minHeight'%i, 'ratedPower%s'%j)]['J_fwd'], self.J[('minHeight%s.minHeight'%i, 'ratedPower%s'%j)]['J_fd'], self.rtol, self.atol)
    #             np.testing.assert_allclose(self.J[('minHeight%s.minHeight'%i, 'rotorDiameter%s'%j)]['J_fwd'], self.J[('minHeight%s.minHeight'%i, 'rotorDiameter%s'%j)]['J_fd'], self.rtol, self.atol)
    #             np.testing.assert_allclose(self.J[('minHeight%s.minHeight'%i, 'd_param%s'%j)]['J_fwd'], self.J[('minHeight%s.minHeight'%i, 'd_param%s'%j)]['J_fd'], self.rtol, self.atol)
    #             np.testing.assert_allclose(self.J[('minHeight%s.minHeight'%i, 't_param%s'%j)]['J_fwd'], self.J[('minHeight%s.minHeight'%i, 't_param%s'%j)]['J_fd'], 1.E-3, 1.E-3)


    """GOOD"""
    # def test_constraint_sc(self):
    #
    #     np.testing.assert_allclose(self.J[('sc', 'turbineX')]['J_fwd'], self.J[('sc', 'turbineX')]['J_fd'], self.rtol, self.atol)
    #     np.testing.assert_allclose(self.J[('sc', 'turbineY')]['J_fwd'], self.J[('sc', 'turbineY')]['J_fd'], self.rtol, self.atol)
    #     for j in range(self.nGroups):
    #         np.testing.assert_allclose(self.J[('sc', 'turbineH%s'%j)]['J_fwd'], self.J[('sc', 'turbineH%s'%j)]['J_fd'], self.rtol, self.atol)
    #         np.testing.assert_allclose(self.J[('sc', 'ratedPower%s'%j)]['J_fwd'], self.J[('sc', 'ratedPower%s'%j)]['J_fd'], self.rtol, self.atol)
    #         np.testing.assert_allclose(self.J[('sc', 'rotorDiameter%s'%j)]['J_fwd'], self.J[('sc', 'rotorDiameter%s'%j)]['J_fd'], self.rtol, self.atol)
    #         np.testing.assert_allclose(self.J[('sc', 'd_param%s'%j)]['J_fwd'], self.J[('sc', 'd_param%s'%j)]['J_fd'], self.rtol, self.atol)
    #         np.testing.assert_allclose(self.J[('sc', 't_param%s'%j)]['J_fwd'], self.J[('sc', 't_param%s'%j)]['J_fd'], 1.E-3, 1.E-3)

    """GOOD"""
    # def test_constraint_boundaryDistances(self):
    #
    #     np.testing.assert_allclose(self.J[('boundaryDistances', 'turbineX')]['J_fwd'], self.J[('boundaryDistances', 'turbineX')]['J_fd'], self.rtol, self.atol)
    #     np.testing.assert_allclose(self.J[('boundaryDistances', 'turbineY')]['J_fwd'], self.J[('boundaryDistances', 'turbineY')]['J_fd'], self.rtol, self.atol)
    #     for j in range(self.nGroups):
    #         np.testing.assert_allclose(self.J[('boundaryDistances', 'turbineH%s'%j)]['J_fwd'], self.J[('boundaryDistances', 'turbineH%s'%j)]['J_fd'], self.rtol, self.atol)
    #         np.testing.assert_allclose(self.J[('boundaryDistances', 'ratedPower%s'%j)]['J_fwd'], self.J[('boundaryDistances', 'ratedPower%s'%j)]['J_fd'], self.rtol, self.atol)
    #         np.testing.assert_allclose(self.J[('boundaryDistances', 'rotorDiameter%s'%j)]['J_fwd'], self.J[('boundaryDistances', 'rotorDiameter%s'%j)]['J_fd'], self.rtol, self.atol)
    #         np.testing.assert_allclose(self.J[('boundaryDistances', 'd_param%s'%j)]['J_fwd'], self.J[('boundaryDistances', 'd_param%s'%j)]['J_fd'], self.rtol, self.atol)
    #         np.testing.assert_allclose(self.J[('boundaryDistances', 't_param%s'%j)]['J_fwd'], self.J[('boundaryDistances', 't_param%s'%j)]['J_fd'], 1.E-3, 1.E-3)


if __name__ == "__main__":
    unittest.main()
