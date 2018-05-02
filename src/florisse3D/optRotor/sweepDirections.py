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
import matplotlib.pyplot as plt

from time import time

if __name__ == '__main__':
    """setup the turbine locations"""
    nGroups = 1

    rotor_diameter = 126.4

    turbineX, turbineY = amaliaLayout()

    # """opt with other stuffs"""
    # turbineX = np.array([   90.298508 ,155.262800  ,233.738977  , 70.286379,\
    #  165.067084,182.309432, 297.470351,316.282984    , 43.224253,\
    #      58.566446 ,300.238050  ,328.665026  ,327.388458  ,343.663390  , 20.683050  ,\
    #     54.352222  ,309.512770  ,326.775251  ,345.388818  ,362.583109  ,379.794647  ,\
    #     37.516991  ,54.771210  ,        72.004292  ,       344.973348  ,
    #    362.041246  ,       379.032313  ,        19.317168  ,        35.980271  ,
    #     52.360937  ,       326.823282  ,       343.898114  ,       361.067055  ,
    #    377.802154  ,         0.058716  ,         5.442075  ,        22.842260  ,
    #     40.046833  ,       340.498921  ,       357.678520  ,       374.939716  ,
    #     16.764158  ,       252.837519  ,       270.156635  ,       287.513968  ,
    #    304.779563  ,       321.974699  ,       339.028919  ,         1.987179  ,
    #      1.505881  ,        20.103199  ,       238.909061  ,       256.288963  ,
    #    273.539201  ,        51.499146  ,        68.151868  ,       192.459028  ,
    #    209.538851  ,       151.134100  ,       168.369400  ])*10.
    # turbineY = np.array([14.760692    ,    -0.000000    ,    12.306316    ,    54.117877    ,
    # 24.388189    ,    36.718311    ,    48.771174    ,    62.090479    ,
    # 07.340059    ,    77.167080    ,        75.942939,        88.380191,
    #    116.940994,       103.359497,       156.370165,       125.381645,
    #    128.333550,       140.635471,       152.950759,       165.347810,
    #    177.720908,       169.252282,       181.565790,       193.908862,
    #    189.449457,       202.019954,       214.694115,       242.423888,
    #    229.321540,       215.867755,       226.823234,       239.384310,
    #    251.816445,       264.826710,       297.702193,       258.449256,
    #    270.555623,       282.938403,       278.952954,       291.370358,
    #    303.674081,       310.750516,       316.078886,       328.300945,
    #    340.468670,       352.766219,       365.162098,       377.751144,
    #    389.786310,       366.804321,       378.779933,       395.978948,
    #    408.114415,       420.433498,       451.674383,       438.558842,
    #    449.590872,       462.145169,       476.637027,       488.977000])*10.
    #
    # """opt XY"""
    # turbineX = np.array([  1.20982469e+03 ,  1.80498157e+03 ,  2.14992748e+03,   8.49509811e+02,
    #    1.25330538e+03,  2.06980502e+03,   2.66942131e+03,   3.17333932e+03,
    #    5.20887416e+02,  1.17197888e+03,   1.77467535e+03,   2.34835242e+03,
    #    2.94834858e+03,  3.48446196e+03,   2.23615977e+02,   8.43519645e+02,
    #    1.45861016e+03,  2.04937411e+03,   2.63434115e+03,   3.19248755e+03,
    #    3.80091400e+03,  5.16387353e+02,   1.12946156e+03,   1.71380366e+03,
    #    2.31148383e+03,  2.89972789e+03,   3.49469687e+03,   1.14272698e+02,
    #    8.06322431e+02,  1.39573273e+03,   1.98558136e+03,   2.58664099e+03,
    #    3.17215526e+03,  3.77780823e+03,   1.91560512e+00,   5.18791653e+02,
    #    1.08456359e+03,  1.65583492e+03,   2.25970921e+03,   2.84701416e+03,
    #    3.46531448e+03,  3.34670329e+02,   7.88492138e+02,   1.36536723e+03,
    #    1.93192962e+03,  2.53436710e+03,   3.13698768e+03,   3.64135898e+03,
    #    1.96152164e+01,  4.30704295e+02,   1.08546714e+03,   1.62322271e+03,
    #    2.23308037e+03,  2.79012942e+03,   3.16317851e+02,   8.04483424e+02,
    #    1.30105685e+03,  1.92375036e+03,   1.06966504e+03,   1.58475993e+03])
    # turbineY = np.array([    5.96179678 ,    8.41178565 ,   19.90998261 ,  403.52020005,   487.51119414,
    #    529.63058749 ,  558.16046347,   636.74515979 ,  932.06577682,   996.09368606,
    #   1030.0555613  , 1078.1241112 ,  1082.997087   , 1105.86393307,  1483.69947921,
    #   1522.15853244 , 1538.69420215,  1546.42624459 , 1556.97242871,  1592.20631442,
    #   1633.284      , 2046.44771216,  2049.79602541 , 2060.82705369,  2075.5604536,
    #   2093.19354768 , 2114.11724017,  2183.62367092 , 2587.49853571,  2589.93456222,
    #   2594.03812402 , 2609.57394084,  2612.99724227 , 2651.16200026,  3040.45514463,
    #   3001.53608378 , 3128.58176805,  3113.33961726 , 3130.46762743,  3129.57353227,
    #   3164.25311369 , 3641.92308981,  3537.00301337 , 3567.09645254,  3640.7846954,
    #   3654.32875358 , 3689.72810109,  3613.87889096 , 3885.61158172,  4070.50942825,
    #   4105.3435327  , 4063.14206301,  4189.43565096 , 4168.66034541,  4401.65850536,
    #   4455.76597777, 4598.55535345,  4635.31668121 , 4838.04864516,  4886.2048084 ])

    nTurbs = len(turbineX)

    nPoints = 3
    nFull = 15

    d_param = np.array([6.3,4.7515362,3.87])
    t_param = np.array([0.0200858,0.01623167,0.00975147])

    shearExp = 0.08
    ratedPower = np.array([5000.,5000.])
    rotorDiameter = np.array([126.4,126.4])
    turbineZ = np.array([90.,90.])

    numDirs = 30
    numSpeeds = 1
    COE = np.zeros((numDirs,numSpeeds))
    AEP = np.zeros((numDirs,numSpeeds))
    cost = np.zeros((numDirs,numSpeeds))
    nD = np.zeros((numDirs,numSpeeds))
    nS = np.zeros((numDirs,numSpeeds))

    for k in range(numDirs):
        for l in range(numSpeeds):
            # nDirs = 23+k
            nDirs = k+2
            nSpeeds = l+2
            dirs, freqs, speeds = amaliaRose(nDirs)
            # print freqs
            windDirections, windFrequencies, windSpeeds = setup_weibull(dirs,freqs,speeds,nSpeeds)
            # print windDirections
            # print windFrequencies
            # print windSpeeds
            print 'sampled'
            nDirections = len(windDirections)
            yaw = np.ones((nDirections, nTurbs))*0.

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

            print 'added'

            root.connect('turbineZ', 'Zs.Array')

            for i in range(nGroups):
                root.connect('rotorDiameter%s'%i, 'Rotor%s.rotorDiameter'%i)
                root.connect('ratedPower%s'%i, 'Rotor%s.turbineRating'%i)
                root.connect('Rotor%s.ratedQ'%i, 'rotor_nacelle_costs%s.rotor_torque'%i)

                root.connect('Rotor%s.blade_mass'%i, 'rotor_nacelle_costs%s.blade_mass'%i)
                root.connect('Rotor%s.Vrated'%i,'Tower%s_max_thrust.Vel'%i)
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

            for j in range(nGroups):
                root.connect('rotor_diameters%s'%j,'rotor_nacelle_costs%s.rotor_diameter'%j)
                root.connect('rated_powers%s'%j,'rotor_nacelle_costs%s.machine_rating'%j)

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
            print 'connected'

            prob.root.ln_solver.options['single_voi_relevance_reduction'] = True
            prob.setup(check=True)
            print 'setup'

            end_setup = time()

            start_assign = time()

            setupTower(nFull, prob)
            simpleSetup(nTurbs, prob)
            prob['Uref'] = windSpeeds
            prob['windDirections'] = windDirections
            prob['windFrequencies'] = windFrequencies

            for i in range(nDirections):
                prob['yaw%s'%i] = yaw[i]
            prob['turbineX'] = turbineX
            prob['turbineY'] = turbineY
            prob['shearExp'] = shearExp

            prob['turbineX'] = turbineX
            prob['turbineY'] = turbineY

            for i in range(nGroups):
                prob['Tower%s_max_speed.Vel'%i] = 70.
            print 'running'
            prob.run()
            COE[k][l] = prob['COE']
            AEP[k][l] = prob['AEP']
            cost[k][l] = prob['cost']
            nD[k][l] = nDirs
            nS[k][l] = nSpeeds
            print k
            print l


    # nSpeeds = np.linspace(2,num+1,num)
    print 'nDirections = np.', repr(nD)
    print 'nSpeeds = np.', repr(nS)
    print 'COE = np.', repr(COE)
    print 'AEP = np.', repr(AEP)

    plt.figure(1)
    plt.title('COE')
    plt.plot(nD,AEP[0],'ob',markersize=5)

    #
    # plt.figure(2)
    # plt.title('AEP')
    # plt.plot(nSpeeds,AEP,linewidth=2)
    #
    # plt.figure(3)
    # plt.title('cost')
    # plt.plot(nSpeeds,cost,linewidth=2)
    #
    plt.show()
