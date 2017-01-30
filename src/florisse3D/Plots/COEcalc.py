import numpy as np
from openmdao.api import Group, Problem, IndepVarComp, pyOptSparseDriver
# from FLORISSE3D.setupOptimization import *
from setupOptimization import *
from FLORISSE3D.simpleTower import Tower
from FLORISSE3D.GeneralWindFarmComponents import calculate_boundary, SpacingComp,\
            BoundaryComp, get_z, getTurbineZ, AEPobj
from FLORISSE3D.COE import COEGroup
from FLORISSE3D.floris import AEPGroup
import cPickle as pickle
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.patches import Circle
from sys import argv


if __name__ == '__main__':
    use_rotor_components = True

    if use_rotor_components:
        NREL5MWCPCT = pickle.load(open('doc/tune/NREL5MWCPCT_smooth_dict.p'))
        # print(NREL5MWCPCT)
        # NREL5MWCPCT = pickle.Unpickler(open('./input_files/NREL5MWCPCT.p')).load()
        datasize = NREL5MWCPCT['CP'].size
    else:
        datasize = 0

    rotor_diameter = 126.4

    nRows = 5
    nTurbs = nRows**2

    rotorDiameter = np.zeros(nTurbs)
    axialInduction = np.zeros(nTurbs)
    Ct = np.zeros(nTurbs)
    Cp = np.zeros(nTurbs)
    generatorEfficiency = np.zeros(nTurbs)
    # yaw = np.zeros(nTurbs)

    H1_H2 = np.array([])
    for i in range(nTurbs/2):
        H1_H2 = np.append(H1_H2, 0)
        H1_H2 = np.append(H1_H2, 1)
    if len(H1_H2) < nTurbs:
        H1_H2 = np.append(H1_H2, 0)
    # H1_H2 = np.zeros(nTurbs)

    # define initial values
    for turbI in range(0, nTurbs):
        rotorDiameter[turbI] = rotor_diameter            # m
        axialInduction[turbI] = 1.0/3.0
        Ct[turbI] = 4.0*axialInduction[turbI]*(1.0-axialInduction[turbI])
        # Cp[turbI] = 0.7737/0.944 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
        Cp[turbI] = 0.7737 * 4.0 * 1.0/3.0 * np.power((1. - 1.0/3.0), 2)
        generatorEfficiency[turbI] = 1.0#0.944
        # yaw[turbI] = 0.     # deg.

    """Define wind flow"""
    air_density = 1.1716    # kg/m^3

    windSpeeds, windFrequencies, windDirections, nDirections = amaliaWind()

    n = 15

    L_reinforced, Toweryaw, E, G, rho, sigma_y, kidx, kx, ky, kz, ktx, kty, ktz, nK, \
                midx, m, mIxx, mIyy, mIzz, mIxy, mIxz, mIyz, mrhox, mrhoy, mrhoz, nMass, \
                addGravityLoadForExtraMass, wind_zref, wind_z0, shearExp, wind_Uref1, \
                plidx1, Fx1, Fy1, Fz1, Mxx1, Myy1, Mzz1, nPL, wind_Uref2, \
                plidx2, Fx2, Fy2, Fz2, Mxx2, Myy2, Mzz2, gamma_f, gamma_m, gamma_n, \
                gamma_b, z_DEL, M_DEL, nDEL, gamma_fatigue, life, m_SN, min_d_to_t, min_taper \
                = setupTower(n)

    nPoints = 3
    nFull = n
    rhoAir = air_density

    shear_ex = np.array([0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.3])
    # shear_ex = np.array([0.17,0.18,0.22,0.26,0.27,0.28,0.29,0.3])
    # shear_ex = np.array([0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.2,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.3])
    # shear_ex = np.array([0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.3])

    # shear_ex = np.array([2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0,8.5,9.0,9.5,10.0])
    # shear_ex = np.array([2.0])
    # shear_ex = np.array([2.0,2.5,3.0,3.5,4.0,4.5,5.5,6.0,6.5,7.0])
    COE = np.zeros(len(shear_ex))
    AEP = np.zeros(len(shear_ex))
    diff = np.zeros(len(shear_ex))
    zs = np.zeros((len(shear_ex),2))

    for i in range(len(shear_ex)):
    # for i in range(1):
        shearExp = shear_ex[i]
        # shearExp = 0.1
        print shear_ex[i]
        opt_filenameXYZ = 'src/florisse3D/Plots/Results_w_yaw/XYZ_XYZdt_analytic_%s.txt'%shear_ex[i]
        opt_filename_dt = 'src/florisse3D/Plots/Results_w_yaw/dt_XYZdt_analytic_%s.txt'%shear_ex[i]
        # opt_filenameXYZ = 'src/florisse3D/Plots/Z2_XYZ_XYZdt_%s.txt'%shear_ex[i]
        # opt_filename_dt = 'src/florisse3D/Plots/Z2_dt_XYZdt_%s.txt'%shear_ex[i]
        # # opt_filenameXYZ = 'XYZ_XYZdt7_analytic_%s.txt'%0.11
        # # opt_filename_dt = 'dt_XYZdt7_analytic_%s.txt'%0.11
        opt_filename_yaw = 'src/florisse3D/Plots/Results_w_yaw/YAW_XYZdt_analytic_%s.txt'%shear_ex[i]

        optXYZ = open(opt_filenameXYZ)
        optimizedXYZ = np.loadtxt(optXYZ)
        turbineX = optimizedXYZ[:,0]
        turbineY = optimizedXYZ[:,1]
        turbineZ = optimizedXYZ[:,2]
        turbineH1 = np.float(turbineZ[0])
        turbineH2 = np.float(turbineZ[1])

        opt_dt = open(opt_filename_dt)
        optimized_dt = np.loadtxt(opt_dt)
        d_paramH1 = optimized_dt[:,0]
        d_paramH2 = optimized_dt[:,1]
        t_paramH1 = optimized_dt[:,2]
        t_paramH2 = optimized_dt[:,3]
        diff[i] = abs(turbineH1-turbineH2)

        opt_yaw = open(opt_filename_yaw)
        optimizedYaw = np.loadtxt(opt_yaw)
        yaw = np.zeros((nDirections, nTurbs))
        for j in range(nDirections):
            yaw[j] = optimizedYaw[j]

        prob = Problem()
        root = prob.root = Group()

        root.add('turbineH1', IndepVarComp('turbineH1', turbineH1), promotes=['*'])
        root.add('turbineH2', IndepVarComp('turbineH2', turbineH2), promotes=['*'])
        root.add('d_paramH1', IndepVarComp('d_paramH1', d_paramH1), promotes=['*'])
        root.add('t_paramH1', IndepVarComp('t_paramH1', t_paramH1), promotes=['*'])
        root.add('d_paramH2', IndepVarComp('d_paramH2', d_paramH2), promotes=['*'])
        root.add('t_paramH2', IndepVarComp('t_paramH2', t_paramH2), promotes=['*'])
        root.add('getTurbineZ', getTurbineZ(nTurbs), promotes=['*'])
        root.add('get_z_paramH1', get_z(nPoints))
        root.add('get_z_fullH1', get_z(nFull))
        root.add('get_z_paramH2', get_z(nPoints))
        root.add('get_z_fullH2', get_z(nFull))
        root.add('TowerH1_max_thrust', Tower(nPoints, nFull), promotes=['L_reinforced','m','mrhox','E','sigma_y','gamma_f','gamma_b','rhoAir','z0','zref','shearExp','rho'])
        root.add('TowerH1_max_speed', Tower(nPoints, nFull), promotes=['L_reinforced','m','mrhox','E','sigma_y','gamma_f','gamma_b','rhoAir','z0','zref','shearExp','rho'])
        root.add('TowerH2_max_thrust', Tower(nPoints, nFull), promotes=['L_reinforced','m','mrhox','E','sigma_y','gamma_f','gamma_b','rhoAir','z0','zref','shearExp','rho'])
        root.add('TowerH2_max_speed', Tower(nPoints, nFull), promotes=['L_reinforced','m','mrhox','E','sigma_y','gamma_f','gamma_b','rhoAir','z0','zref','shearExp','rho'])
        root.add('AEPGroup', AEPGroup(nTurbs, nDirections=nDirections,
                    use_rotor_components=use_rotor_components, datasize=datasize, differentiable=True,
                    optimizingLayout=False, nSamples=0), promotes=['*'])
        root.add('COEGroup', COEGroup(nTurbs), promotes=['*'])
        root.add('maxAEP', AEPobj(), promotes=['*'])

        root.connect('get_z_paramH1.z_param', 'TowerH1_max_thrust.z_param')
        root.connect('get_z_fullH1.z_param', 'TowerH1_max_thrust.z_full')
        root.connect('get_z_paramH1.z_param', 'TowerH1_max_speed.z_param')
        root.connect('get_z_fullH1.z_param', 'TowerH1_max_speed.z_full')
        root.connect('turbineH1', 'get_z_paramH1.turbineZ')
        root.connect('turbineH1', 'get_z_fullH1.turbineZ')
        root.connect('turbineH1', 'TowerH1_max_thrust.L')
        root.connect('turbineH2', 'TowerH2_max_thrust.L')
        root.connect('turbineH1', 'TowerH1_max_speed.L')
        root.connect('turbineH2', 'TowerH2_max_speed.L')

        root.connect('get_z_paramH2.z_param', 'TowerH2_max_thrust.z_param')
        root.connect('get_z_fullH2.z_param', 'TowerH2_max_thrust.z_full')
        root.connect('get_z_paramH2.z_param', 'TowerH2_max_speed.z_param')
        root.connect('get_z_fullH2.z_param', 'TowerH2_max_speed.z_full')
        root.connect('turbineH2', 'get_z_paramH2.turbineZ')
        root.connect('turbineH2', 'get_z_fullH2.turbineZ')

        root.connect('TowerH1_max_thrust.mass', 'mass1')
        root.connect('TowerH2_max_thrust.mass', 'mass2')

        root.connect('d_paramH1', 'TowerH1_max_thrust.d_param')
        root.connect('t_paramH1', 'TowerH1_max_thrust.t_param')
        root.connect('d_paramH1', 'TowerH1_max_speed.d_param')
        root.connect('t_paramH1', 'TowerH1_max_speed.t_param')
        root.connect('d_paramH2', 'TowerH2_max_thrust.d_param')
        root.connect('t_paramH2', 'TowerH2_max_thrust.t_param')
        root.connect('d_paramH2', 'TowerH2_max_speed.d_param')
        root.connect('t_paramH2', 'TowerH2_max_speed.t_param')

        root.connect('TowerH1_max_speed.Mt', 'TowerH1_max_speed.Mt')
        root.connect('TowerH1_max_speed.It', 'TowerH1_max_speed.It')
        root.connect('TowerH2_max_speed.Mt', 'TowerH2_max_speed.Mt')
        root.connect('TowerH2_max_speed.It', 'TowerH2_max_speed.It')

        # ----------------------

        prob.root.ln_solver.options['single_voi_relevance_reduction'] = True

        prob.setup(check=True)

        prob['H1_H2'] = H1_H2
        prob['turbineX'] = turbineX
        prob['turbineY'] = turbineY

        for j in range(nDirections):
            prob['yaw%s'%j] = yaw[j]

        prob['ratedPower'] = np.ones_like(turbineX)*5000. # in kw

        # assign values to constant inputs (not design variables)
        prob['rotorDiameter'] = rotorDiameter
        prob['axialInduction'] = axialInduction
        prob['generatorEfficiency'] = generatorEfficiency
        prob['air_density'] = air_density
        prob['windDirections'] = windDirections
        prob['windFrequencies'] = windFrequencies
        prob['Uref'] = windSpeeds

        if use_rotor_components == True:
            prob['gen_params:windSpeedToCPCT_CP'] = NREL5MWCPCT['CP']
            prob['gen_params:windSpeedToCPCT_CT'] = NREL5MWCPCT['CT']
            prob['gen_params:windSpeedToCPCT_wind_speed'] = NREL5MWCPCT['wind_speed']
        else:
            prob['Ct_in'] = Ct
            prob['Cp_in'] = Cp
        prob['floris_params:cos_spread'] = 1E12 # turns off cosine spread (just needs to be very large)

        prob['L_reinforced'] = L_reinforced
        prob['rho'] = rho
        # prob['rhoAir'] = rhoAir
        prob['shearExp'] = shearExp
        prob['E'] = E
        prob['gamma_f'] = gamma_f
        prob['gamma_b'] = gamma_b
        prob['sigma_y'] = sigma_y
        prob['turbineH1'] = turbineH1
        prob['m'] = m
        prob['mrhox'] = mrhox
        prob['zref'] = 90.
        prob['z0'] = 0.

        prob['turbineX'] = turbineX
        prob['turbineY'] = turbineY

        prob['TowerH1_max_thrust.Fy'] = Fy1
        prob['TowerH1_max_thrust.Fx'] = Fx1
        prob['TowerH1_max_thrust.Fz'] = Fz1
        prob['TowerH1_max_thrust.Mxx'] = Mxx1
        prob['TowerH1_max_thrust.Myy'] = Myy1
        prob['TowerH1_max_thrust.Vel'] = wind_Uref1
        prob['TowerH1_max_thrust.Mt'] = m[0]
        prob['TowerH1_max_thrust.It'] = mIzz[0]

        prob['TowerH2_max_thrust.Fy'] = Fy1
        prob['TowerH2_max_thrust.Fx'] = Fx1
        prob['TowerH2_max_thrust.Fz'] = Fz1
        prob['TowerH2_max_thrust.Mxx'] = Mxx1
        prob['TowerH2_max_thrust.Myy'] = Myy1
        prob['TowerH2_max_thrust.Vel'] = wind_Uref1
        prob['TowerH2_max_thrust.Mt'] = m[0]
        prob['TowerH2_max_thrust.It'] = mIzz[0]

        prob['TowerH1_max_speed.Fy'] = Fy2
        prob['TowerH1_max_speed.Fx'] = Fx2
        prob['TowerH1_max_speed.Fz'] = Fz2
        prob['TowerH1_max_speed.Mxx'] = Mxx2
        prob['TowerH1_max_speed.Myy'] = Myy2
        prob['TowerH1_max_speed.Vel'] = wind_Uref2

        prob['TowerH2_max_speed.Fy'] = Fy2
        prob['TowerH2_max_speed.Fx'] = Fx2
        prob['TowerH2_max_speed.Fz'] = Fz2
        prob['TowerH2_max_speed.Mxx'] = Mxx2
        prob['TowerH2_max_speed.Myy'] = Myy2
        prob['TowerH2_max_speed.Vel'] = wind_Uref2

        prob.run()

        # print 'turbineZ: ', prob['turbineZ']
        COE[i] = prob['COE']
        AEP[i] = prob['AEP']
        zs[i][0] = prob['turbineH1']
        zs[i][1] = prob['turbineH2']

    print 'COE: ', repr(COE)
    print 'nDirections: ', nDirections
    # print 'AEP: ', repr(AEP)
    # print 'diff: ', repr(diff)

    # print 'TurbineX: ', turbineX
    # print 'TurbineY: ', turbineY
    # print 'dH1: ', d_paramH1
    # print 'dH2: ', d_paramH2
    # print 'tH1: ', t_paramH1
    # print 'tH2: ', t_paramH2

    # for i in range(nDirections):
    #     print 'Direction ', i
    #     print prob['yaw%s'%i]

    # print repr(zs)
