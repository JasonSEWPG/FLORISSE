import numpy as np
from openmdao.api import Group, Problem, IndepVarComp, pyOptSparseDriver
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
        NREL5MWCPCT = pickle.load(open('../../../doc/tune/NREL5MWCPCT_smooth_dict.p'))
        # print(NREL5MWCPCT)
        # NREL5MWCPCT = pickle.Unpickler(open('./input_files/NREL5MWCPCT.p')).load()
        datasize = NREL5MWCPCT['CP'].size
    else:
        datasize = 0

    rotor_diameter = 126.4

    nRows = 5
    nTurbs = nRows**2
    # spacing = float(argv[1])  # turbine grid spacing in diameters
    spacing = 3.0
    points = np.linspace(start=spacing*rotor_diameter, stop=nRows*spacing*rotor_diameter, num=nRows)
    xpoints, ypoints = np.meshgrid(points, points)
    turbineX = np.ndarray.flatten(xpoints)
    turbineY = np.ndarray.flatten(ypoints)

    rotorDiameter = np.zeros(nTurbs)
    axialInduction = np.zeros(nTurbs)
    Ct = np.zeros(nTurbs)
    Cp = np.zeros(nTurbs)
    generatorEfficiency = np.zeros(nTurbs)
    yaw = np.zeros(nTurbs)


    # define initial values
    for turbI in range(0, nTurbs):
        rotorDiameter[turbI] = rotor_diameter            # m
        axialInduction[turbI] = 1.0/3.0
        Ct[turbI] = 4.0*axialInduction[turbI]*(1.0-axialInduction[turbI])
        # Cp[turbI] = 0.7737/0.944 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
        Cp[turbI] = 0.7737 * 4.0 * 1.0/3.0 * np.power((1. - 1.0/3.0), 2)
        generatorEfficiency[turbI] = 1.0#0.944
        yaw[turbI] = 0.     # deg.

    minSpacing = 2.0

    # generate boundary constraint
    locations = np.zeros((len(turbineX),2))
    for i in range(len(turbineX)):
        locations[i][0] = turbineX[i]
        locations[i][1] = turbineY[i]
    print locations
    boundaryVertices, boundaryNormals = calculate_boundary(locations)
    nVertices = boundaryVertices.shape[0]

    """set up 3D aspects of wind farm"""
    H1_H2 = np.array([])
    for i in range(nTurbs/2):
        H1_H2 = np.append(H1_H2, 0)
        H1_H2 = np.append(H1_H2, 1)
    if len(H1_H2) < nTurbs:
        H1_H2 = np.append(H1_H2, 0)
    H1_H2 = np.zeros(nTurbs)

    """Define wind flow"""
    air_density = 1.1716    # kg/m^3

    windData = "Amalia"

    """Amalia Wind Arrays"""
    if windData == "Amalia":
        windSpeeds, windFrequencies, windDirections, nDirections = amaliaWind()
        # nDirections = 72
        # windSpeeds = np.ones(nDirections)*10.
        # windFrequencies = np.ones(nDirections)/nDirections
        # windDirections = np.linspace(0.,360.-360./nDirections,nDirections)


    """Define tower structural properties"""
    # --- geometry ----
    d_paramH1 = np.array([5.0, 4.0, 3.0])
    t_paramH1 = np.array([0.03,0.02,0.015])

    d_paramH2 = np.array([5.0, 4.0, 3.0])+1.0
    t_paramH2 = np.array([0.03,0.02,0.015])+0.01

    n = 15

    turbineH1 = 110.
    turbineH2 = 73.2
    turbineH1 = 90.
    turbineH2 = 90.

    L_reinforced, Toweryaw, E, G, rho, sigma_y, kidx, kx, ky, kz, ktx, kty, ktz, nK, \
                midx, m, mIxx, mIyy, mIzz, mIxy, mIxz, mIyz, mrhox, mrhoy, mrhoz, nMass, \
                addGravityLoadForExtraMass, wind_zref, wind_z0, shearExp, wind_Uref1, \
                plidx1, Fx1, Fy1, Fz1, Mxx1, Myy1, Mzz1, nPL, wind_Uref2, \
                plidx2, Fx2, Fy2, Fz2, Mxx2, Myy2, Mzz2, gamma_f, gamma_m, gamma_n, \
                gamma_b, z_DEL, M_DEL, nDEL, gamma_fatigue, life, m_SN, min_d_to_t, min_taper \
                = setupTower(n)

    nPoints = len(d_paramH1)
    nFull = n
    rhoAir = air_density

    shearExp = float(argv[1])
    # shearExp = 0.1

    
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

    prob.driver = pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'SNOPT'
    prob.driver.opt_settings['Summary file'] = 'XYdt_analytic_%s.out'%spacing
    # prob.driver.opt_settings['Print file'] = 'dt_analytic_testFD.out'
    prob.driver.opt_settings['Major iterations limit'] = 1000
    prob.driver.opt_settings['Major optimality tolerance'] = 1.0E-4
    prob.driver.opt_settings['Major feasibility tolerance'] = 1.0E-4

    prob.driver.opt_settings['Function precision'] = 1.0E-8

    # --- Objective ---
    prob.driver.add_objective('COE', scaler=1.0E-1)

    # # --- Design Variables ---
    # prob.driver.add_desvar('turbineH1', lower=rotor_diameter/2.+10., upper=None, scaler=1.0E-1)
    # prob.driver.add_desvar('turbineH2', lower=rotor_diameter/2.+10., upper=None, scaler=1.0E-1)
    prob.driver.add_desvar('turbineX', lower=np.ones(nTurbs)*min(turbineX), upper=np.ones(nTurbs)*max(turbineX), scaler=1.0E-3)
    prob.driver.add_desvar('turbineY', lower=np.ones(nTurbs)*min(turbineY), upper=np.ones(nTurbs)*max(turbineY), scaler=1.0E-3)
    prob.driver.add_desvar('d_paramH1', lower=np.array([1.0, 1.0, 3.87]), upper=np.ones(nPoints)*6.3, scaler=1.0E-1)
    prob.driver.add_desvar('t_paramH1', lower=np.ones(nPoints)*.001, upper=np.ones(nPoints)*0.05, scaler=1.0E1)
    prob.driver.add_desvar('d_paramH2', lower=np.array([1.0, 1.0, 3.87]), upper=np.ones(nPoints)*6.3, scaler=1.0E-1)
    prob.driver.add_desvar('t_paramH2', lower=np.ones(nPoints)*.001, upper=np.ones(nPoints)*0.05, scaler=1.0E1)

    # boundary constraint (convex hull)
    prob.driver.add_constraint('boundaryDistances', lower=np.zeros(nVertices*nTurbs), scaler=1.0)
    # spacing constraint
    prob.driver.add_constraint('sc', lower=np.zeros(((nTurbs-1.)*nTurbs/2.)), scaler=1.0/rotor_diameter)

    prob.driver.add_constraint('TowerH1_max_thrust.shell_buckling', upper=np.ones(nFull))
    prob.driver.add_constraint('TowerH2_max_thrust.shell_buckling', upper=np.ones(nFull))
    prob.driver.add_constraint('TowerH1_max_speed.shell_buckling', upper=np.ones(nFull))
    prob.driver.add_constraint('TowerH2_max_speed.shell_buckling', upper=np.ones(nFull))
    freq1p = 0.2  # 1P freq in Hz
    prob.driver.add_constraint('TowerH1_max_thrust.freq', lower=1.1*freq1p)
    prob.driver.add_constraint('TowerH2_max_thrust.freq', lower=1.1*freq1p)
    prob.driver.add_constraint('TowerH1_max_speed.freq', lower=1.1*freq1p)
    prob.driver.add_constraint('TowerH2_max_speed.freq', lower=1.1*freq1p)

    # ----------------------

    prob.root.ln_solver.options['single_voi_relevance_reduction'] = True

    prob.setup(check=True)

    prob['H1_H2'] = H1_H2
    prob['turbineX'] = turbineX
    prob['turbineY'] = turbineY
    prob['yaw0'] = yaw
    prob['ratedPower'] = np.ones_like(turbineX)*5000. # in kw

    prob['boundaryVertices'] = boundaryVertices
    prob['boundaryNormals'] = boundaryNormals

    # assign values to constant inputs (not design variables)
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

    print 'TOWER 1: max_thrust'
    print 'Mass: ', prob['TowerH1_max_thrust.mass']
    print 'Shell Buckling: ', prob['TowerH1_max_thrust.shell_buckling']
    print 'Frequency: ', prob['TowerH1_max_thrust.freq']
    print ''
    print 'TOWER 2: max_thrust'
    print 'Mass: ', prob['TowerH2_max_thrust.mass']
    print 'Shell Buckling: ', prob['TowerH2_max_thrust.shell_buckling']
    print 'Frequency: ', prob['TowerH2_max_thrust.freq']
    print ''
    print 'TOWER 1: max_speed'
    print 'Mass: ', prob['TowerH1_max_speed.mass']
    print 'Shell Buckling: ', prob['TowerH1_max_speed.shell_buckling']
    print 'Frequency: ', prob['TowerH1_max_speed.freq']
    print ''
    print 'TOWER 2: max_speed'
    print 'Mass: ', prob['TowerH2_max_speed.mass']
    print 'Shell Buckling: ', prob['TowerH2_max_speed.shell_buckling']
    print 'Frequency: ', prob['TowerH2_max_speed.freq']
    print ''
    print 'AEP: ', prob['AEP']
    print 'COE: ', prob['COE']
    print 'Cost: ', prob['cost']
    print 'H1: ', prob['turbineH1']
    print 'H2: ', prob['turbineH2']
    print 'D1: ', prob['d_paramH1']
    print 'D2: ', prob['d_paramH2']
    print 'T1: ', prob['t_paramH1']
    print 'T2: ', prob['t_paramH2']

    np.savetxt('XYZ_XYdt_analytic_%s.txt'%spacing, np.c_[prob['turbineX'], prob['turbineY'], prob['turbineZ']], header="turbineX, turbineY, turbineZ")
    np.savetxt('dt_XYdt_analytic_%s.txt'%spacing, np.c_[prob['d_paramH1'], prob['d_paramH2'], prob['t_paramH1'], prob['t_paramH2'],])

    # """plot results"""
    # fig = plt.gcf()
    # ax = fig.gca()
    #
    # turbineXopt = prob['turbineX']
    # turbineYopt = prob['turbineY']
    # Z = prob['turbineZ']
    #
    # spacingGrid = spacing
    # points = np.linspace(start=spacingGrid*rotor_diameter, stop=nRows*spacingGrid*rotor_diameter, num=nRows)
    # xpoints, ypoints = np.meshgrid(points, points)
    # turbineXstart = np.ndarray.flatten(xpoints)
    # turbineYstart = np.ndarray.flatten(ypoints)
    # points = np.zeros((nTurbs,2))
    # for j in range(nTurbs):
    #     points[j] = (turbineXstart[j],turbineYstart[j])
    # hull = ConvexHull(points)
    #
    # ax.set_aspect('equal')
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.get_xaxis().tick_bottom()
    # ax.get_yaxis().tick_left()
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)
    #
    # for simplex in hull.simplices:
    #     ax.plot(points[simplex, 0], points[simplex, 1], 'k--')
    #
    # for j in range(nTurbs):
    #     if H1_H2[j] == 0:
    #         ax.add_artist(Circle(xy=(turbineXopt[j],turbineYopt[j]),
    #                   radius=rotor_diameter/2., fill=False, edgecolor='blue'))
    #     else:
    #         ax.add_artist(Circle(xy=(turbineXopt[j],turbineYopt[j]),
    #                   radius=rotor_diameter/2., fill=False, edgecolor='red'))
    #
    # ax.axis([min(turbineXopt)-200,max(turbineXopt)+200,min(turbineYopt)-200,max(turbineYopt)+200])
    # # plt.axes().set_aspect('equal')
    # # plt.legend()
    # plt.axis('off')
    # plt.title('Optimized Turbine Layout')
    # plt.show()
