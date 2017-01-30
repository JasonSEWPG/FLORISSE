import numpy as np
from openmdao.api import Group, Problem, IndepVarComp, pyOptSparseDriver
from FLORISSE3D.setupOptimization import *
from FLORISSE3D.simpleTower import Tower
from FLORISSE3D.GeneralWindFarmComponents import calculate_boundary, SpacingComp,\
            BoundaryComp, get_z, getTurbineZ, AEPobj, DeMUX
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
        # NREL5MWCPCT = pickle.load(open('../../../doc/tune/NREL5MWCPCT_smooth_dict.p'))
        NREL5MWCPCT = pickle.load(open('doc/tune/NREL5MWCPCT_smooth_dict.p'))
        # print(NREL5MWCPCT)
        # NREL5MWCPCT = pickle.Unpickler(open('./input_files/NREL5MWCPCT.p')).load()
        datasize = NREL5MWCPCT['CP'].size
    else:
        datasize = 0

    rotor_diameter = 126.4

    nRows = 3
    nTurbs = nRows**2
    spacing = 2.0  # turbine grid spacing in diameters
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

    """Define wind flow"""
    air_density = 1.1716    # kg/m^3

    windData = "Amalia"

    """Amalia Wind Arrays"""
    if windData == "Amalia":
        windSpeeds, windFrequencies, windDirections, nDirections = amaliaWind()
        nDirections = 2
        windSpeeds = np.ones(nDirections)*10.
        windFrequencies = np.ones(nDirections)/nDirections
        # windDirections = np.linspace(0.,360.-360./nDirections,nDirections)
        windDirections = np.array([0.,90.])


    """Define tower structural properties"""
    # --- geometry ----
    d_param = np.array([5.0, 4.0, 3.0])
    t_param = np.array([0.03,0.02,0.015])

    n = 15

    L_reinforced, Toweryaw, E, G, rho, sigma_y, kidx, kx, ky, kz, ktx, kty, ktz, nK, \
                midx, m, mIxx, mIyy, mIzz, mIxy, mIxz, mIyz, mrhox, mrhoy, mrhoz, nMass, \
                addGravityLoadForExtraMass, wind_zref, wind_z0, shearExp, wind_Uref1, \
                plidx1, Fx1, Fy1, Fz1, Mxx1, Myy1, Mzz1, nPL, wind_Uref2, \
                plidx2, Fx2, Fy2, Fz2, Mxx2, Myy2, Mzz2, gamma_f, gamma_m, gamma_n, \
                gamma_b, z_DEL, M_DEL, nDEL, gamma_fatigue, life, m_SN, min_d_to_t, min_taper \
                = setupTower(n)

    nPoints = len(d_param)
    nFull = n
    rhoAir = air_density

    shearExp = 0.17

    prob = Problem()
    root = prob.root = Group()

    # turbineZ = np.array([75.,77.,79.,81.,83.,85.,87.,89.,91.,93.,95.,97.,99.,101.,103.,105.])
    turbineZ = np.array([75.,77.,79.,81.,83.,85.,87.,89.,91.])
    # turbineZ = np.array([70.,80.,90.,100.])

    root.add('turbineZ', IndepVarComp('turbineZ', turbineZ), promotes=['*'])
    for i in range(nTurbs):
        root.add('d_param%s'%i, IndepVarComp('d_param%s'%i, d_param), promotes=['*'])
        root.add('t_param%s'%i, IndepVarComp('t_param%s'%i, t_param), promotes=['*'])
        root.add('get_z_param%s'%i, get_z(nPoints))
        root.add('get_z_full%s'%i, get_z(nFull))
        root.add('Tower%s_max_thrust'%i, Tower(nPoints, nFull), promotes=['L_reinforced','m','mrhox','E','sigma_y','gamma_f','gamma_b','rhoAir','z0','zref','shearExp','rho'])
        root.add('Tower%s_max_speed'%i, Tower(nPoints, nFull), promotes=['L_reinforced','m','mrhox','E','sigma_y','gamma_f','gamma_b','rhoAir','z0','zref','shearExp','rho'])
    root.add('Zs', DeMUX(nTurbs))
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

    root.connect('turbineZ', 'Zs.Array')
    for i in range(nTurbs):
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

        root.connect('Tower%s_max_thrust.mass'%i, 'mass%s'%i)

        root.connect('d_param%s'%i, 'Tower%s_max_thrust.d_param'%i)
        root.connect('t_param%s'%i, 'Tower%s_max_thrust.t_param'%i)
        root.connect('d_param%s'%i, 'Tower%s_max_speed.d_param'%i)
        root.connect('t_param%s'%i, 'Tower%s_max_speed.t_param'%i)

        root.connect('Tower%s_max_speed.Mt'%i, 'Tower%s_max_speed.Mt'%i)
        root.connect('Tower%s_max_speed.It'%i, 'Tower%s_max_speed.It'%i)

    prob.driver = pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'SNOPT'
    prob.driver.opt_settings['Summary file'] = 'src/florisse3D/runFiles/AAAAA_SNOPT.out'
    # prob.driver.opt_settings['Print file'] = 'XYZdt_analytic_testFD.out'
    prob.driver.opt_settings['Major iterations limit'] = 1000
    prob.driver.opt_settings['Major optimality tolerance'] = 1.0E-4
    prob.driver.opt_settings['Major feasibility tolerance'] = 1.0E-4

    prob.driver.opt_settings['Function precision'] = 1.0E-8

    # --- Objective ---
    prob.driver.add_objective('COE', scaler=1.0E-1)

    # --- Design Variables ---
    prob.driver.add_desvar('turbineX', lower=np.ones(nTurbs)*min(turbineX), upper=np.ones(nTurbs)*max(turbineX), scaler=1.0E-3)
    prob.driver.add_desvar('turbineY', lower=np.ones(nTurbs)*min(turbineY), upper=np.ones(nTurbs)*max(turbineY), scaler=1.0E-3)
    prob.driver.add_desvar('turbineZ', lower=73.2, upper=np.ones(nTurbs)*max(turbineY), scaler=1.0E-2)

    for i in range(nTurbs):
        prob.driver.add_desvar('d_param%s'%i, lower=np.array([1.0, 1.0, 3.87]), upper=np.ones(nPoints)*6.3, scaler=1.0E-1)
        prob.driver.add_desvar('t_param%s'%i, lower=np.ones(nPoints)*.001, upper=np.ones(nPoints)*0.05, scaler=1.0E1)

    # for direction_id in range(nDirections):
    #     prob.driver.add_desvar('yaw%i' % direction_id, lower=-30.0, upper=30.0, scaler=1)

    # prob.driver.add_desvar('yaw1', lower=-30.0, upper=30.0, scaler=1)

    # boundary constraint (convex hull)
    prob.driver.add_constraint('boundaryDistances', lower=np.zeros(nVertices*nTurbs), scaler=1.0)
    # spacing constraint
    prob.driver.add_constraint('sc', lower=np.zeros(((nTurbs-1.)*nTurbs/2.)), scaler=1.0/rotor_diameter)


    for i in range(nTurbs):
        prob.driver.add_constraint('Tower%s_max_thrust.shell_buckling'%i, upper=np.ones(nFull))
        prob.driver.add_constraint('Tower%s_max_speed.shell_buckling'%i, upper=np.ones(nFull))
        freq1p = 0.2  # 1P freq in Hz
        prob.driver.add_constraint('Tower%s_max_thrust.freq'%i, lower=1.1*freq1p)
        prob.driver.add_constraint('Tower%s_max_speed.freq'%i, lower=1.1*freq1p)

    # ----------------------

    prob.root.ln_solver.options['single_voi_relevance_reduction'] = True

    prob.setup(check=True)

    prob['turbineX'] = turbineX
    prob['turbineY'] = turbineY
    # for i in range(nDirections):
    #     prob['yaw%s'%i] = yaw

    prob['ratedPower'] = np.ones_like(turbineX)*5000. # in kw

    prob['boundaryVertices'] = boundaryVertices
    prob['boundaryNormals'] = boundaryNormals

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
    prob['shearExp'] = shearExp
    prob['E'] = E
    prob['gamma_f'] = gamma_f
    prob['gamma_b'] = gamma_b
    prob['sigma_y'] = sigma_y
    prob['m'] = m
    prob['mrhox'] = mrhox
    prob['zref'] = 90.
    prob['z0'] = 0.

    prob['turbineX'] = turbineX
    prob['turbineY'] = turbineY

    for i in range(nTurbs):
        prob['Tower%s_max_thrust.Fy'%i] = Fy1
        prob['Tower%s_max_thrust.Fx'%i] = Fx1
        prob['Tower%s_max_thrust.Fz'%i] = Fz1
        prob['Tower%s_max_thrust.Mxx'%i] = Mxx1
        prob['Tower%s_max_thrust.Myy'%i] = Myy1
        prob['Tower%s_max_thrust.Vel'%i] = wind_Uref1
        prob['Tower%s_max_thrust.Mt'%i] = m[0]
        prob['Tower%s_max_thrust.It'%i] = mIzz[0]

        prob['Tower%s_max_speed.Fy'%i] = Fy2
        prob['Tower%s_max_speed.Fx'%i] = Fx2
        prob['Tower%s_max_speed.Fz'%i] = Fz2
        prob['Tower%s_max_speed.Mxx'%i] = Mxx2
        prob['Tower%s_max_speed.Myy'%i] = Myy2
        prob['Tower%s_max_speed.Vel'%i] = wind_Uref2


    prob.run()

    # print 'TOWER 1: max_thrust'
    # print 'Mass: ', prob['TowerH1_max_thrust.mass']
    # print 'Shell Buckling: ', prob['TowerH1_max_thrust.shell_buckling']
    # print 'Frequency: ', prob['TowerH1_max_thrust.freq']
    # print ''
    # print 'TOWER 2: max_thrust'
    # print 'Mass: ', prob['TowerH2_max_thrust.mass']
    # print 'Shell Buckling: ', prob['TowerH2_max_thrust.shell_buckling']
    # print 'Frequency: ', prob['TowerH2_max_thrust.freq']
    # print ''
    # print 'TOWER 1: max_speed'
    # print 'Mass: ', prob['TowerH1_max_speed.mass']
    # print 'Shell Buckling: ', prob['TowerH1_max_speed.shell_buckling']
    # print 'Frequency: ', prob['TowerH1_max_speed.freq']
    # print ''
    # print 'TOWER 2: max_speed'
    # print 'Mass: ', prob['TowerH2_max_speed.mass']
    # print 'Shell Buckling: ', prob['TowerH2_max_speed.shell_buckling']
    # print 'Frequency: ', prob['TowerH2_max_speed.freq']
    # print ''
    print 'AEP: ', prob['AEP']
    print 'COE: ', prob['COE']
    # print 'Cost: ', prob['cost']

    print 'TurbineX: ', prob['turbineX']
    print 'TurbineY: ', prob['turbineY']
    print 'TurbineZ: ', prob['turbineZ']
    yawAngles = np.zeros((nDirections, nTurbs))
    for i in range(nDirections):
        yawAngles[i] = prob['yaw%s'%i]
        print prob['yaw%s'%i]
    diameters = np.zeros((nTurbs, 3))
    thicknesses = np.zeros((nTurbs, 3))
    for i in range(nTurbs):
        diameters[i] = prob['d_param%s'%i]
        thicknesses[i] = prob['t_param%s'%i]


    print 'Yaw Angles: ', yawAngles


    # np.savetxt('src/florisse3D/runFiles/A_XYZ.txt', np.c_[prob['turbineX'], prob['turbineY'], prob['turbineZ']], header="turbineX, turbineY, turbineZ")
    # np.savetxt('src/florisse3D/runFiles/A_d.txt', np.c_[diameters], header="diameters")
    # np.savetxt('src/florisse3D/runFiles/A_t.txt', np.c_[thicknesses], header="thicknesses")
    # np.savetxt('src/florisse3D/runFiles/A_yaw.txt', np.c_[yawAngles], header="yaw")

    fig = plt.gcf()
    ax = fig.gca()

    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    color = (0,0.6,0.8)
    for j in range(nTurbs):
        ax.add_artist(Circle(xy=(prob['turbineX'][j],prob['turbineY'][j]),
                  radius=rotor_diameter/2., fill=False, edgecolor=color))

    ax.axis([min(prob['turbineX'])-200,max(prob['turbineX'])+200,min(prob['turbineY'])-200,max(prob['turbineY'])+200])

    plt.axis('off')
    plt.show()
