import numpy as np
from openmdao.api import Group, Problem, IndepVarComp, pyOptSparseDriver
from FLORISSE3D.setupOptimization import *
from FLORISSE3D.simpleTower import Tower
from FLORISSE3D.GeneralWindFarmComponents import calculate_boundary, SpacingComp,\
            BoundaryComp, get_z, getTurbineZ, AEPobj, DeMUX, hGroups, randomStart,\
            getRotorDiameter, Loads, minHeight
from FLORISSE3D.COE import COEGroup
from FLORISSE3D.floris import AEPGroup
import cPickle as pickle
import matplotlib.pyplot as plt
import matplotlib

if __name__ == '__main__':
    use_rotor_components = True

    if use_rotor_components:
        # NREL5MWCPCT = pickle.load(open('doc/tune/NREL5MWCPCT_smooth_dict.p'))
        NREL5MWCPCT = pickle.load(open('../../../doc/tune/NREL5MWCPCT_smooth_dict.p'))
        datasize = NREL5MWCPCT['CP'].size
    else:
	datasize = 0
    rotor_diameter = 126.4

    nRows = 9
    nTurbs = nRows**2
    nGroups = 2
    density = 0.1354922143
    spacing = nRows/(2.*nRows-2.)*np.sqrt(3.1415926535/density)
    print 'SPACING: ', spacing
    spacing = 4.

    """Define wind flow"""
    air_density = 1.1716    # kg/m^3

    windData = "Amalia"

    """Amalia Wind Arrays"""
    if windData == "Amalia":
        windSpeeds, windFrequencies, windDirections, nDirections = amaliaWind()

    windSpeeds = np.array([10.])
    windDirections = np.array([0.])
    windFrequencies = np.array([1.0])
    nDirections = len(windSpeeds)

    axialInduction = np.zeros(nTurbs)
    Ct = np.zeros(nTurbs)
    Cp = np.zeros(nTurbs)
    generatorEfficiency = np.zeros(nTurbs)
    yaw = np.zeros((nDirections, nTurbs))

    # define initial values
    for turbI in range(0, nTurbs):
        axialInduction[turbI] = 1.0/3.0
        Ct[turbI] = 4.0*axialInduction[turbI]*(1.0-axialInduction[turbI])
        # Cp[turbI] = 0.7737/0.944 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
        Cp[turbI] = 0.7737 * 4.0 * 1.0/3.0 * np.power((1. - 1.0/3.0), 2)
        generatorEfficiency[turbI] = 1.0#0.944

    minSpacing = 2.0

    points = np.linspace(start=spacing*rotor_diameter, stop=nRows*spacing*rotor_diameter, num=nRows)
    xpoints, ypoints = np.meshgrid(points, points)
    turbineX_bounds = np.ndarray.flatten(xpoints)
    turbineY_bounds = np.ndarray.flatten(ypoints)

    # generate boundary constraint
    locations = np.zeros((len(turbineX_bounds),2))
    for i in range(len(turbineX_bounds)):
        locations[i][0] = turbineX_bounds[i]
        locations[i][1] = turbineY_bounds[i]
    print locations
    boundaryVertices, boundaryNormals = calculate_boundary(locations)
    nVertices = boundaryVertices.shape[0]


    """Define tower structural properties"""
    # --- geometry ---
    n = 15


    """same as for NREL 5MW"""
    L_reinforced = 30.0*np.ones(n)  # [m] buckling length
    Toweryaw = 0.0

    # --- material props ---
    E = 210.e9*np.ones(n)
    G = 80.8e9*np.ones(n)
    rho = 8500.0*np.ones(n)
    sigma_y = 450.0e6*np.ones(n)

    # --- spring reaction data.  Use float('inf') for rigid constraints. ---
    kidx = np.array([0], dtype=int)  # applied at base
    kx = np.array([float('inf')])
    ky = np.array([float('inf')])
    kz = np.array([float('inf')])
    ktx = np.array([float('inf')])
    kty = np.array([float('inf')])
    ktz = np.array([float('inf')])
    nK = len(kidx)

    """scale with rotor diameter"""
    # --- extra mass ----
    midx = np.array([n-1], dtype=int)  # RNA mass at top
    # m = np.array([285598.8])*(rotor_diameter/126.4)**3
    m = np.array([78055.827])
    mIxx = np.array([3.5622774377E+006])
    mIyy = np.array([1.9539222007E+006])
    mIzz = np.array([1.821096074E+006])
    mIxy = np.array([0.00000000e+00])
    mIxz = np.array([1.1141296293E+004])
    mIyz = np.array([0.00000000e+00])
    # mrhox = np.array([-1.13197635]) # Does not change with rotor_diameter
    mrhox = np.array([-0.1449])
    mrhoy = np.array([0.])
    mrhoz = np.array([1.389])
    nMass = len(midx)
    addGravityLoadForExtraMass = True
    # -----------

    # --- wind ---
    wind_zref = 90.0
    wind_z0 = 0.0
    # ---------------

    # if addGravityLoadForExtraMass=True be sure not to double count by adding those force here also
    # # --- loading case 1: max Thrust ---
    wind_Uref1 = 11.73732
    plidx1 = np.array([n-1], dtype=int)  # at  top
    Fx1 = np.array([283000.])
    Fy1 = np.array([0.])
    Fz1 = np.array([-765727.66])
    Mxx1 = np.array([1513000.])
    Myy1 = np.array([-1360000.])
    Mzz1 = np.array([-127400.])
    nPL = len(plidx1)
    # # ---------------

    # # --- loading case 2: max wind speed ---
    wind_Uref2 = 70.0
    plidx2 = np.array([n-1], dtype=int)  # at  top
    Fx2 = np.array([204901.5477])
    Fy2 = np.array([0.])
    Fz2 = np.array([-832427.12368949])
    Mxx2 = np.array([-642674.9329])
    Myy2 = np.array([-1507872])
    Mzz2 = np.array([54115.])
    # # ---------------

    # --- safety factors ---
    gamma_f = 1.35
    gamma_m = 1.3
    gamma_n = 1.0
    gamma_b = 1.1
    # ---------------

    # --- constraints ---
    min_d_to_t = 120.0
    min_taper = 0.4
    # ---------------

    nPoints = 3
    nFull = n
    rhoAir = air_density

    d_param0 = np.array([  5.60087163,  4.14158799  ,3.87     ])
    t_param0 = np.array([ 0.02258316 , 0.01944239,  0.01059205])

    d_param1 = np.array([ 6.3,  6.3 , 6.3     ])
    t_param1 = np.array([ 0.03589133,  0.01813387,  0.00965172])


    turbineX = turbineX_bounds
    turbineY = turbineY_bounds
    #t_param[2] = 0.01
    shearExp = 0.3
    rotorDiameter = np.random.rand(nGroups)*100.
    turbineZ = np.random.rand(nGroups)*100.
    num = 1
    rotor2 = 48.
    rotorDiameter = np.array([126.4, 126.4])
    turbineZ = np.array([120.,53.2])
    rotor = np.linspace(20.,126.4,num)

    """OpenMDAO"""

    COE = np.zeros(num)

    prob = Problem()
    root = prob.root = Group()


    root.add('d_param0', IndepVarComp('d_param0', d_param0), promotes=['*'])
    root.add('t_param0', IndepVarComp('t_param0', t_param0), promotes=['*'])
    root.add('d_param1', IndepVarComp('d_param1', d_param1), promotes=['*'])
    root.add('t_param1', IndepVarComp('t_param1', t_param1), promotes=['*'])
    for i in range(nGroups):
        # root.add('d_param%s'%i, IndepVarComp('d_param%s'%i, d_param), promotes=['*'])
        # root.add('t_param%s'%i, IndepVarComp('t_param%s'%i, t_param), promotes=['*'])
        root.add('get_z_param%s'%i, get_z(nPoints))
        root.add('get_z_full%s'%i, get_z(nFull))
        root.add('Tower%s_max_thrust'%i, Tower(nPoints, nFull), promotes=['L_reinforced','m','mrhox','E','sigma_y','gamma_f','gamma_b','rhoAir','z0','zref','shearExp','rho'])
        root.add('Tower%s_max_speed'%i, Tower(nPoints, nFull), promotes=['L_reinforced','m','mrhox','E','sigma_y','gamma_f','gamma_b','rhoAir','z0','zref','shearExp','rho'])
        root.add('turbineH%s'%i, IndepVarComp('turbineH%s'%i, float(turbineZ[i])), promotes=['*'])
        root.add('rotorDiameter%s'%i, IndepVarComp('rotorDiameter%s'%i, float(rotorDiameter[i])), promotes=['*'])
        root.add('minHeight%s'%i, minHeight())

    root.add('Zs', DeMUX(nTurbs))
    root.add('hGroups', hGroups(nTurbs), promotes=['*'])
    root.add('getRotorDiameter', getRotorDiameter(nTurbs), promotes=['*'])
    for i in range(nGroups):
        root.add('Loads%s'%i, Loads())
    root.add('AEPGroup', AEPGroup(nTurbs, nDirections=nDirections,
                use_rotor_components=use_rotor_components, datasize=datasize, differentiable=True,
                optimizingLayout=False, nSamples=0), promotes=['*'])
    root.add('COEGroup', COEGroup(nTurbs, nGroups), promotes=['*'])

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

        root.connect('Tower%s_max_thrust.mass'%i, 'mass%s'%i)

        root.connect('d_param%s'%i, 'Tower%s_max_thrust.d_param'%i)
        root.connect('t_param%s'%i, 'Tower%s_max_thrust.t_param'%i)
        root.connect('d_param%s'%i, 'Tower%s_max_speed.d_param'%i)
        root.connect('t_param%s'%i, 'Tower%s_max_speed.t_param'%i)

        root.connect('Tower%s_max_speed.Mt'%i, 'Tower%s_max_speed.Mt'%i)
        root.connect('Tower%s_max_speed.It'%i, 'Tower%s_max_speed.It'%i)

        root.connect('Loads%s.Fx1'%i, 'Tower%s_max_thrust.Fx'%i)
        root.connect('Loads%s.Fy1'%i, 'Tower%s_max_thrust.Fy'%i)
        root.connect('Loads%s.Fz1'%i, 'Tower%s_max_thrust.Fz'%i)
        root.connect('Loads%s.Mxx1'%i, 'Tower%s_max_thrust.Mxx'%i)
        root.connect('Loads%s.Myy1'%i, 'Tower%s_max_thrust.Myy'%i)

        root.connect('Loads%s.Fx2'%i, 'Tower%s_max_speed.Fx'%i)
        root.connect('Loads%s.Fy2'%i, 'Tower%s_max_speed.Fy'%i)
        root.connect('Loads%s.Fz2'%i, 'Tower%s_max_speed.Fz'%i)
        root.connect('Loads%s.Mxx2'%i, 'Tower%s_max_speed.Mxx'%i)
        root.connect('Loads%s.Myy2'%i, 'Tower%s_max_speed.Myy'%i)

        root.connect('Loads%s.m'%i, 'Tower%s_max_speed.Mt'%i)
        root.connect('Loads%s.mIzz'%i, 'Tower%s_max_speed.It'%i)
        root.connect('Loads%s.rotor'%i, 'rotorDiameter%s'%i)

        root.connect('rotorDiameter%s'%i, 'minHeight%s.diameter'%i)
        root.connect('turbineH%s'%i, 'minHeight%s.height'%i)

    prob.root.ln_solver.options['single_voi_relevance_reduction'] = True

    prob.setup(check=True)

    for i in range(nDirections):
        prob['yaw%s'%i] = yaw[i]
    prob['nGroups'] = nGroups
    prob['turbineX'] = turbineX
    prob['turbineY'] = turbineY

    prob['ratedPower'] = np.ones(nTurbs)*5000.#1543.209877 # in kw

    prob['boundaryVertices'] = boundaryVertices
    prob['boundaryNormals'] = boundaryNormals

    # assign values to constant inputs (not design variables)
    # prob['rotorDiameter'] = rotorDiameter
    # prob['rotor_diameter'] = rotor_diameter
    prob['axialInduction'] = axialInduction
    prob['generatorEfficiency'] = generatorEfficiency
    prob['air_density'] = air_density
    prob['windDirections'] = windDirections
    prob['windFrequencies'] = windFrequencies
    prob['Uref'] = windSpeeds


    prob['gen_params:windSpeedToCPCT_CP'] = NREL5MWCPCT['CP']
    prob['gen_params:windSpeedToCPCT_CT'] = NREL5MWCPCT['CT']
    prob['gen_params:windSpeedToCPCT_wind_speed'] = NREL5MWCPCT['wind_speed']

    prob['floris_params:cos_spread'] = 1E12 # turns off cosine spread (just needs to be very large)

    prob['L_reinforced'] = L_reinforced
    prob['rho'] = rho
    prob['shearExp'] = shearExp
    prob['E'] = E
    prob['gamma_f'] = gamma_f
    prob['gamma_b'] = gamma_b
    prob['sigma_y'] = sigma_y
    # prob['m'] = m
    prob['mrhox'] = mrhox
    prob['zref'] = 50.
    prob['z0'] = 0.

    prob['turbineX'] = turbineX
    prob['turbineY'] = turbineY

    for i in range(nGroups):
        prob['Tower%s_max_thrust.Vel'%i] = wind_Uref1
        prob['Tower%s_max_speed.Vel'%i] = wind_Uref2


    spac = np.array([3.,4.,5.])
    shear = np.array([0.08,0.15,0.3])
    COE = np.zeros((3,num))
    for l in range(3):
        prob['shearExp'] = 0.15
        spacing = spac[l]
        points = np.linspace(start=spacing*rotor_diameter, stop=nRows*spacing*rotor_diameter, num=nRows)
        xpoints, ypoints = np.meshgrid(points, points)
        prob['turbineX'] = np.ndarray.flatten(xpoints)
        prob['turbineY'] = np.ndarray.flatten(ypoints)
        for k in range(num):
            prob['rotorDiameter1'] = rotor[k]
            prob['turbineH1'] = rotor[k]/2. + 10.
            prob.run()
            COE[l][k] = prob['COE']

        print 'K!!!!: ', k


    font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 15}

    matplotlib.rc('font', **font)


    fig = plt.figure()
    ax = plt.subplot(111)


    ax.plot(rotor, COE[0], label='3 D')
    ax.plot(rotor, COE[1], label='4 D')
    ax.plot(rotor, COE[2], label='5 D')
    # plt.title('%s D Grid Spacing'%4.0)
    plt.title('%s Wind Shear Exponent'%0.15)
    plt.xlabel('Group 2 Rotor Diameter (m)')
    plt.ylabel('COE ($/MWh)')
    ax.text(128.5, 49., 'grid spacing', fontsize=15)
    plt.axis([20.,126.4,30.,60.])
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
