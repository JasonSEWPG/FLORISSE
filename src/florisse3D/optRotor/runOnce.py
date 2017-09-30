import numpy as np
from openmdao.api import Group, Problem, IndepVarComp, pyOptSparseDriver
from FLORISSE3D.setupOptimization import *
from FLORISSE3D.simpleTower import Tower
from FLORISSE3D.GeneralWindFarmComponents import calculate_boundary, SpacingComp,\
            BoundaryComp, get_z, getTurbineZ, AEPobj, DeMUX, hGroups, randomStart,\
            getRotorDiameter, Loads
from FLORISSE3D.COE import COEGroup
from rotorse.precomp import Profile, Orthotropic2DMaterial, CompositeSection, _precomp
from FLORISSE3D.floris import AEPGroup
from FLORISSE3D.rotorComponents import getRating
import cPickle as pickle
from scipy.spatial import ConvexHull
from sys import argv
from rotorse.rotor import RotorSE
import os


if __name__ == '__main__':

    use_rotor_components = False
    datasize = 0
    rotor_diameter = 126.4

    nRows = 5
    nTurbs = nRows**2
    nGroups = 2
    # density = 0.1354922143
    # spacing = nRows/(2.*nRows-2.)*np.sqrt(3.1415926535/density)
    # print 'SPACING: ', spacing
    spacing = 5.

    """Define wind flow"""
    air_density = 1.1716    # kg/m^3


    # windSpeeds, windFrequencies, windDirections, nDirections = amaliaWind()

    windSpeeds = np.array([12.])
    windFrequencies = np.array([1.])
    windDirections = np.array([0.])
    nDirections = 1


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
    # print locations
    boundaryVertices, boundaryNormals = calculate_boundary(locations)
    nVertices = boundaryVertices.shape[0]

    turbineX = turbineX_bounds
    turbineY = turbineY_bounds


    nPoints = 3
    nFull = 15
    rhoAir = air_density

    d_param = np.array([6.3,5.3,4.3])
    t_param = np.array([0.02,0.01,0.006])

    # turbineX = turbineX_bounds
    # turbineY = turbineY_bounds
    #t_param[2] = 0.01
    shearExp = 0.08
    rotorDiameter = np.array([70.,70.,150.,155.,141.])
    turbineZ = np.array([45., 70., 100., 120., 30.])
    ratedPower = np.random.rand(nGroups)*10000.
    """OpenMDAO"""

    prob = Problem()
    root = prob.root = Group()

    for i in range(nGroups):
        root.add('ratedPower%s'%i, IndepVarComp('ratedPower%s'%i, ratedPower[i]), promotes=['*'])
        root.add('d_param%s'%i, IndepVarComp('d_param%s'%i, d_param), promotes=['*'])
        root.add('t_param%s'%i, IndepVarComp('t_param%s'%i, t_param), promotes=['*'])
        root.add('get_z_param%s'%i, get_z(nPoints))
        root.add('get_z_full%s'%i, get_z(nFull))
        root.add('Tower%s_max_thrust'%i, Tower(nPoints, nFull), promotes=['L_reinforced','m','mrhox','E','sigma_y','gamma_f','gamma_b','rhoAir','z0','zref','shearExp','rho'])
        root.add('Tower%s_max_speed'%i, Tower(nPoints, nFull), promotes=['L_reinforced','m','mrhox','E','sigma_y','gamma_f','gamma_b','rhoAir','z0','zref','shearExp','rho'])
        root.add('turbineH%s'%i, IndepVarComp('turbineH%s'%i, float(turbineZ[i])), promotes=['*'])
        root.add('rotorDiameter%s'%i, IndepVarComp('rotorDiameter%s'%i, float(rotorDiameter[i])), promotes=['*'])



    root.add('Zs', DeMUX(nTurbs))
    root.add('hGroups', hGroups(nTurbs), promotes=['*'])
    root.add('getRotorDiameter', getRotorDiameter(nTurbs, nGroups), promotes=['*'])
    root.add('getRating', getRating(nTurbs), promotes=['*'])
    for i in range(nGroups):
        root.add('Loads%s'%i, Loads())
        # root.add('Rotor%s_max_speed'%i, RotorSE(naero=17, nstr=38, npower=20))
        # root.add('Rotor%s_max_thrust'%i, RotorSE(naero=17, nstr=38, npower=20))
    root.add('COEGroup', COEGroup(nTurbs, nGroups, nDirections, datasize), promotes=['*'])



    # root.connect('rotorDiameter','getRating.rotorDiameter')
    # root.connect('getRating.ratedPower','ratedPower')

    # root.add('spacing_comp', SpacingComp(nTurbines=nTurbs), promotes=['*'])
    #
    # # add constraint definitions
    # root.add('spacing_con', ExecComp('sc = wtSeparationSquared-(minSpacing*rotorDiameter[0])**2',
    #                              minSpacing=minSpacing, rotorDiameter=np.zeros(nTurbs),
    #                              sc=np.zeros(((nTurbs-1)*nTurbs/2)),
    #                              wtSeparationSquared=np.zeros(((nTurbs-1)*nTurbs/2))),
    #                              promotes=['*'])

    # if nVertices > 0:
    #     # add component that enforces a convex hull wind farm boundary
    #     root.add('boundary_con', BoundaryComp(nVertices=nVertices, nTurbines=nTurbs), promotes=['*'])

    root.connect('turbineZ', 'Zs.Array')

    for j in range(nTurbs):
        root.connect('rotor_diameters%s'%j,'nrel_csm_tcc_2015%s.rotor_diameter'%j)
        root.connect('rated_powers%s'%j,'nrel_csm_tcc_2015%s.machine_rating'%j)

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


    prob.root.ln_solver.options['single_voi_relevance_reduction'] = True

    prob.setup(check=True)

    for i in range(nDirections):
        prob['yaw%s'%i] = yaw[i]
    prob['nGroups'] = nGroups
    prob['turbineX'] = turbineX
    prob['turbineY'] = turbineY

    # prob['boundaryVertices'] = boundaryVertices
    # prob['boundaryNormals'] = boundaryNormals

    # assign values to constant inputs (not design variables)
    # prob['rotorDiameter'] = rotorDiameter
    # prob['rotor_diameter'] = rotor_diameter
    prob['axialInduction'] = axialInduction
    prob['generatorEfficiency'] = generatorEfficiency
    prob['air_density'] = air_density
    prob['windDirections'] = windDirections
    prob['windFrequencies'] = windFrequencies
    prob['Uref'] = windSpeeds

    prob['turbine_class'] = 'II/III'
    prob['blade_has_carbon'] = True
    prob['bearing_number'] = 2

    # if use_rotor_components == True:
    #     prob['gen_params:windSpeedToCPCT_CP'] = NREL5MWCPCT['CP']
    #     prob['gen_params:windSpeedToCPCT_CT'] = NREL5MWCPCT['CT']
    #     prob['gen_params:windSpeedToCPCT_wind_speed'] = NREL5MWCPCT['wind_speed']
    # else:
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
    # prob['m'] = m
    prob['mrhox'] = mrhox
    prob['zref'] = 50.
    prob['z0'] = 0.

    prob['turbineX'] = turbineX
    prob['turbineY'] = turbineY

    for i in range(nGroups):
        prob['Tower%s_max_thrust.Vel'%i] = wind_Uref1
        prob['Tower%s_max_speed.Vel'%i] = wind_Uref2


    #ROTOR stuff
        # === blade grid ===
    for i in range(nGroups):


        prob['Rotor%s_max_speed.initial_aero_grid'%i] = np.array([0.02222276, 0.06666667, 0.11111057, 0.16666667, 0.23333333, 0.3, 0.36666667,
            0.43333333, 0.5, 0.56666667, 0.63333333, 0.7, 0.76666667, 0.83333333, 0.88888943, 0.93333333,
            0.97777724])  # (Array): initial aerodynamic grid on unit radius
        prob['Rotor%s_max_speed.initial_str_grid'%i] = np.array([0.0, 0.00492790457512, 0.00652942887106, 0.00813095316699, 0.00983257273154,
            0.0114340970275, 0.0130356213234, 0.02222276, 0.024446481932, 0.026048006228, 0.06666667, 0.089508406455,
            0.11111057, 0.146462614229, 0.16666667, 0.195309105255, 0.23333333, 0.276686558545, 0.3, 0.333640766319,
            0.36666667, 0.400404310407, 0.43333333, 0.5, 0.520818918408, 0.56666667, 0.602196371696, 0.63333333,
            0.667358391486, 0.683573824984, 0.7, 0.73242031601, 0.76666667, 0.83333333, 0.88888943, 0.93333333, 0.97777724,
            1.0])  # (Array): initial structural grid on unit radius
        prob['Rotor%s_max_speed.idx_cylinder_aero'%i] = 3  # (Int): first idx in r_aero_unit of non-cylindrical section, constant twist inboard of here
        prob['Rotor%s_max_speed.idx_cylinder_str'%i] = 14  # (Int): first idx in r_str_unit of non-cylindrical section
        prob['Rotor%s_max_speed.hubFraction'%i] = 0.025  # (Float): hub location as fraction of radius
        # ------------------

        # === blade geometry ===
        prob['Rotor%s_max_speed.r_aero'%i] = np.array([0.02222276, 0.06666667, 0.11111057, 0.2, 0.23333333, 0.3, 0.36666667, 0.43333333,
            0.5, 0.56666667, 0.63333333, 0.64, 0.7, 0.83333333, 0.88888943, 0.93333333,
            0.97777724])  # (Array): new aerodynamic grid on unit radius
        prob['Rotor%s_max_speed.r_max_chord'%i] = 0.23577  # (Float): location of max chord on unit radius
        prob['Rotor%s_max_speed.chord_sub'%i] = np.array([3.2612, 4.5709, 3.3178, 1.4621])  # (Array, m): chord at control points. defined at hub, then at linearly spaced locations from r_max_chord to tip
        # prob['chord_sub'] = np.array([2.2612, 4.5709, 3.3178, 1.4621])
        prob['Rotor%s_max_speed.theta_sub'%i] = np.array([13.2783, 7.46036, 2.89317, -0.0878099])  # (Array, deg): twist at control points.  defined at linearly spaced locations from r[idx_cylinder] to tip
        prob['Rotor%s_max_speed.precurve_sub'%i] = np.array([0.0, 0.0, 0.0])  # (Array, m): precurve at control points.  defined at same locations at chord, starting at 2nd control point (root must be zero precurve)
        prob['Rotor%s_max_speed.delta_precurve_sub'%i] = np.array([0.0, 0.0, 0.0])  # (Array, m): adjustment to precurve to account for curvature from loading
        prob['Rotor%s_max_speed.sparT'%i] = np.array([0.05, 0.047754, 0.045376, 0.031085, 0.0061398])  # (Array, m): spar cap thickness parameters
        prob['Rotor%s_max_speed.teT'%i] = np.array([0.1, 0.09569, 0.06569, 0.02569, 0.00569])  # (Array, m): trailing-edge thickness parameters
        prob['Rotor%s_max_speed.bladeLength'%i] = 61.5  # (Float, m): blade length (if not precurved or swept) otherwise length of blade before curvature
        prob['Rotor%s_max_speed.delta_bladeLength'%i] = 0.0  # (Float, m): adjustment to blade length to account for curvature from loading
        prob['Rotor%s_max_speed.precone'%i] = 2.5  # (Float, deg): precone angle
        prob['Rotor%s_max_speed.tilt'%i] = 5.0  # (Float, deg): shaft tilt
        prob['Rotor%s_max_speed.yaw'%i] = 0.0  # (Float, deg): yaw error
        prob['Rotor%s_max_speed.nBlades'%i] = 3  # (Int): number of blades
        # ------------------

        # === airfoil files ===
        basepath = '/Users/ningrsrch/Dropbox/Programs/RotorSE/src/rotorse/5MW_AFFiles'

        # load all airfoils
        airfoil_types = [0]*8
        airfoil_types[0] = os.path.join(basepath, 'Cylinder1.dat')
        airfoil_types[1] = os.path.join(basepath, 'Cylinder2.dat')
        airfoil_types[2] = os.path.join(basepath, 'DU40_A17.dat')
        airfoil_types[3] = os.path.join(basepath, 'DU35_A17.dat')
        airfoil_types[4] = os.path.join(basepath, 'DU30_A17.dat')
        airfoil_types[5] = os.path.join(basepath, 'DU25_A17.dat')
        airfoil_types[6] = os.path.join(basepath, 'DU21_A17.dat')
        airfoil_types[7] = os.path.join(basepath, 'NACA64_A17.dat')

        # place at appropriate radial stations
        prob['Rotor%s_max_speed.af_idx'%i] = np.array([0, 0, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7])
        prob['Rotor%s_max_speed.af_str_idx'%i] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, \
                5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7])

        n = len(prob['Rotor%s_max_speed.af_idx'%i])
        af = [0]*n
        for j in range(n):
            af[j] = airfoil_types[prob['Rotor%s_max_speed.af_idx'%i][j]]
        prob['Rotor%s_max_speed.airfoil_types'%i] = airfoil_types  # (List): names of airfoil file
        # ----------------------

        # === atmosphere ===
        prob['Rotor%s_max_speed.rho'%i] = 1.225  # (Float, kg/m**3): density of air
        prob['Rotor%s_max_speed.mu'%i] = 1.81206e-5  # (Float, kg/m/s): dynamic viscosity of air
        prob['Rotor%s_max_speed.shearExp'%i] = 0.25  # (Float): shear exponent
        prob['Rotor%s_max_speed.hubHt'%i] = np.array([90.0])  # (Float, m): hub height
        prob['Rotor%s_max_speed.turbine_class'%i] = 'I'  # (Enum): IEC turbine class
        prob['Rotor%s_max_speed.turbulence_class'%i] = 'B'  # (Enum): IEC turbulence class class
        prob['Rotor%s_max_speed.cdf_reference_height_wind_speed'%i] = 90.0  # (Float): reference hub height for IEC wind speed (used in CDF calculation)
        prob['Rotor%s_max_speed.g'%i] = 9.81  # (Float, m/s**2): acceleration of gravity
        # ----------------------

        # === control ===
        prob['Rotor%s_max_speed.control:Vin'%i] = 3.0  # (Float, m/s): cut-in wind speed
        prob['Rotor%s_max_speed.control:Vout'%i] = 25.0  # (Float, m/s): cut-out wind speed
        prob['Rotor%s_max_speed.control:ratedPower'%i] = 5e6  # (Float, W): rated power
        prob['Rotor%s_max_speed.control:minOmega'%i] = 0.0  # (Float, rpm): minimum allowed prob rotation speed
        prob['Rotor%s_max_speed.control:maxOmega'%i] = 12.0  # (Float, rpm): maximum allowed prob rotation speed
        prob['Rotor%s_max_speed.control:tsr'%i] = 7.55  # (Float): tip-speed ratio in Region 2 (should be optimized externally)
        prob['Rotor%s_max_speed.control:pitch'%i] = 0.0  # (Float, deg): pitch angle in region 2 (and region 3 for fixed pitch machines)
        prob['Rotor%s_max_speed.pitch_extreme'%i] = 0.0  # (Float, deg): worst-case pitch at survival wind condition
        prob['Rotor%s_max_speed.azimuth_extreme'%i] = 0.0  # (Float, deg): worst-case azimuth at survival wind condition
        prob['Rotor%s_max_speed.VfactorPC'%i] = 0.7  # (Float): fraction of rated speed at which the deflection is assumed to representative throughout the power curve calculation
        # ----------------------

        # === aero and structural analysis options ===
        prob['Rotor%s_max_speed.nSector'%i] = 4  # (Int): number of sectors to divide prob face into in computing thrust and power
        prob['Rotor%s_max_speed.npts_coarse_power_curve'%i] = 20  # (Int): number of points to evaluate aero analysis at
        prob['Rotor%s_max_speed.npts_spline_power_curve'%i] = 200  # (Int): number of points to use in fitting spline to power curve
        prob['Rotor%s_max_speed.AEP_loss_factor'%i] = 1.0  # (Float): availability and other losses (soiling, array, etc.)
        prob['Rotor%s_max_speed.drivetrainType'%i] = 'geared'  # (Enum)
        prob['Rotor%s_max_speed.nF'%i] = 5  # (Int): number of natural frequencies to compute
        prob['Rotor%s_max_speed.dynamic_amplication_tip_deflection'%i] = 1.35  # (Float): a dynamic amplification factor to adjust the static deflection calculation
        # ----------------------

        # === materials and composite layup  ===
        basepath = '/Users/ningrsrch/Dropbox/Programs/RotorSE/src/rotorse/5MW_PrecompFiles'

        materials = Orthotropic2DMaterial.listFromPreCompFile(os.path.join(basepath, 'materials.inp'))

        ncomp = len(prob['Rotor%s_max_speed.initial_str_grid'%i])
        upper = [0]*ncomp
        lower = [0]*ncomp
        webs = [0]*ncomp
        profile = [0]*ncomp

        prob['Rotor%s_max_speed.leLoc'%i] = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.498, 0.497, 0.465, 0.447, 0.43, 0.411,
            0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4,
            0.4, 0.4, 0.4, 0.4])    # (Array): array of leading-edge positions from a reference blade axis (usually blade pitch axis). locations are normalized by the local chord length. e.g. leLoc[i] = 0.2 means leading edge is 0.2*chord[i] from reference axis.  positive in -x direction for airfoil-aligned coordinate system
        prob['Rotor%s_max_speed.sector_idx_strain_spar'%i] = [2]*ncomp  # (Array): index of sector for spar (PreComp definition of sector)
        prob['Rotor%s_max_speed.sector_idx_strain_te'%i] = [3]*ncomp  # (Array): index of sector for trailing-edge (PreComp definition of sector)
        web1 = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.4114, 0.4102, 0.4094, 0.3876, 0.3755, 0.3639, 0.345, 0.3342, 0.3313, 0.3274, 0.323, 0.3206, 0.3172, 0.3138, 0.3104, 0.307, 0.3003, 0.2982, 0.2935, 0.2899,\
                0.2867, 0.2833, 0.2817, 0.2799, 0.2767, 0.2731, 0.2664, 0.2607, 0.2562, 0.1886, -1.0])
        web2 = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.5886, 0.5868, 0.5854, 0.5508, 0.5315, 0.5131, 0.4831, 0.4658, 0.4687, 0.4726, 0.477, 0.4794, 0.4828, 0.4862, 0.4896, 0.493, 0.4997, 0.5018, 0.5065, 0.5101, \
                0.5133, 0.5167, 0.5183, 0.5201, 0.5233, 0.5269, 0.5336, 0.5393, 0.5438, 0.6114, -1.0])
        web3 = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, \
                1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
        prob['Rotor%s_max_speed.chord_str_ref'%i] = np.array([3.2612, 3.3100915356, 3.32587052924, 3.34159388653, 3.35823798667, 3.37384375335,
            3.38939112914, 3.4774055542, 3.49839685, 3.51343645709, 3.87017220335, 4.04645623801, 4.19408216643,
             4.47641008477, 4.55844487985, 4.57383098262, 4.57285771934, 4.51914315648, 4.47677655262, 4.40075650022,
             4.31069949379, 4.20483735936, 4.08985563932, 3.82931757126, 3.74220276467, 3.54415796922, 3.38732428502,
             3.24931446473, 3.23421422609, 3.22701537997, 3.21972125648, 3.08979310611, 2.95152261813, 2.330753331,
             2.05553464181, 1.82577817774, 1.5860853279, 1.4621])  # (Array, m): chord distribution for reference section, thickness of structural layup scaled with reference thickness
        prob['Rotor%s_max_speed.thick_str_ref'%i] = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.404457084248, 0.404457084248,
                                       0.349012780126, 0.349012780126, 0.349012780126, 0.349012780126, 0.29892003076, 0.29892003076, 0.25110545018, 0.25110545018, 0.25110545018, 0.25110545018,
                                       0.211298863564, 0.211298863564, 0.211298863564, 0.211298863564, 0.17933792591, 0.17933792591, 0.17933792591, 0.17933792591, 0.17933792591, 0.17933792591,
                                       0.17933792591, 0.17933792591])  # (Array, m): airfoil thickness distribution for reference section, thickness of structural layup scaled with reference thickness

        prob['Rotor%s_max_speed.capTriaxThk'%i] = np.array([0.30, 0.29, 0.28, 0.275, 0.27])
        prob['Rotor%s_max_speed.capCarbThk'%i] = np.array([4.2, 2.5, 1.0, 0.90, 0.658])
        prob['Rotor%s_max_speed.tePanelTriaxThk'%i] = np.array([0.30, 0.29, 0.28, 0.275, 0.27])
        prob['Rotor%s_max_speed.tePanelFoamThk'%i] = np.array([9.00, 7.00, 5.00, 3.00, 2.00])

        for j in range(ncomp):

            webLoc = []
            if web1[j] != -1:
                webLoc.append(web1[j])
            if web2[j] != -1:
                webLoc.append(web2[j])
            if web3[j] != -1:
                webLoc.append(web3[j])

            upper[j], lower[j], webs[j] = CompositeSection.initFromPreCompLayupFile(os.path.join(basepath, 'layup_' + str(j+1) + '.inp'), webLoc, materials)
            profile[j] = Profile.initFromPreCompFile(os.path.join(basepath, 'shape_' + str(j+1) + '.inp'))

        prob['Rotor%s_max_speed.materials'%i] = materials  # (List): list of all Orthotropic2DMaterial objects used in defining the geometry
        prob['Rotor%s_max_speed.upperCS'%i] = upper  # (List): list of CompositeSection objections defining the properties for upper surface
        prob['Rotor%s_max_speed.lowerCS'%i] = lower  # (List): list of CompositeSection objections defining the properties for lower surface
        prob['Rotor%s_max_speed.websCS'%i] = webs  # (List): list of CompositeSection objections defining the properties for shear webs
        prob['Rotor%s_max_speed.profile'%i] = profile  # (List): airfoil shape at each radial position
        # --------------------------------------


        # === fatigue ===
        prob['Rotor%s_max_speed.rstar_damage'%i] = np.array([0.000, 0.022, 0.067, 0.111, 0.167, 0.233, 0.300, 0.367, 0.433, 0.500,
            0.567, 0.633, 0.700, 0.767, 0.833, 0.889, 0.933, 0.978])  # (Array): nondimensional radial locations of damage equivalent moments
        prob['Rotor%s_max_speed.Mxb_damage'%i] = 1e3*np.array([2.3743E+003, 2.0834E+003, 1.8108E+003, 1.5705E+003, 1.3104E+003,
            1.0488E+003, 8.2367E+002, 6.3407E+002, 4.7727E+002, 3.4804E+002, 2.4458E+002, 1.6339E+002,
            1.0252E+002, 5.7842E+001, 2.7349E+001, 1.1262E+001, 3.8549E+000, 4.4738E-001])  # (Array, N*m): damage equivalent moments about blade c.s. x-direction
        prob['Rotor%s_max_speed.Myb_damage'%i] = 1e3*np.array([2.7732E+003, 2.8155E+003, 2.6004E+003, 2.3933E+003, 2.1371E+003,
            1.8459E+003, 1.5582E+003, 1.2896E+003, 1.0427E+003, 8.2015E+002, 6.2449E+002, 4.5229E+002,
            3.0658E+002, 1.8746E+002, 9.6475E+001, 4.2677E+001, 1.5409E+001, 1.8426E+000])  # (Array, N*m): damage equivalent moments about blade c.s. y-direction
        prob['Rotor%s_max_speed.strain_ult_spar'%i] = 1.0e-2  # (Float): ultimate strain in spar cap
        prob['Rotor%s_max_speed.strain_ult_te'%i] = 2500*1e-6 * 2   # (Float): uptimate strain in trailing-edge panels, note that I am putting a factor of two for the damage part only.
        prob['Rotor%s_max_speed.eta_damage'%i] = 1.35*1.3*1.0  # (Float): safety factor for fatigue
        prob['Rotor%s_max_speed.m_damage'%i] = 10.0  # (Float): slope of S-N curve for fatigue analysis
        prob['Rotor%s_max_speed.N_damage'%i] = 365*24*3600*20.0  # (Float): number of cycles used in fatigue analysis  TODO: make function of rotation speed
        # ----------------



            # === blade grid ===
        prob['Rotor%s_max_thrust.initial_aero_grid'%i] = np.array([0.02222276, 0.06666667, 0.11111057, 0.16666667, 0.23333333, 0.3, 0.36666667,
            0.43333333, 0.5, 0.56666667, 0.63333333, 0.7, 0.76666667, 0.83333333, 0.88888943, 0.93333333,
            0.97777724])  # (Array): initial aerodynamic grid on unit radius
        prob['Rotor%s_max_thrust.initial_str_grid'%i] = np.array([0.0, 0.00492790457512, 0.00652942887106, 0.00813095316699, 0.00983257273154,
            0.0114340970275, 0.0130356213234, 0.02222276, 0.024446481932, 0.026048006228, 0.06666667, 0.089508406455,
            0.11111057, 0.146462614229, 0.16666667, 0.195309105255, 0.23333333, 0.276686558545, 0.3, 0.333640766319,
            0.36666667, 0.400404310407, 0.43333333, 0.5, 0.520818918408, 0.56666667, 0.602196371696, 0.63333333,
            0.667358391486, 0.683573824984, 0.7, 0.73242031601, 0.76666667, 0.83333333, 0.88888943, 0.93333333, 0.97777724,
            1.0])  # (Array): initial structural grid on unit radius
        prob['Rotor%s_max_thrust.idx_cylinder_aero'%i] = 3  # (Int): first idx in r_aero_unit of non-cylindrical section, constant twist inboard of here
        prob['Rotor%s_max_thrust.idx_cylinder_str'%i] = 14  # (Int): first idx in r_str_unit of non-cylindrical section
        prob['Rotor%s_max_thrust.hubFraction'%i] = 0.025  # (Float): hub location as fraction of radius
        # ------------------

        # === blade geometry ===
        prob['Rotor%s_max_thrust.r_aero'%i] = np.array([0.02222276, 0.06666667, 0.11111057, 0.2, 0.23333333, 0.3, 0.36666667, 0.43333333,
            0.5, 0.56666667, 0.63333333, 0.64, 0.7, 0.83333333, 0.88888943, 0.93333333,
            0.97777724])  # (Array): new aerodynamic grid on unit radius
        prob['Rotor%s_max_thrust.r_max_chord'%i] = 0.23577  # (Float): location of max chord on unit radius
        prob['Rotor%s_max_thrust.chord_sub'%i] = np.array([3.2612, 4.5709, 3.3178, 1.4621])  # (Array, m): chord at control points. defined at hub, then at linearly spaced locations from r_max_chord to tip
        # prob['chord_sub'] = np.array([2.2612, 4.5709, 3.3178, 1.4621])
        prob['Rotor%s_max_thrust.theta_sub'%i] = np.array([13.2783, 7.46036, 2.89317, -0.0878099])  # (Array, deg): twist at control points.  defined at linearly spaced locations from r[idx_cylinder] to tip
        prob['Rotor%s_max_thrust.precurve_sub'%i] = np.array([0.0, 0.0, 0.0])  # (Array, m): precurve at control points.  defined at same locations at chord, starting at 2nd control point (root must be zero precurve)
        prob['Rotor%s_max_thrust.delta_precurve_sub'%i] = np.array([0.0, 0.0, 0.0])  # (Array, m): adjustment to precurve to account for curvature from loading
        prob['Rotor%s_max_thrust.sparT'%i] = np.array([0.05, 0.047754, 0.045376, 0.031085, 0.0061398])  # (Array, m): spar cap thickness parameters
        prob['Rotor%s_max_thrust.teT'%i] = np.array([0.1, 0.09569, 0.06569, 0.02569, 0.00569])  # (Array, m): trailing-edge thickness parameters
        prob['Rotor%s_max_thrust.bladeLength'%i] = 61.5  # (Float, m): blade length (if not precurved or swept) otherwise length of blade before curvature
        prob['Rotor%s_max_thrust.delta_bladeLength'%i] = 0.0  # (Float, m): adjustment to blade length to account for curvature from loading
        prob['Rotor%s_max_thrust.precone'%i] = 2.5  # (Float, deg): precone angle
        prob['Rotor%s_max_thrust.tilt'%i] = 5.0  # (Float, deg): shaft tilt
        prob['Rotor%s_max_thrust.yaw'%i] = 0.0  # (Float, deg): yaw error
        prob['Rotor%s_max_thrust.nBlades'%i] = 3  # (Int): number of blades
        # ------------------

        # === airfoil files ===
        basepath = '/Users/ningrsrch/Dropbox/Programs/RotorSE/src/rotorse/5MW_AFFiles'

        # load all airfoils
        airfoil_types = [0]*8
        airfoil_types[0] = os.path.join(basepath, 'Cylinder1.dat')
        airfoil_types[1] = os.path.join(basepath, 'Cylinder2.dat')
        airfoil_types[2] = os.path.join(basepath, 'DU40_A17.dat')
        airfoil_types[3] = os.path.join(basepath, 'DU35_A17.dat')
        airfoil_types[4] = os.path.join(basepath, 'DU30_A17.dat')
        airfoil_types[5] = os.path.join(basepath, 'DU25_A17.dat')
        airfoil_types[6] = os.path.join(basepath, 'DU21_A17.dat')
        airfoil_types[7] = os.path.join(basepath, 'NACA64_A17.dat')

        # place at appropriate radial stations
        prob['Rotor%s_max_thrust.af_idx'%i] = np.array([0, 0, 1, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7, 7])
        prob['Rotor%s_max_thrust.af_str_idx'%i] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, \
                5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7])

        n = len(prob['Rotor%s_max_thrust.af_idx'%i])
        af = [0]*n
        for j in range(n):
            af[j] = airfoil_types[prob['Rotor%s_max_thrust.af_idx'%i][j]]
        prob['Rotor%s_max_thrust.airfoil_types'%i] = airfoil_types  # (List): names of airfoil file
        # ----------------------

        # === atmosphere ===
        prob['Rotor%s_max_thrust.rho'%i] = 1.225  # (Float, kg/m**3): density of air
        prob['Rotor%s_max_thrust.mu'%i] = 1.81206e-5  # (Float, kg/m/s): dynamic viscosity of air
        prob['Rotor%s_max_thrust.shearExp'%i] = 0.25  # (Float): shear exponent
        prob['Rotor%s_max_thrust.hubHt'%i] = np.array([90.0])  # (Float, m): hub height
        prob['Rotor%s_max_thrust.turbine_class'%i] = 'I'  # (Enum): IEC turbine class
        prob['Rotor%s_max_thrust.turbulence_class'%i] = 'B'  # (Enum): IEC turbulence class class
        prob['Rotor%s_max_thrust.cdf_reference_height_wind_speed'%i] = 90.0  # (Float): reference hub height for IEC wind speed (used in CDF calculation)
        prob['Rotor%s_max_thrust.g'%i] = 9.81  # (Float, m/s**2): acceleration of gravity
        # ----------------------

        # === control ===
        prob['Rotor%s_max_thrust.control:Vin'%i] = 3.0  # (Float, m/s): cut-in wind thrust
        prob['Rotor%s_max_thrust.control:Vout'%i] = 25.0  # (Float, m/s): cut-out wind speed
        prob['Rotor%s_max_thrust.control:ratedPower'%i] = 5e6  # (Float, W): rated power
        prob['Rotor%s_max_thrust.control:minOmega'%i] = 0.0  # (Float, rpm): minimum allowed prob rotation speed
        prob['Rotor%s_max_thrust.control:maxOmega'%i] = 12.0  # (Float, rpm): maximum allowed prob rotation speed
        prob['Rotor%s_max_thrust.control:tsr'%i] = 7.55  # (Float): tip-speed ratio in Region 2 (should be optimized externally)
        prob['Rotor%s_max_thrust.control:pitch'%i] = 0.0  # (Float, deg): pitch angle in region 2 (and region 3 for fixed pitch machines)
        prob['Rotor%s_max_thrust.pitch_extreme'%i] = 0.0  # (Float, deg): worst-case pitch at survival wind condition
        prob['Rotor%s_max_thrust.azimuth_extreme'%i] = 0.0  # (Float, deg): worst-case azimuth at survival wind condition
        prob['Rotor%s_max_thrust.VfactorPC'%i] = 0.7  # (Float): fraction of rated speed at which the deflection is assumed to representative throughout the power curve calculation
        # ----------------------

        # === aero and structural analysis options ===
        prob['Rotor%s_max_thrust.nSector'%i] = 4  # (Int): number of sectors to divide prob face into in computing thrust and power
        prob['Rotor%s_max_thrust.npts_coarse_power_curve'%i] = 20  # (Int): number of points to evaluate aero analysis at
        prob['Rotor%s_max_thrust.npts_spline_power_curve'%i] = 200  # (Int): number of points to use in fitting spline to power curve
        prob['Rotor%s_max_thrust.AEP_loss_factor'%i] = 1.0  # (Float): availability and other losses (soiling, array, etc.)
        prob['Rotor%s_max_thrust.drivetrainType'%i] = 'geared'  # (Enum)
        prob['Rotor%s_max_thrust.nF'%i] = 5  # (Int): number of natural frequencies to compute
        prob['Rotor%s_max_thrust.dynamic_amplication_tip_deflection'%i] = 1.35  # (Float): a dynamic amplification factor to adjust the static deflection calculation
        # ----------------------

        # === materials and composite layup  ===
        basepath = '/Users/ningrsrch/Dropbox/Programs/RotorSE/src/rotorse/5MW_PrecompFiles'

        materials = Orthotropic2DMaterial.listFromPreCompFile(os.path.join(basepath, 'materials.inp'))

        ncomp = len(prob['Rotor%s_max_thrust.initial_str_grid'%i])
        upper = [0]*ncomp
        lower = [0]*ncomp
        webs = [0]*ncomp
        profile = [0]*ncomp

        prob['Rotor%s_max_thrust.leLoc'%i] = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.498, 0.497, 0.465, 0.447, 0.43, 0.411,
            0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4,
            0.4, 0.4, 0.4, 0.4])    # (Array): array of leading-edge positions from a reference blade axis (usually blade pitch axis). locations are normalized by the local chord length. e.g. leLoc[i] = 0.2 means leading edge is 0.2*chord[i] from reference axis.  positive in -x direction for airfoil-aligned coordinate system
        prob['Rotor%s_max_thrust.sector_idx_strain_spar'%i] = [2]*ncomp  # (Array): index of sector for spar (PreComp definition of sector)
        prob['Rotor%s_max_thrust.sector_idx_strain_te'%i] = [3]*ncomp  # (Array): index of sector for trailing-edge (PreComp definition of sector)
        web1 = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.4114, 0.4102, 0.4094, 0.3876, 0.3755, 0.3639, 0.345, 0.3342, 0.3313, 0.3274, 0.323, 0.3206, 0.3172, 0.3138, 0.3104, 0.307, 0.3003, 0.2982, 0.2935, 0.2899,\
                0.2867, 0.2833, 0.2817, 0.2799, 0.2767, 0.2731, 0.2664, 0.2607, 0.2562, 0.1886, -1.0])
        web2 = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.5886, 0.5868, 0.5854, 0.5508, 0.5315, 0.5131, 0.4831, 0.4658, 0.4687, 0.4726, 0.477, 0.4794, 0.4828, 0.4862, 0.4896, 0.493, 0.4997, 0.5018, 0.5065, 0.5101, \
                0.5133, 0.5167, 0.5183, 0.5201, 0.5233, 0.5269, 0.5336, 0.5393, 0.5438, 0.6114, -1.0])
        web3 = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, \
                1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
        prob['Rotor%s_max_thrust.chord_str_ref'%i] = np.array([3.2612, 3.3100915356, 3.32587052924, 3.34159388653, 3.35823798667, 3.37384375335,
            3.38939112914, 3.4774055542, 3.49839685, 3.51343645709, 3.87017220335, 4.04645623801, 4.19408216643,
             4.47641008477, 4.55844487985, 4.57383098262, 4.57285771934, 4.51914315648, 4.47677655262, 4.40075650022,
             4.31069949379, 4.20483735936, 4.08985563932, 3.82931757126, 3.74220276467, 3.54415796922, 3.38732428502,
             3.24931446473, 3.23421422609, 3.22701537997, 3.21972125648, 3.08979310611, 2.95152261813, 2.330753331,
             2.05553464181, 1.82577817774, 1.5860853279, 1.4621])  # (Array, m): chord distribution for reference section, thickness of structural layup scaled with reference thickness
        prob['Rotor%s_max_thrust.thick_str_ref'%i] = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.404457084248, 0.404457084248,
                                       0.349012780126, 0.349012780126, 0.349012780126, 0.349012780126, 0.29892003076, 0.29892003076, 0.25110545018, 0.25110545018, 0.25110545018, 0.25110545018,
                                       0.211298863564, 0.211298863564, 0.211298863564, 0.211298863564, 0.17933792591, 0.17933792591, 0.17933792591, 0.17933792591, 0.17933792591, 0.17933792591,
                                       0.17933792591, 0.17933792591])  # (Array, m): airfoil thickness distribution for reference section, thickness of structural layup scaled with reference thickness

        prob['Rotor%s_max_thrust.capTriaxThk'%i] = np.array([0.30, 0.29, 0.28, 0.275, 0.27])
        prob['Rotor%s_max_thrust.capCarbThk'%i] = np.array([4.2, 2.5, 1.0, 0.90, 0.658])
        prob['Rotor%s_max_thrust.tePanelTriaxThk'%i] = np.array([0.30, 0.29, 0.28, 0.275, 0.27])
        prob['Rotor%s_max_thrust.tePanelFoamThk'%i] = np.array([9.00, 7.00, 5.00, 3.00, 2.00])

        for j in range(ncomp):

            webLoc = []
            if web1[j] != -1:
                webLoc.append(web1[j])
            if web2[j] != -1:
                webLoc.append(web2[j])
            if web3[j] != -1:
                webLoc.append(web3[j])

            upper[j], lower[j], webs[j] = CompositeSection.initFromPreCompLayupFile(os.path.join(basepath, 'layup_' + str(j+1) + '.inp'), webLoc, materials)
            profile[j] = Profile.initFromPreCompFile(os.path.join(basepath, 'shape_' + str(j+1) + '.inp'))

        prob['Rotor%s_max_thrust.materials'%i] = materials  # (List): list of all Orthotropic2DMaterial objects used in defining the geometry
        prob['Rotor%s_max_thrust.upperCS'%i] = upper  # (List): list of CompositeSection objections defining the properties for upper surface
        prob['Rotor%s_max_thrust.lowerCS'%i] = lower  # (List): list of CompositeSection objections defining the properties for lower surface
        prob['Rotor%s_max_thrust.websCS'%i] = webs  # (List): list of CompositeSection objections defining the properties for shear webs
        prob['Rotor%s_max_thrust.profile'%i] = profile  # (List): airfoil shape at each radial position
        # --------------------------------------


        # === fatigue ===
        prob['Rotor%s_max_thrust.rstar_damage'%i] = np.array([0.000, 0.022, 0.067, 0.111, 0.167, 0.233, 0.300, 0.367, 0.433, 0.500,
            0.567, 0.633, 0.700, 0.767, 0.833, 0.889, 0.933, 0.978])  # (Array): nondimensional radial locations of damage equivalent moments
        prob['Rotor%s_max_thrust.Mxb_damage'%i] = 1e3*np.array([2.3743E+003, 2.0834E+003, 1.8108E+003, 1.5705E+003, 1.3104E+003,
            1.0488E+003, 8.2367E+002, 6.3407E+002, 4.7727E+002, 3.4804E+002, 2.4458E+002, 1.6339E+002,
            1.0252E+002, 5.7842E+001, 2.7349E+001, 1.1262E+001, 3.8549E+000, 4.4738E-001])  # (Array, N*m): damage equivalent moments about blade c.s. x-direction
        prob['Rotor%s_max_thrust.Myb_damage'%i] = 1e3*np.array([2.7732E+003, 2.8155E+003, 2.6004E+003, 2.3933E+003, 2.1371E+003,
            1.8459E+003, 1.5582E+003, 1.2896E+003, 1.0427E+003, 8.2015E+002, 6.2449E+002, 4.5229E+002,
            3.0658E+002, 1.8746E+002, 9.6475E+001, 4.2677E+001, 1.5409E+001, 1.8426E+000])  # (Array, N*m): damage equivalent moments about blade c.s. y-direction
        prob['Rotor%s_max_thrust.strain_ult_spar'%i] = 1.0e-2  # (Float): ultimate strain in spar cap
        prob['Rotor%s_max_thrust.strain_ult_te'%i] = 2500*1e-6 * 2   # (Float): uptimate strain in trailing-edge panels, note that I am putting a factor of two for the damage part only.
        prob['Rotor%s_max_thrust.eta_damage'%i] = 1.35*1.3*1.0  # (Float): safety factor for fatigue
        prob['Rotor%s_max_thrust.m_damage'%i] = 10.0  # (Float): slope of S-N curve for fatigue analysis
        prob['Rotor%s_max_thrust.N_damage'%i] = 365*24*3600*20.0  # (Float): number of cycles used in fatigue analysis  TODO: make function of rotation speed
        # ----------------


    prob.run()

    print 'AEP: ', prob['AEP']
    print 'COE: ', prob['COE']
    print 'BOS: ', prob['BOS']
    print 'rotor_cost: ', prob['nrel_csm_tcc_20150.rotor_cost']
    print 'nacelle_cost: ', prob['nrel_csm_tcc_20150.nacelle_cost']
    print 'nacelle_mass: ', prob['nrel_csm_tcc_20150.nacelle_mass']
    print 'rotor_mass: ', prob['nrel_csm_tcc_20150.rotor_mass']
    print prob['nrel_csm_tcc_20150.rotor_mass']+prob['nrel_csm_tcc_20150.nacelle_mass']
    # print 'comp rated power: ', prob['getRating.ratedPower']
    # print 'rated power: ', prob['ratedPower']
    # print 'rotorDiameter: ', prob['rotorDiameter']
    # print 'rotorCost: ', prob['rotorCost']
    # for i in range(nTurbs):
    #     print 'rotor diameters: ', prob['rotor_diameters%s'%i]
    # print 'wtPower0: ', prob['wtPower0']
    # print 'windSpeeds0: ', prob['windSpeeds'][:,0]
    # print 'rotorCost: ', prob['rotorCost']

    #print 'TurbineX: ', prob['turbineX']
    #print 'TurbineY: ', prob['turbineY']
    # print 'TurbineZ: ', prob['turbineZ'][0]
    #print 'Spacing Constraints: ', prob['sc']
    #print 'Boundary Constraints:', prob['boundaryDistances']
    # for i in range(nGroups):
    #     print 'Max Thrust Shell Buckling: ', prob['Tower%s_max_thrust.shell_buckling'%i]
    #     print 'Max Speed Shell Buckling: ', prob['Tower%s_max_speed.shell_buckling'%i]
    #     print 'Max Thrust Frequecty: ', prob['Tower%s_max_thrust.freq'%i]
    #     print 'Max Speed Frequency: ', prob['Tower%s_max_speed.freq'%i]
    #
    # print 'Max thrust Fx: ', prob['Loads0.Fx1']
    # print 'Max thrust Fy: ', prob['Loads0.Fy1']
    # print 'Max thrust Fz: ', prob['Loads0.Fz1']
    # print 'Max speed Fx: ', prob['Loads0.Fx2']
    # print 'Max speed Fy: ', prob['Loads0.Fy2']
    # print 'Max speed Fz: ', prob['Loads0.Fz2']
    #
    # print 'Max thrust Mxx: ', prob['Loads0.Mxx1']
    # print 'Max thrust Myy: ', prob['Loads0.Myy1']
    # print 'Max thrust Mzz: ', prob['Loads0.Mzz1']
    # print 'Max speed Mxx: ', prob['Loads0.Mxx2']
    # print 'Max speed Myy: ', prob['Loads0.Myy2']
    # print 'Max speed Mzz: ', prob['Loads0.Mzz2']
    #
    # print 'mIzz: ', prob['Loads%s.mIzz'%i]
    # print 'm: ', prob['Loads%s.m'%i]




    #print 'BOS Cost: ', prob['BOS']/125000.
    #print 'cost: ', prob['cost']/125000.
