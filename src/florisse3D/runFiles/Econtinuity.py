import numpy as np
from openmdao.api import Group, Problem, IndepVarComp, pyOptSparseDriver
# from FLORISSE3D.setupOptimization import *
from setupOptimization import *
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
        NREL5MWCPCT = pickle.load(open('doc/tune/NREL5MWCPCT_smooth_dict.p'))
        # print(NREL5MWCPCT)
        # NREL5MWCPCT = pickle.Unpickler(open('./input_files/NREL5MWCPCT.p')).load()
        datasize = NREL5MWCPCT['CP'].size
    else:
        datasize = 0

    rotor_diameter = 126.4

    nRows = 2
    nTurbs = nRows**2
    minSpacing = 2
    nRows = 2
    nTurbs = nRows**2
    spacing = 4.0  # turbine grid spacing in diameters
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

    rotorDiameter = np.zeros(nTurbs)
    axialInduction = np.zeros(nTurbs)
    Ct = np.zeros(nTurbs)
    Cp = np.zeros(nTurbs)
    generatorEfficiency = np.zeros(nTurbs)

    # define initial values
    for turbI in range(0, nTurbs):
        rotorDiameter[turbI] = rotor_diameter            # m
        axialInduction[turbI] = 1.0/3.0
        Ct[turbI] = 4.0*axialInduction[turbI]*(1.0-axialInduction[turbI])
        Cp[turbI] = 0.7737 * 4.0 * 1.0/3.0 * np.power((1. - 1.0/3.0), 2)
        generatorEfficiency[turbI] = 1.0#0.944

    """Define wind flow"""
    air_density = 1.1716    # kg/m^3

    windSpeeds, windFrequencies, windDirections, nDirections = amaliaWind()
    nDirections = 2
    windSpeeds = np.ones(nDirections)*10.
    windFrequencies = np.ones(nDirections)/nDirections
    # windDirections = np.linspace(0.,360.-360./nDirections,nDirections)
    windDirections = np.array([0.,90.])

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

    shearExp = 0.1

    opt_filenameXYZ = 'src/florisse3D/runFiles/rand_XYZ.txt'
    opt_filename_d = 'src/florisse3D/runFiles/rand_d.txt'
    opt_filename_t = 'src/florisse3D/runFiles/rand_t.txt'
    opt_filename_yaw = 'src/florisse3D/runFiles/rand_yaw.txt'

    optXYZ = open(opt_filenameXYZ)
    optimizedXYZ = np.loadtxt(optXYZ)
    turbineX = optimizedXYZ[:,0]
    turbineY = optimizedXYZ[:,1]
    turbineZ = optimizedXYZ[:,2]
    turbineXorig = optimizedXYZ[:,0]
    turbineYorig = optimizedXYZ[:,1]
    turbineZorig = optimizedXYZ[:,2]

    opt_d = open(opt_filename_d)
    optimized_d = np.loadtxt(opt_d)
    d_paramorig = np.zeros((nTurbs,3))
    d_param = np.zeros((nTurbs,3))
    for i in range(nTurbs):
        d_param[i] = optimized_d[i]
        d_paramorig[i] = optimized_d[i]

    opt_t = open(opt_filename_t)
    optimized_t = np.loadtxt(opt_t)
    t_param = np.zeros((nTurbs,3))
    for i in range(nTurbs):
        t_param[i] = optimized_t[i]

    opt_yaw = open(opt_filename_yaw)
    optimizedYaw = np.loadtxt(opt_yaw)
    yaw = np.zeros((nDirections, nTurbs))
    for j in range(nDirections):
        yaw[j] = optimizedYaw[j]

    p = 200
    COE = np.zeros((nTurbs,p))
    # SC1 = np.zeros((nTurbs,p))
    # SC2 = np.zeros((nTurbs,p))
    # SC3 = np.zeros((nTurbs,p))
    # SC4 = np.zeros((nTurbs,p))
    # SC5 = np.zeros((nTurbs,p))
    # SC6 = np.zeros((nTurbs,p))
    # BC = np.zeros((nTurbs,p,6))
    num = np.linspace(-1.,1.,p)

    for k in range(p):
        for o in range(nTurbs):
            for l in range(nTurbs):
                turbineY[l] = turbineYorig[l]
            turbineY[o] += num[k]
            print k


            prob = Problem()
            root = prob.root = Group()

            for i in range(nTurbs):
                root.add('d_param%s'%i, IndepVarComp('d_param%s'%i, d_param[i]), promotes=['*'])
                root.add('t_param%s'%i, IndepVarComp('t_param%s'%i, t_param[i]), promotes=['*'])
                root.add('get_z_param%s'%i, get_z(nPoints))
                root.add('get_z_full%s'%i, get_z(nFull))
                root.add('Tower%s_max_thrust'%i, Tower(nPoints, nFull), promotes=['L_reinforced','m','mrhox','E','sigma_y','gamma_f','gamma_b','rhoAir','z0','zref','shearExp','rho'])
                root.add('Tower%s_max_speed'%i, Tower(nPoints, nFull), promotes=['L_reinforced','m','mrhox','E','sigma_y','gamma_f','gamma_b','rhoAir','z0','zref','shearExp','rho'])
            root.add('Zs', DeMUX(nTurbs, units='m'))
            root.add('AEPGroup', AEPGroup(nTurbs, nDirections=nDirections,
                        use_rotor_components=use_rotor_components, datasize=datasize, differentiable=True,
                        optimizingLayout=False, nSamples=0), promotes=['*'])
            root.add('COEGroup', COEGroup(nTurbs), promotes=['*'])
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

            # ----------------------

            prob.root.ln_solver.options['single_voi_relevance_reduction'] = True

            prob.setup(check=True)

            prob['turbineX'] = turbineX
            prob['turbineY'] = turbineY
            prob['turbineZ'] = turbineZ

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

            COE[o][k] = prob['COE']
            # print prob['sc']
            # SC1[o][k] = prob['sc'][0]
            # SC2[o][k] = prob['sc'][1]
            # SC3[o][k] = prob['sc'][2]
            # SC4[o][k] = prob['sc'][3]
            # SC5[o][k] = prob['sc'][4]
            # SC6[o][k] = prob['sc'][5]
            # BC[o][k] = prob['boundaryDistances']

    x = np.linspace(-0.01,0.01,p)
    for i in range(nTurbs):
        plt.figure(i)
        plt.plot(x,COE[i])
    plt.show()


    print 'turbineZ: ', prob['turbineZ']
    print 'COE: ', COE
