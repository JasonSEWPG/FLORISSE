import numpy as np
from openmdao.api import Group, Problem, IndepVarComp, pyOptSparseDriver
from FLORISSE3D.setupOptimization import *
from FLORISSE3D.simpleTower import Tower
from FLORISSE3D.GeneralWindFarmComponents import calculate_boundary, SpacingComp,\
            BoundaryComp, get_z, getTurbineZ, AEPobj, DeMUX, hGroups, randomStart
from FLORISSE3D.COE import COEGroup
from FLORISSE3D.floris import AEPGroup
import cPickle as pickle
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.patches import Circle
from sys import argv
import os
import random


if __name__ == '__main__':
    shearExp = np.linspace(0.08,0.3,23)
    nGroups = 1

    use_rotor_components = False

    datasize = 0
    rotor_diameter = 126.4

    nRows = 5
    nTurbs = nRows**2

    turbineX = np.array([
5.600000001110121275e+02,1.120000000222024482e+03,1.680000000333036496e+03,2.240000000444048965e+03,
2.800000000555060979e+03,5.600000001110121275e+02,1.120000000222024482e+03,1.680000000333036496e+03,
2.240000000444048965e+03,2.800000000555060979e+03,5.600000001110121275e+02,1.120000000222024482e+03,
1.680000000333036496e+03,2.240000000444048965e+03,2.800000000555060979e+03,5.600000001110121275e+02,
1.120000000222024482e+03,1.680000000333036496e+03,2.240000000444048965e+03,2.800000000555060979e+03,
5.600000001110121275e+02,1.120000000222024482e+03,1.680000000333036496e+03,2.240000000444048965e+03,
2.800000000555060979e+03])

    turbineY = np.array([
5.600000001110121275e+02,5.600000001110121275e+02,5.600000001110121275e+02,5.600000001110121275e+02,
5.600000001110121275e+02,1.120000000222024482e+03,1.120000000222024482e+03,1.120000000222024482e+03,
1.120000000222024482e+03,1.120000000222024482e+03,1.680000000333036496e+03,1.680000000333036496e+03,
1.680000000333036496e+03,1.680000000333036496e+03,1.680000000333036496e+03,2.240000000444048965e+03,
2.240000000444048965e+03,2.240000000444048965e+03,2.240000000444048965e+03,2.240000000444048965e+03,
2.800000000555060979e+03,2.800000000555060979e+03,2.800000000555060979e+03,2.800000000555060979e+03,
2.800000000555060979e+03])

    d = np.zeros((len(shearExp),3))
    t = np.zeros((len(shearExp),3))
    z = np.zeros((len(shearExp),1))

    d[0] =  np.array([4.566088431787840740e+00, 4.479376115755802523e+00, 3.870000000000000107e+00])
    d[1] =  np.array([4.566618467326201802e+00, 4.479997009116052809e+00, 3.870000000000000107e+00])
    d[2] =  np.array([4.562836202555482501e+00, 4.476514393894321486e+00, 3.870000000000000107e+00])
    d[3] =  np.array([4.647700167219415590e+00, 4.564737563392125885e+00, 3.870000000000000107e+00])
    d[4] =  np.array([4.780285134301347050e+00, 4.701791289100897941e+00, 3.870000000000000107e+00])
    d[5] =  np.array([4.902660849434010260e+00, 4.828120752058381093e+00, 3.870000000000000107e+00])
    d[6] =  np.array([5.011228611635859309e+00, 4.821208039780066734e+00, 3.870000000000000107e+00])
    d[7] =  np.array([5.361963334139677073e+00, 4.966001813097318696e+00, 3.870000000000000107e+00])
    d[8] =  np.array([6.299999999999999822e+00, 4.602959146767293674e+00, 3.870000000000000107e+00])
    d[9] =  np.array([6.299999999999999822e+00, 4.877019830801411437e+00, 3.870000000000000107e+00])
    d[10] = np.array([6.299999999999999822e+00, 5.045256971792500877e+00, 3.870000000000000107e+00])
    d[11] = np.array([6.299999999999999822e+00, 5.065778416382796578e+00, 3.870000000000000107e+00])
    d[12] = np.array([6.299999999999999822e+00, 5.130427847060348157e+00, 3.870000000000000107e+00])
    d[13] = np.array([6.299999999999999822e+00, 5.173450799588907856e+00, 3.870000000000000107e+00])
    d[14] = np.array([6.299999999999999822e+00, 5.186566902672471890e+00, 3.870000000000000107e+00])
    d[15] = np.array([6.299999999999999822e+00, 5.176421303868232293e+00, 3.870000000000000107e+00])
    d[16] = np.array([6.299999999999999822e+00, 5.165236508622608191e+00, 3.870000000000000107e+00])
    d[17] = np.array([6.299999999999999822e+00, 5.187958590934716341e+00, 3.870000000000000107e+00])
    d[18] = np.array([6.299999999999999822e+00, 5.348430594971038232e+00, 3.870000000000000107e+00])
    d[19] = np.array([6.299999999999999822e+00, 5.496458160856995789e+00, 3.870000000000000107e+00])
    d[20] = np.array([6.299999999999999822e+00, 5.547351077999691960e+00, 3.870000000000000107e+00])
    d[21] = np.array([6.299999999999999822e+00, 5.607833315820727194e+00, 3.870000000000000107e+00])
    d[22] = np.array([6.299999999999999822e+00, 5.758897734770048515e+00, 3.870000000000000107e+00])

    t[0] =  np.array([2.688832721544889837e-02, 1.729265282430157008e-02, 1.096066201490652862e-02])
    t[1] =  np.array([2.688396857566788256e-02, 1.729077907191182636e-02, 1.096067609051564612e-02])
    t[2] =  np.array([2.690881353711879406e-02, 1.730076268166403849e-02, 1.096069030177522149e-02])
    t[3] =  np.array([2.726670208716559102e-02, 1.749930984414065430e-02, 1.096072475728285082e-02])
    t[4] =  np.array([2.779907862721419168e-02, 1.779990309502907722e-02, 1.096077559756285930e-02])
    t[5] =  np.array([2.835554576483724151e-02, 1.811051728292026319e-02, 1.096083072570461919e-02])
    t[6] =  np.array([2.872114893860838389e-02, 1.856481350426380725e-02, 1.096087681273485645e-02])
    t[7] =  np.array([2.758320398517542793e-02, 1.851562329627830586e-02, 1.096092443709947330e-02])
    t[8] =  np.array([2.411126967351377021e-02, 2.051912989849845673e-02, 1.096097636988518226e-02])
    t[9] =  np.array([2.468788674163023555e-02, 1.959627543920989742e-02, 1.096102570912263138e-02])
    t[10] = np.array([2.506297823855952941e-02, 1.907483965005110638e-02, 1.096106591539156710e-02])
    t[11] = np.array([2.513221638813217459e-02, 1.900428890732258624e-02, 1.096110026837787861e-02])
    t[12] = np.array([2.529707715228515091e-02, 1.883838778693672222e-02, 1.096108590989716518e-02])
    t[13] = np.array([2.542985629619987087e-02, 1.873048763185278937e-02, 1.096117814306560648e-02])
    t[14] = np.array([2.549152966352279809e-02, 1.868861547868169715e-02, 1.096118057610406121e-02])
    t[15] = np.array([2.552047693340249446e-02, 1.871505064419442046e-02, 1.096107578266107205e-02])
    t[16] = np.array([2.554145712548991365e-02, 1.873693610248957156e-02, 1.096128755057212446e-02])
    t[17] = np.array([2.563360978429822870e-02, 1.870332390948563003e-02, 1.096118568150045679e-02])
    t[18] = np.array([2.600352601104230846e-02, 1.843545619527833801e-02, 1.096138183270227787e-02])
    t[19] = np.array([2.634416307738180613e-02, 1.821735089388886245e-02, 1.096110192717250457e-02])
    t[20] = np.array([2.650271659642950903e-02, 1.815247633947810954e-02, 1.096143943255061502e-02])
    t[21] = np.array([2.668472848739468495e-02, 1.806834918374813220e-02, 1.096151846825895934e-02])
    t[22] = np.array([2.707989784460341706e-02, 1.791931798902713507e-02, 1.096161314955146522e-02])

    z[0] =  np.array([7.320000000000000284e+01])
    z[1] =  np.array([7.320000000000000284e+01])
    z[2] =  np.array([7.320000000000000284e+01])
    z[3] =  np.array([7.674979713164134409e+01])
    z[4] =  np.array([8.242538044858360990e+01])
    z[5] =  np.array([8.815796619033645243e+01])
    z[6] =  np.array([9.169143545446904398e+01])
    z[7] =  np.array([9.489570084305380249e+01])
    z[8] =  np.array([9.824260896593891346e+01])
    z[9] =  np.array([1.001042063880765483e+02])
    z[10] = np.array([1.012688368775444019e+02])
    z[11] = np.array([1.014195441699747988e+02])
    z[12] = np.array([1.018949892088543834e+02])
    z[13] = np.array([1.022247930359628612e+02])
    z[14] = np.array([1.023343509828825972e+02])
    z[15] = np.array([1.022882824925131757e+02])
    z[16] = np.array([1.022300879141511842e+02])
    z[17] = np.array([1.024289592049737507e+02])
    z[18] = np.array([1.036638913983497190e+02])
    z[19] = np.array([1.048181077606238745e+02])
    z[20] = np.array([1.052431875613115011e+02])
    z[21] = np.array([1.057412141890418269e+02])
    z[22] = np.array([1.069872098012252621e+02])

    """Define wind flow"""
    air_density = 1.1716    # kg/m^3

    windData = "Amalia"

    """Amalia Wind Arrays"""
    if windData == "Amalia":
        windSpeeds, windFrequencies, windDirections, nDirections = amaliaWind()

    rotorDiameter = np.zeros(nTurbs)
    axialInduction = np.zeros(nTurbs)
    Ct = np.zeros(nTurbs)
    Cp = np.zeros(nTurbs)
    generatorEfficiency = np.zeros(nTurbs)
    yaw = np.zeros((nDirections, nTurbs))

    # define initial values
    for turbI in range(0, nTurbs):
        rotorDiameter[turbI] = rotor_diameter            # m
        axialInduction[turbI] = 1.0/3.0
        Ct[turbI] = 4.0*axialInduction[turbI]*(1.0-axialInduction[turbI])
        Cp[turbI] = 0.7737 * 4.0 * 1.0/3.0 * np.power((1. - 1.0/3.0), 2)
        generatorEfficiency[turbI] = 1.0#0.944

    """Define tower structural properties"""
    # --- geometry ---
    n = 15

    L_reinforced, Toweryaw, E, G, rho, sigma_y, kidx, kx, ky, kz, ktx, kty, ktz, nK, \
                midx, m, mIxx, mIyy, mIzz, mIxy, mIxz, mIyz, mrhox, mrhoy, mrhoz, nMass, \
                addGravityLoadForExtraMass, wind_zref, wind_z0, shearExp, wind_Uref1, \
                plidx1, Fx1, Fy1, Fz1, Mxx1, Myy1, Mzz1, nPL, wind_Uref2, \
                plidx2, Fx2, Fy2, Fz2, Mxx2, Myy2, Mzz2, gamma_f, gamma_m, gamma_n, \
                gamma_b, z_DEL, M_DEL, nDEL, gamma_fatigue, life, m_SN, min_d_to_t, min_taper \
                = setupTower(n)

    shearExp = np.linspace(0.08,0.3,23)

    nPoints = 3
    nFull = n
    rhoAir = air_density

    """OpenMDAO"""

    COE = np.zeros(23)
    AEP = np.zeros(23)
    idealAEP = np.zeros(23)
    cost = np.zeros(23)
    tower_cost = np.zeros(23)

    prob = Problem()
    root = prob.root = Group()

    for i in range(nGroups):
        root.add('d_param%s'%i, IndepVarComp('d_param%s'%i, d[0]), promotes=['*'])
        root.add('t_param%s'%i, IndepVarComp('t_param%s'%i, t[0]), promotes=['*'])
        root.add('get_z_param%s'%i, get_z(nPoints))
        root.add('get_z_full%s'%i, get_z(nFull))
        root.add('Tower%s_max_thrust'%i, Tower(nPoints, nFull), promotes=['L_reinforced','m','mrhox','E','sigma_y','gamma_f','gamma_b','rhoAir','z0','zref','shearExp','rho'])
        root.add('Tower%s_max_speed'%i, Tower(nPoints, nFull), promotes=['L_reinforced','m','mrhox','E','sigma_y','gamma_f','gamma_b','rhoAir','z0','zref','shearExp','rho'])

        root.add('turbineH%s'%i, IndepVarComp('turbineH%s'%i, float(z[0])), promotes=['*'])

    root.add('Zs', DeMUX(nTurbs))
    root.add('hGroups', hGroups(nTurbs), promotes=['*'])
    root.add('AEPGroup', AEPGroup(nTurbs, nDirections=nDirections,
                use_rotor_components=use_rotor_components, datasize=datasize, differentiable=True,
                optimizingLayout=False, nSamples=0), promotes=['*'])
    root.add('COEGroup', COEGroup(nTurbs, nGroups), promotes=['*'])

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

    # ----------------------

    prob.root.ln_solver.options['single_voi_relevance_reduction'] = True

    prob.setup(check=True)

    for i in range(nDirections):
        prob['yaw%s'%i] = yaw[i]
    prob['nGroups'] = nGroups
    prob['turbineX'] = turbineX
    prob['turbineY'] = turbineY

    prob['ratedPower'] = np.ones(nTurbs)*5000. # in kw

    # assign values to constant inputs (not design variables)
    prob['rotorDiameter'] = rotorDiameter
    prob['rotor_diameter'] = rotor_diameter
    prob['axialInduction'] = axialInduction
    prob['generatorEfficiency'] = generatorEfficiency
    prob['air_density'] = air_density
    prob['windDirections'] = windDirections
    prob['windFrequencies'] = windFrequencies
    prob['Uref'] = windSpeeds

    prob['Ct_in'] = Ct
    prob['Cp_in'] = Cp
    prob['floris_params:cos_spread'] = 1E12 # turns off cosine spread (just needs to be very large)

    prob['L_reinforced'] = L_reinforced
    prob['rho'] = rho
    prob['E'] = E
    prob['gamma_f'] = gamma_f
    prob['gamma_b'] = gamma_b
    prob['sigma_y'] = sigma_y
    prob['m'] = m
    prob['mrhox'] = mrhox
    prob['zref'] = 50.
    prob['z0'] = 0.

    for i in range(nGroups):
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

    for k in range(23):
        prob['shearExp'] = shearExp[k]
        prob['d_param0'] = d[k]
        prob['t_param0'] = t[k]
        prob['turbineH0'] = z[k]
        prob.run()
        COE[k] = prob['COE']
        AEP[k] = prob['AEP']
        cost[k] = prob['farm_cost']
        tower_cost[k] = prob['tower_cost']



    nGroups = 1
    prob = Problem()
    root = prob.root = Group()

    for i in range(nGroups):
        root.add('d_param%s'%i, IndepVarComp('d_param%s'%i, d[0]), promotes=['*'])
        root.add('t_param%s'%i, IndepVarComp('t_param%s'%i, t[0]), promotes=['*'])
        root.add('get_z_param%s'%i, get_z(nPoints))
        root.add('get_z_full%s'%i, get_z(nFull))
        root.add('Tower%s_max_thrust'%i, Tower(nPoints, nFull), promotes=['L_reinforced','m','mrhox','E','sigma_y','gamma_f','gamma_b','rhoAir','z0','zref','shearExp','rho'])
        root.add('Tower%s_max_speed'%i, Tower(nPoints, nFull), promotes=['L_reinforced','m','mrhox','E','sigma_y','gamma_f','gamma_b','rhoAir','z0','zref','shearExp','rho'])

        root.add('turbineH%s'%i, IndepVarComp('turbineH%s'%i, float(z[0])), promotes=['*'])

    root.add('Zs', DeMUX(1))
    root.add('hGroups', hGroups(1), promotes=['*'])
    root.add('AEPGroup', AEPGroup(1, nDirections=nDirections,
                use_rotor_components=use_rotor_components, datasize=datasize, differentiable=True,
                optimizingLayout=False, nSamples=0), promotes=['*'])
    root.add('COEGroup', COEGroup(1, nGroups), promotes=['*'])

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

    # ----------------------

    prob.root.ln_solver.options['single_voi_relevance_reduction'] = True

    prob.setup(check=True)

    for i in range(nDirections):
        prob['yaw%s'%i] = np.array([0.])
    prob['nGroups'] = nGroups
    prob['turbineX'] = np.array([0.])
    prob['turbineY'] = np.array([0.])

    prob['ratedPower'] = np.ones(1)*5000. # in kw

    # assign values to constant inputs (not design variables)
    prob['rotorDiameter'] = np.ones(1)*rotor_diameter
    prob['rotor_diameter'] = rotor_diameter
    prob['axialInduction'] = axialInduction[0]
    prob['generatorEfficiency'] = generatorEfficiency[0]
    prob['air_density'] = air_density
    prob['windDirections'] = windDirections
    prob['windFrequencies'] = windFrequencies
    prob['Uref'] = windSpeeds

    prob['Ct_in'] = Ct[0]
    prob['Cp_in'] = Cp[0]
    prob['floris_params:cos_spread'] = 1E12 # turns off cosine spread (just needs to be very large)

    prob['L_reinforced'] = L_reinforced
    prob['rho'] = rho
    prob['E'] = E
    prob['gamma_f'] = gamma_f
    prob['gamma_b'] = gamma_b
    prob['sigma_y'] = sigma_y
    prob['m'] = m
    prob['mrhox'] = mrhox
    prob['zref'] = 50.
    prob['z0'] = 0.

    for i in range(nGroups):
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

    for k in range(23):
        prob['shearExp'] = shearExp[k]
        prob['d_param0'] = d[k]
        prob['t_param0'] = t[k]
        prob['turbineH0'] = z[k]
        prob.run()
        idealAEP[k] = prob['AEP']*25.

    print 'ideal AEP: ', repr(idealAEP)
    print 'AEP: ', repr(AEP)
    print 'COE: ', repr(COE)
    print 'cost: ', repr(cost)
    print 'tower cost: ', repr(tower_cost)

    print 'wake loss: ', repr((idealAEP-AEP)/idealAEP*100.)

    # ideal AEP:  array([  3.93091892e+08,   3.97612765e+08,   4.02185631e+08,
    #          4.13218364e+08,   4.29454316e+08,   4.47525653e+08,
    #          4.62780350e+08,   4.78617603e+08,   4.94965234e+08,
    #          5.08975810e+08,   5.21664959e+08,   5.32127750e+08,
    #          5.43345446e+08,   5.54328823e+08,   5.64642120e+08,
    #          5.74433908e+08,   5.83332927e+08,   5.92700614e+08,
    #          6.05661946e+08,   6.19020995e+08,   6.30429051e+08,
    #          6.42516955e+08,   6.58068241e+08])
    # AEP:  array([  3.24409104e+08,   3.28140069e+08,   3.31913944e+08,
    #          3.41018988e+08,   3.54418121e+08,   3.69331953e+08,
    #          3.81921280e+08,   3.94991377e+08,   4.09007117e+08,
    #          4.21041442e+08,   4.32072997e+08,   4.41209191e+08,
    #          4.51242063e+08,   4.61163519e+08,   4.70400137e+08,
    #          4.79247924e+08,   4.87777952e+08,   4.97073418e+08,
    #          5.09888263e+08,   5.23017755e+08,   5.34113990e+08,
    #          5.45604584e+08,   5.60257541e+08])
    # COE:  array([ 31.18236051,  30.89726889,  30.61519804,  30.36056272,
    #         30.0784872 ,  29.76252193,  29.40934475,  29.01725747,
    #         28.61521826,  28.18625921,  27.7586714 ,  27.32819729,
    #         26.91530558,  26.50939635,  26.12238675,  25.75095411,
    #         25.40284878,  25.06939004,  24.74364341,  24.41726963,
    #         24.09138189,  23.77427542,  23.46733418])
    # cost:  array([  1.01158416e+10,   1.01386319e+10,   1.01616111e+10,
    #          1.03535284e+10,   1.06603609e+10,   1.09922503e+10,
    #          1.12320546e+10,   1.14615665e+10,   1.17038279e+10,
    #          1.18675832e+10,   1.19937723e+10,   1.20574518e+10,
    #          1.21453180e+10,   1.22251665e+10,   1.22879743e+10,
    #          1.23410913e+10,   1.23909495e+10,   1.24613274e+10,
    #          1.26164934e+10,   1.27706655e+10,   1.28675441e+10,
    #          1.29713536e+10,   1.31477509e+10])
    # tower cost:  array([ 11933787.08043623,  11933744.2786329 ,  11933167.0172122 ,
    #         12849376.81660994,  14365744.16650327,  15962886.64775567,
    #         17002013.5134667 ,  17987272.81336023,  19056091.06359904,
    #         19673263.49759668,  20076430.16466942,  20130424.72055979,
    #         20313718.90614374,  20446701.34266414,  20490764.33956183,
    #         20484891.98752275,  20469682.42204927,  20564359.03825301,
    #         21097781.63224484,  21611702.11209247,  21814422.43086967,
    #         22048512.75370747,  22654583.30699107])
    # wake loss:  array([ 17.47245106,  17.47245106,  17.47245106,  17.47245106,
    #         17.47245106,  17.47245106,  17.47245106,  17.47245106,
    #         17.36649588,  17.27672845,  17.17423434,  17.08585189,
    #         16.95116508,  16.80686612,  16.69056907,  16.57039787,
    #         16.38086439,  16.13414831,  15.81305924,  15.50888275,
    #         15.27770032,  15.08323959,  14.86330657])
