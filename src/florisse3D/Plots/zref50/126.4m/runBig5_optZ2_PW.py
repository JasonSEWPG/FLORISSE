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
    nGroups = 2

    use_rotor_components = False

    datasize = 0
    rotor_diameter = 126.4

    nRows = 5
    nTurbs = nRows**2

    turbineX = np.array([
7.000000003836817086e+02,1.400000000767363417e+03,2.100000001151045126e+03,2.800000001534726835e+03,
3.500000001918408543e+03,7.000000003836817086e+02,1.400000000767363417e+03,2.100000001151045126e+03,
2.800000001534726835e+03,3.500000001918408543e+03,7.000000003836817086e+02,1.400000000767363417e+03,
2.100000001151045126e+03,2.800000001534726835e+03,3.500000001918408543e+03,7.000000003836817086e+02,
1.400000000767363417e+03,2.100000001151045126e+03,2.800000001534726835e+03,3.500000001918408543e+03,
7.000000003836817086e+02,1.400000000767363417e+03,2.100000001151045126e+03,2.800000001534726835e+03,
3.500000001918408543e+03    ])

    turbineY = np.array([
7.000000003836817086e+02,7.000000003836817086e+02,7.000000003836817086e+02,7.000000003836817086e+02,
7.000000003836817086e+02,1.400000000767363417e+03,1.400000000767363417e+03,1.400000000767363417e+03,
1.400000000767363417e+03,1.400000000767363417e+03,2.100000001151045126e+03,2.100000001151045126e+03,
2.100000001151045126e+03,2.100000001151045126e+03,2.100000001151045126e+03,2.800000001534726835e+03,
2.800000001534726835e+03,2.800000001534726835e+03,2.800000001534726835e+03,2.800000001534726835e+03,
3.500000001918408543e+03,3.500000001918408543e+03,3.500000001918408543e+03,3.500000001918408543e+03,
3.500000001918408543e+03    ])

    d1 = np.zeros((len(shearExp),3))
    t1 = np.zeros((len(shearExp),3))
    z1 = np.zeros((len(shearExp),1))
    d2 = np.zeros((len(shearExp),3))
    t2 = np.zeros((len(shearExp),3))
    z2 = np.zeros((len(shearExp),1))

    d1[0] =  np.array([4.573894804251582791e+00, 4.486842238738764976e+00, 3.870000000000000107e+00])
    d1[1] =  np.array([4.561216039270996170e+00, 4.474880300269095557e+00, 3.870000000000000107e+00])
    d1[2] =  np.array([4.567512007208835989e+00, 4.480944643191486776e+00, 3.870000000000000107e+00])
    d1[3] =  np.array([4.753474767282043878e+00, 4.672369192142406646e+00, 3.870000000000000107e+00])
    d1[4] =  np.array([4.893255576761290193e+00, 4.805318276605833994e+00, 3.870000000000000107e+00])
    d1[5] =  np.array([4.899795172357448436e+00, 4.811355746626317931e+00, 3.870000000000000107e+00])
    d1[6] =  np.array([5.202831304878690055e+00, 4.905955757216364788e+00, 3.870000000000000107e+00])
    d1[7] =  np.array([6.299999999999999822e+00, 4.315346093297200980e+00, 3.870000000000000107e+00])
    d1[8] =  np.array([6.299999999999999822e+00, 4.674896389425859944e+00, 3.870000000000000107e+00])
    d1[9] =  np.array([6.299999999999999822e+00, 4.924295441442450461e+00, 3.870000000000000107e+00])
    d1[10] = np.array([6.299999999999999822e+00, 5.041023513817457058e+00, 3.870000000000000107e+00])
    d1[11] = np.array([6.299999999999999822e+00, 5.067922063149112866e+00, 3.870000000000000107e+00])
    d1[12] = np.array([6.299999999999999822e+00, 5.076670596510426847e+00, 3.870000000000000107e+00])
    d1[13] = np.array([6.299999999999999822e+00, 5.125571310356279220e+00, 3.870000000000000107e+00])
    d1[14] = np.array([6.299999999999999822e+00, 5.186479157646486371e+00, 3.870000000000000107e+00])
    d1[15] = np.array([6.299999999999999822e+00, 5.176297136889931316e+00, 3.870000000000000107e+00])
    d1[16] = np.array([6.299999999999999822e+00, 5.165312855255198343e+00, 3.870000000000000107e+00])
    d1[17] = np.array([6.299999999999999822e+00, 5.157773663498415928e+00, 3.870000000000000107e+00])
    d1[18] = np.array([6.299999999999999822e+00, 5.244736342570166521e+00, 3.870000000000000107e+00])
    d1[19] = np.array([6.299999999999999822e+00, 5.189438822338538593e+00, 3.870000000000000107e+00])
    d1[20] = np.array([6.299999999999999822e+00, 5.419431458623146014e+00, 3.870000000000000107e+00])
    d1[21] = np.array([6.299999999999999822e+00, 5.420475577541170153e+00, 3.870000000000000107e+00])
    d1[22] = np.array([6.299999999999999822e+00, 5.590623750579359452e+00, 3.870000000000000107e+00])

    d2[0] =  np.array([4.552081744583600376e+00, 4.466418381125778936e+00, 3.870000000000000107e+00])
    d2[1] =  np.array([4.571948673068883195e+00, 4.485059736425028198e+00, 3.870000000000000107e+00])
    d2[2] =  np.array([4.566362183464462099e+00, 4.479865383157009440e+00, 3.870000000000000107e+00])
    d2[3] =  np.array([4.523686533802865561e+00, 4.439620336103359932e+00, 3.870000000000000107e+00])
    d2[4] =  np.array([4.580273552933067371e+00, 4.495522616847241082e+00, 3.870000000000000107e+00])
    d2[5] =  np.array([4.860794562787074113e+00, 4.785190099172068834e+00, 3.870000000000000107e+00])
    d2[6] =  np.array([4.890388147318763146e+00, 4.802369918760017065e+00, 3.870000000000000107e+00])
    d2[7] =  np.array([6.299999999999999822e+00, 4.226728035549740525e+00, 3.870000000000000107e+00])
    d2[8] =  np.array([5.308962418427123353e+00, 4.939391907011692062e+00, 3.870000000000000107e+00])
    d2[9] =  np.array([6.299999999999999822e+00, 4.847236682531582019e+00, 3.870000000000000107e+00])
    d2[10] = np.array([6.299999999999999822e+00, 4.866851440685684871e+00, 3.870000000000000107e+00])
    d2[11] = np.array([6.299999999999999822e+00, 5.066554855230537946e+00, 3.870000000000000107e+00])
    d2[12] = np.array([6.299999999999999822e+00, 5.071067554590649173e+00, 3.870000000000000107e+00])
    d2[13] = np.array([6.299999999999999822e+00, 5.198049532732144762e+00, 3.870000000000000107e+00])
    d2[14] = np.array([6.299999999999999822e+00, 5.186943872089627483e+00, 3.870000000000000107e+00])
    d2[15] = np.array([6.299999999999999822e+00, 5.176081859493910642e+00, 3.870000000000000107e+00])
    d2[16] = np.array([6.299999999999999822e+00, 5.164886493695452607e+00, 3.870000000000000107e+00])
    d2[17] = np.array([6.299999999999999822e+00, 5.178732614147150137e+00, 3.870000000000000107e+00])
    d2[18] = np.array([6.299999999999999822e+00, 5.160642389881264336e+00, 3.870000000000000107e+00])
    d2[19] = np.array([6.299999999999999822e+00, 5.176800123862794756e+00, 3.870000000000000107e+00])
    d2[20] = np.array([6.299999999999999822e+00, 5.381515981746717081e+00, 3.870000000000000107e+00])
    d2[21] = np.array([6.299999999999999822e+00, 5.417281722032547542e+00, 3.870000000000000107e+00])
    d2[22] = np.array([6.299999999999999822e+00, 5.535728236950308023e+00, 3.870000000000000107e+00])

    t1[0] =  np.array([2.683174143388857708e-02, 1.726953343139316088e-02, 1.096066201490672118e-02])
    t1[1] =  np.array([2.692181744633235715e-02, 1.730626481538230510e-02, 1.096067609051564612e-02])
    t1[2] =  np.array([2.687675995695647593e-02, 1.728761895723454275e-02, 1.096069030180397627e-02])
    t1[3] =  np.array([2.745477817220650860e-02, 1.762840517012715172e-02, 1.096073158310854022e-02])
    t1[4] =  np.array([2.868339677252375319e-02, 1.829149659747583700e-02, 1.096081207308033205e-02])
    t1[5] =  np.array([2.859582634282086325e-02, 1.825261195217075041e-02, 1.096083462555565852e-02])
    t1[6] =  np.array([2.808642552041638873e-02, 1.852981953948526495e-02, 1.096088888307427539e-02])
    t1[7] =  np.array([2.363627156265003804e-02, 2.156033890822216595e-02, 1.096093626523642062e-02])
    t1[8] =  np.array([2.424476264762158439e-02, 2.027199849017965669e-02, 1.096098246983996140e-02])
    t1[9] =  np.array([2.478527453862070656e-02, 1.944792060765533626e-02, 1.096102829831472301e-02])
    t1[10] = np.array([2.505055785389323497e-02, 1.908935560227708814e-02, 1.096106202904354593e-02])
    t1[11] = np.array([2.514005018742214090e-02, 1.899690899541220043e-02, 1.096109397534301630e-02])
    t1[12] = np.array([2.518626679986495087e-02, 1.897713905546233662e-02, 1.096112791462435128e-02])
    t1[13] = np.array([2.532580707748618293e-02, 1.885049632957218335e-02, 1.096117078573127616e-02])
    t1[14] = np.array([2.549083404807598457e-02, 1.868776013678705508e-02, 1.096120981116213873e-02])
    t1[15] = np.array([2.552090269110681570e-02, 1.871484785852762719e-02, 1.096125149927488877e-02])
    t1[16] = np.array([2.555021564631664369e-02, 1.872500127544248980e-02, 1.096129989547095507e-02])
    t1[17] = np.array([2.557519540675316783e-02, 1.875733948016202260e-02, 1.096132387247854625e-02])
    t1[18] = np.array([2.579543499699884362e-02, 1.860958036981305574e-02, 1.096136658991776411e-02])
    t1[19] = np.array([2.572332746418488220e-02, 1.870650787013899627e-02, 1.096026905039171502e-02])
    t1[20] = np.array([2.624362874758751768e-02, 1.833537527919319732e-02, 1.096147650362840065e-02])
    t1[21] = np.array([2.629677735852763501e-02, 1.833885295921314218e-02, 1.096151514013909110e-02])
    t1[22] = np.array([2.669224851250039110e-02, 1.810187496700456714e-02, 1.096157533053543354e-02])

    t2[0] =  np.array([2.698739254798891013e-02, 1.733367250146886779e-02, 1.096066201490672291e-02])
    t2[1] =  np.array([2.684942120871970303e-02, 1.727661592719008724e-02, 1.096067609051564612e-02])
    t2[2] =  np.array([2.688697159913314844e-02, 1.729185995537260018e-02, 1.096069030180397627e-02])
    t2[3] =  np.array([2.720539195024943013e-02, 1.742278665034176915e-02, 1.096071991840514605e-02])
    t2[4] =  np.array([2.706865867072181381e-02, 1.738115478859204865e-02, 1.096072553222457099e-02])
    t2[5] =  np.array([2.819253737093879372e-02, 1.801678286463464021e-02, 1.096082039582545707e-02])
    t2[6] =  np.array([2.864785444697975567e-02, 1.827335012873603642e-02, 1.096085729418647473e-02])
    t2[7] =  np.array([2.338262295843992686e-02, 2.183184278911541251e-02, 1.096092675666079409e-02])
    t2[8] =  np.array([2.775097707671820083e-02, 1.853769687645690986e-02, 1.096094593245631847e-02])
    t2[9] =  np.array([2.462637088114148776e-02, 1.969209787907747963e-02, 1.096102350836771280e-02])
    t2[10] = np.array([2.469384756657817814e-02, 1.962300232023492935e-02, 1.096105069609858151e-02])
    t2[11] = np.array([2.513726057291362087e-02, 1.900229408514030049e-02, 1.096110220211232028e-02])
    t2[12] = np.array([2.517703397979038568e-02, 1.899141340456496352e-02, 1.096113205723949977e-02])
    t2[13] = np.array([2.548162420236795012e-02, 1.867225201452989494e-02, 1.096117651523503995e-02])
    t2[14] = np.array([2.548778973481392526e-02, 1.869256600908351940e-02, 1.096120576363186425e-02])
    t2[15] = np.array([2.552482757996183543e-02, 1.871487652897101117e-02, 1.096125465251882383e-02])
    t2[16] = np.array([2.554384492391794173e-02, 1.874123397207915626e-02, 1.096128772540510068e-02])
    t2[17] = np.array([2.561748427157953256e-02, 1.871917798726895413e-02, 1.096132418347368388e-02])
    t2[18] = np.array([2.562493529809097351e-02, 1.875820107964991726e-02, 1.096090198941514562e-02])
    t2[19] = np.array([2.568633032683052819e-02, 1.872982355482038716e-02, 1.095904873000247415e-02])
    t2[20] = np.array([2.616718985186648749e-02, 1.839437467324471498e-02, 1.096146681290561392e-02])
    t2[21] = np.array([2.629031697684640692e-02, 1.834367908522854460e-02, 1.096151422020749493e-02])
    t2[22] = np.array([2.657801579361919464e-02, 1.817558219936448935e-02, 1.096151487331942925e-02])

    z1[0] =  np.array([7.320000000000000284e+01])
    z1[1] =  np.array([7.320000000000000284e+01])
    z1[2] =  np.array([7.320000000000000284e+01])
    z1[3] =  np.array([8.034108063872513128e+01])
    z1[4] =  np.array([8.909655427787777171e+01])
    z1[5] =  np.array([8.892411830540642370e+01])
    z1[6] =  np.array([9.352928544104503317e+01])
    z1[7] =  np.array([9.637521219987944221e+01])
    z1[8] =  np.array([9.872115300486949252e+01])
    z1[9] =  np.array([1.004271667575290081e+02])
    z1[10] = np.array([1.012388532561814856e+02])
    z1[11] = np.array([1.014360919440963897e+02])
    z1[12] = np.array([1.015145762036831769e+02])
    z1[13] = np.array([1.018817237412550298e+02])
    z1[14] = np.array([1.023330695661364587e+02])
    z1[15] = np.array([1.022871933687294472e+02])
    z1[16] = np.array([1.022280104778704839e+02])
    z1[17] = np.array([1.022044713526470048e+02])
    z1[18] = np.array([1.028832974509924725e+02])
    z1[19] = np.array([1.024908294067113701e+02])
    z1[20] = np.array([1.042617483317697094e+02])
    z1[21] = np.array([1.043000216882574449e+02])
    z1[22] = np.array([1.056357844430337423e+02])

    z2[0] =  np.array([7.320000000000000284e+01])
    z2[1] =  np.array([7.320000000000000284e+01])
    z2[2] =  np.array([7.320000000000000284e+01])
    z2[3] =  np.array([7.320000000000000284e+01])
    z2[4] =  np.array([7.421942819292917193e+01])
    z2[5] =  np.array([8.628070140183064041e+01])
    z2[6] =  np.array([8.886251991770606651e+01])
    z2[7] =  np.array([9.527585935861991118e+01])
    z2[8] =  np.array([9.439989461167191109e+01])
    z2[9] =  np.array([9.990130928346987105e+01])
    z2[10] = np.array([1.000465134474528384e+02])
    z2[11] = np.array([1.014270095866318826e+02])
    z2[12] = np.array([1.014759694506588232e+02])
    z2[13] = np.array([1.024017974392469057e+02])
    z2[14] = np.array([1.023371755078206888e+02])
    z2[15] = np.array([1.022876283232144488e+02])
    z2[16] = np.array([1.022302507042775801e+02])
    z2[17] = np.array([1.023606245709758582e+02])
    z2[18] = np.array([1.022531059770656867e+02])
    z2[19] = np.array([1.023926721506527713e+02])
    z2[20] = np.array([1.039736991095166303e+02])
    z2[21] = np.array([1.042756736196387237e+02])
    z2[22] = np.array([1.052111552754314943e+02])

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

    COE = np.zeros(23)
    AEP = np.zeros(23)
    idealAEP = np.zeros(23)
    cost = np.zeros(23)
    tower_cost = np.zeros(23)

    """OpenMDAO"""

    prob = Problem()
    root = prob.root = Group()

    root.add('d_param0', IndepVarComp('d_param0', d1[0]), promotes=['*'])
    root.add('t_param0', IndepVarComp('t_param0', t1[0]), promotes=['*'])
    root.add('turbineH0', IndepVarComp('turbineH0', float(z1[0])), promotes=['*'])

    root.add('d_param1', IndepVarComp('d_param1', d2[0]), promotes=['*'])
    root.add('t_param1', IndepVarComp('t_param1', t2[0]), promotes=['*'])
    root.add('turbineH1', IndepVarComp('turbineH1', float(z2[0])), promotes=['*'])

    for i in range(nGroups):
        root.add('get_z_param%s'%i, get_z(nPoints))
        root.add('get_z_full%s'%i, get_z(nFull))
        root.add('Tower%s_max_thrust'%i, Tower(nPoints, nFull), promotes=['L_reinforced','m','mrhox','E','sigma_y','gamma_f','gamma_b','rhoAir','z0','zref','shearExp','rho'])
        root.add('Tower%s_max_speed'%i, Tower(nPoints, nFull), promotes=['L_reinforced','m','mrhox','E','sigma_y','gamma_f','gamma_b','rhoAir','z0','zref','shearExp','rho'])

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
        prob['d_param0'] = d1[k]
        prob['d_param1'] = d2[k]
        prob['t_param0'] = t1[k]
        prob['t_param1'] = t2[k]
        prob['turbineH0'] = z1[k]
        prob['turbineH1'] = z2[k]
        prob.run()
        print prob['shearExp']
        print 'd1: ', prob['d_param0']
        print 'd2: ', prob['d_param1']
        print 't1: ', prob['t_param0']
        print 't2: ', prob['t_param1']
        print 'z1: ', prob['turbineH0']
        print 'z2: ', prob['turbineH1']
        COE[k] = prob['COE']
        AEP[k] = prob['AEP']
        cost[k] = prob['farm_cost']
        tower_cost[k] = prob['tower_cost']





    nGroups = 1
    prob = Problem()
    root = prob.root = Group()

    root.add('d_param0', IndepVarComp('d_param0', d1[k]), promotes=['*'])
    root.add('t_param0', IndepVarComp('t_param0', t1[k]), promotes=['*'])
    root.add('turbineH0', IndepVarComp('turbineH0', float(z1[k])), promotes=['*'])

    root.add('get_z_param0', get_z(nPoints))
    root.add('get_z_full0', get_z(nFull))
    root.add('Tower0_max_thrust', Tower(nPoints, nFull), promotes=['L_reinforced','m','mrhox','E','sigma_y','gamma_f','gamma_b','rhoAir','z0','zref','shearExp','rho'])
    root.add('Tower0_max_speed', Tower(nPoints, nFull), promotes=['L_reinforced','m','mrhox','E','sigma_y','gamma_f','gamma_b','rhoAir','z0','zref','shearExp','rho'])

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
        prob['d_param0'] = d1[k]
        prob['t_param0'] = t1[k]
        prob['turbineH0'] = z1[k]
        prob.run()
        AEP1 = prob['AEP']

        prob['d_param0'] = d2[k]
        prob['t_param0'] = t2[k]
        prob['turbineH0'] = z2[k]
        prob.run()
        AEP2 = prob['AEP']

        idealAEP[k] = AEP1*13.+AEP2*12.

    print 'ideal AEP: ', repr(idealAEP)
    print 'AEP: ', repr(AEP)
    print 'COE: ', repr(COE)
    print 'cost: ', repr(cost)
    print 'tower cost: ', repr(tower_cost)

    print 'wake loss: ', repr((idealAEP-AEP)/idealAEP*100.)

    # ideal AEP:  array([  3.93091892e+08,   3.97612765e+08,   4.02185631e+08,
    #          4.13410122e+08,   4.28162508e+08,   4.46516631e+08,
    #          4.61889936e+08,   4.80676837e+08,   4.91314637e+08,
    #          5.09146682e+08,   5.20151573e+08,   5.32160656e+08,
    #          5.42248358e+08,   5.54063418e+08,   5.64644213e+08,
    #          5.74431188e+08,   5.83330075e+08,   5.92247733e+08,
    #          6.02181934e+08,   6.10932907e+08,   6.26392712e+08,
    #          6.36973288e+08,   6.51798159e+08])
    # AEP:  array([  3.45896588e+08,   3.49874677e+08,   3.53898517e+08,
    #          3.63966934e+08,   3.77565376e+08,   3.92942408e+08,
    #          4.06533556e+08,   4.23022240e+08,   4.32859769e+08,
    #          4.49140296e+08,   4.59373911e+08,   4.70584833e+08,
    #          4.80160703e+08,   4.91335627e+08,   5.01451249e+08,
    #          5.10909364e+08,   5.20047149e+08,   5.29505517e+08,
    #          5.39863442e+08,   5.48759948e+08,   5.63764583e+08,
    #          5.74031347e+08,   5.87812425e+08])
    # COE:  array([ 29.62484089,  29.35760877,  29.09317968,  28.8544282 ,
    #         28.57944992,  28.28557978,  27.95578545,  27.61147961,
    #         27.20355467,  26.81304274,  26.40646756,  26.00514916,
    #         25.61759699,  25.24641153,  24.88331674,  24.53379278,
    #         24.20547131,  23.89125059,  23.58665065,  23.28964739,
    #         23.01944086,  22.73787917,  22.47486214])
    # cost:  array([  1.02471314e+10,   1.02714839e+10,   1.02960331e+10,
    #          1.05020578e+10,   1.07906108e+10,   1.11146038e+10,
    #          1.13649649e+10,   1.16802699e+10,   1.17753244e+10,
    #          1.20428179e+10,   1.21304423e+10,   1.22376288e+10,
    #          1.23005634e+10,   1.24044614e+10,   1.24777703e+10,
    #          1.25345445e+10,   1.25879863e+10,   1.26505490e+10,
    #          1.27335704e+10,   1.27804257e+10,   1.29775455e+10,
    #          1.30522554e+10,   1.32110032e+10])
    # tower cost:  array([ 11933796.08654021,  11934134.09199596,  11933868.75196715,
    #         12906445.66867079,  14294465.314464  ,  15822862.44751854,
    #         16891127.17332344,  18312281.6113744 ,  18549629.11785712,
    #         19697610.59564706,  19869847.54252298,  20135202.0415589 ,
    #         20166064.26553036,  20411875.4434912 ,  20491286.24470657,
    #         20484924.0034958 ,  20468771.3397968 ,  20502458.78338046,
    #         20639747.19526129,  20588043.95390093,  21320066.97724361,
    #         21403802.48118504,  21922285.55474718])
    # wake loss:  array([ 12.00617588,  12.00617588,  12.00617588,  11.95983978,
    #         11.81727279,  11.99825926,  11.98475577,  11.99446142,
    #         11.89764448,  11.78567748,  11.68460604,  11.5709087 ,
    #         11.45004029,  11.32140998,  11.19164276,  11.05821302,
    #         10.84856216,  10.59391399,  10.34878135,  10.17672457,
    #          9.99822125,   9.88140978,   9.81680178])
