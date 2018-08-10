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

    d1 = np.zeros((len(shearExp),3))
    t1 = np.zeros((len(shearExp),3))
    z1 = np.zeros((len(shearExp),1))
    d2 = np.zeros((len(shearExp),3))
    t2 = np.zeros((len(shearExp),3))
    z2 = np.zeros((len(shearExp),1))

    d1[0] =  np.array([4.577271238345233861e+00, 4.490015545812129361e+00, 3.870000000000000107e+00])
    d1[1] =  np.array([4.567316539647614526e+00, 4.480657226094654000e+00, 3.870000000000000107e+00])
    d1[2] =  np.array([4.562221840912838111e+00, 4.475951958478070480e+00, 3.870000000000000107e+00])
    d1[3] =  np.array([5.474684372337176974e+00, 4.176420833281239808e+00, 3.870000000000000107e+00])
    d1[4] =  np.array([6.299999999999999822e+00, 4.776283464889905517e+00, 3.870000000000000107e+00])
    d1[5] =  np.array([6.299999999999999822e+00, 5.173622364757271441e+00, 3.870000000000000107e+00])
    d1[6] =  np.array([6.299999999999999822e+00, 5.156085903472624743e+00, 3.870000000000000107e+00])
    d1[7] =  np.array([6.299999999999999822e+00, 5.248201804856629238e+00, 3.870000000000000107e+00])
    d1[8] =  np.array([6.299999999999999822e+00, 5.120675301299620230e+00, 3.870000000000000107e+00])
    d1[9] =  np.array([6.299999999999999822e+00, 4.806066899645673018e+00, 3.870000000000000107e+00])
    d1[10] = np.array([6.299999999999999822e+00, 5.083362462714313246e+00, 3.870000000000000107e+00])
    d1[11] = np.array([6.299999999999999822e+00, 5.067996862938369595e+00, 3.870000000000000107e+00])
    d1[12] = np.array([6.299999999999999822e+00, 5.059515548585867606e+00, 3.870000000000000107e+00])
    d1[13] = np.array([6.299999999999999822e+00, 5.198083411812302046e+00, 3.870000000000000107e+00])
    d1[14] = np.array([6.299999999999999822e+00, 5.187845372694654955e+00, 3.870000000000000107e+00])
    d1[15] = np.array([6.299999999999999822e+00, 5.176065165185261385e+00, 3.870000000000000107e+00])
    d1[16] = np.array([6.299999999999999822e+00, 5.165369609280103802e+00, 3.870000000000000107e+00])
    d1[17] = np.array([6.299999999999999822e+00, 5.184489881014253676e+00, 3.870000000000000107e+00])
    d1[18] = np.array([6.299999999999999822e+00, 5.407387212391199682e+00, 3.870000000000000107e+00])
    d1[19] = np.array([6.299999999999999822e+00, 5.495005225680073657e+00, 3.870000000000000107e+00])
    d1[20] = np.array([6.299999999999999822e+00, 5.549170896129709263e+00, 3.870000000000000107e+00])
    d1[21] = np.array([6.299999999999999822e+00, 5.603688631864925718e+00, 3.870000000000000107e+00])
    d1[22] = np.array([6.299999999999999822e+00, 5.780864075918358935e+00, 3.870000000000000107e+00])

    d2[0] =  np.array([4.571267367153580174e+00, 4.484318361933839370e+00, 3.870000000000000107e+00])
    d2[1] =  np.array([4.566612296754059663e+00, 4.479988673263306964e+00, 3.870000000000000107e+00])
    d2[2] =  np.array([4.559441033868593784e+00, 4.473297463282074560e+00, 3.870000000000000107e+00])
    d2[3] =  np.array([6.267090858083472327e+00, 3.921807486114207375e+00, 3.870000000000000107e+00])
    d2[4] =  np.array([4.560446680407978626e+00, 4.474481608856901715e+00, 3.870000000000000107e+00])
    d2[5] =  np.array([4.579322199404793459e+00, 4.492510736541270866e+00, 3.870000000000000107e+00])
    d2[6] =  np.array([4.565154056566048446e+00, 4.479178598580546122e+00, 3.870000000000000107e+00])
    d2[7] =  np.array([5.495412769657979091e+00, 4.111440545362316179e+00, 3.870000000000000107e+00])
    d2[8] =  np.array([5.028423658135961816e+00, 4.833005563731949117e+00, 3.870000000000000107e+00])
    d2[9] =  np.array([6.286649831384035103e+00, 4.773451934182548761e+00, 3.870000000000000107e+00])
    d2[10] = np.array([6.299999999999999822e+00, 4.981435145986352175e+00, 3.870000000000000107e+00])
    d2[11] = np.array([6.299999999999999822e+00, 5.064094582227236963e+00, 3.870000000000000107e+00])
    d2[12] = np.array([6.299999999999999822e+00, 5.162747714355315765e+00, 3.870000000000000107e+00])
    d2[13] = np.array([6.299999999999999822e+00, 5.126994037803771498e+00, 3.870000000000000107e+00])
    d2[14] = np.array([6.299999999999999822e+00, 5.187035866648455062e+00, 3.870000000000000107e+00])
    d2[15] = np.array([6.299999999999999822e+00, 5.174053450335255988e+00, 3.870000000000000107e+00])
    d2[16] = np.array([6.299999999999999822e+00, 5.165163816408282926e+00, 3.870000000000000107e+00])
    d2[17] = np.array([6.299999999999999822e+00, 5.178105750938246210e+00, 3.870000000000000107e+00])
    d2[18] = np.array([6.299999999999999822e+00, 5.225687558207188665e+00, 3.870000000000000107e+00])
    d2[19] = np.array([6.299999999999999822e+00, 5.497765006093040085e+00, 3.870000000000000107e+00])
    d2[20] = np.array([6.299999999999999822e+00, 5.551725020573552882e+00, 3.870000000000000107e+00])
    d2[21] = np.array([6.299999999999999822e+00, 5.609722395517112048e+00, 3.870000000000000107e+00])
    d2[22] = np.array([6.299999999999999822e+00, 5.610235716428511488e+00, 3.870000000000000107e+00])

    t1[0] =  np.array([2.681076127957266256e-02, 1.726100815717048972e-02, 1.096066198813428795e-02])
    t1[1] =  np.array([2.688005111900758012e-02, 1.728913399954659647e-02, 1.096067609051564612e-02])
    t1[2] =  np.array([2.691959748176208853e-02, 1.730534419424049203e-02, 1.096069030180397627e-02])
    t1[3] =  np.array([2.276995260166887053e-02, 1.868212632920882021e-02, 1.096070533721893968e-02])
    t1[4] =  np.array([2.435785124954882239e-02, 1.995855520640596112e-02, 1.096086588918233386e-02])
    t1[5] =  np.array([2.517772914845551116e-02, 1.874668373074655017e-02, 1.096052057726605924e-02])
    t1[6] =  np.array([2.517151895397786582e-02, 1.878907052617788398e-02, 1.096094025894772672e-02])
    t1[7] =  np.array([2.538553408344880122e-02, 1.856766483020861822e-02, 1.096097612937459512e-02])
    t1[8] =  np.array([2.515797011602353972e-02, 1.887268620871612387e-02, 1.096100096742019629e-02])
    t1[9] =  np.array([2.453985109143219309e-02, 1.982756613540104587e-02, 1.096100251313611093e-02])
    t1[10] = np.array([2.514142852211691259e-02, 1.896350210372717629e-02, 1.096106640089372468e-02])
    t1[11] = np.array([2.514043770772006817e-02, 1.899685279483793315e-02, 1.096110247880483120e-02])
    t1[12] = np.array([2.514747860758517750e-02, 1.902129615477868296e-02, 1.096112810311306550e-02])
    t1[13] = np.array([2.548104668683861054e-02, 1.867196678694750533e-02, 1.096112111786220686e-02])
    t1[14] = np.array([2.549733563912246967e-02, 1.869055476008889966e-02, 1.096121437118285392e-02])
    t1[15] = np.array([2.551967966098107887e-02, 1.871345046915944346e-02, 1.096123371521313949e-02])
    t1[16] = np.array([2.554007797277865754e-02, 1.873907746046363579e-02, 1.095967321229602588e-02])
    t1[17] = np.array([2.562734926746671263e-02, 1.871082775560781522e-02, 1.096110041068838747e-02])
    t1[18] = np.array([2.611928993480280847e-02, 1.834562236700259044e-02, 1.096138180411921660e-02])
    t1[19] = np.array([2.634208708375000707e-02, 1.822058797392064541e-02, 1.096119873310119446e-02])
    t1[20] = np.array([2.650682357771187020e-02, 1.814923760198032879e-02, 1.096149052435539673e-02])
    t1[21] = np.array([2.667383998210743831e-02, 1.806949332597300797e-02, 1.096155126649127863e-02])
    t1[22] = np.array([2.713095285732182449e-02, 1.790360382104715445e-02, 1.096655574533107053e-02])

    t2[0] =  np.array([2.685383382455508450e-02, 1.727867250314033681e-02, 1.096066198813160780e-02])
    t2[1] =  np.array([2.688489826474787375e-02, 1.729111314044673442e-02, 1.096067609051564438e-02])
    t2[2] =  np.array([2.693596487829576225e-02, 1.731195202096955099e-02, 1.096069030180397627e-02])
    t2[3] =  np.array([2.116049214842443543e-02, 2.127654093831350679e-02, 1.096075356174110518e-02])
    t2[4] =  np.array([2.693132497217875718e-02, 1.730991461383565161e-02, 1.096071913666167574e-02])
    t2[5] =  np.array([2.679734682104641455e-02, 1.725435798986413652e-02, 1.096051029162729666e-02])
    t2[6] =  np.array([2.689343683151903391e-02, 1.729380112274896825e-02, 1.096074853009213271e-02])
    t2[7] =  np.array([2.297981522531048620e-02, 1.920611190619765651e-02, 1.096077749926752283e-02])
    t2[8] =  np.array([2.854343496899614321e-02, 1.849283296842282875e-02, 1.096093245371363226e-02])
    t2[9] =  np.array([2.449975278462988154e-02, 1.992753085422967890e-02, 1.096090794850440413e-02])
    t2[10] = np.array([2.493108333717586333e-02, 1.926734313916962996e-02, 1.096105904368992884e-02])
    t2[11] = np.array([2.513206256846588563e-02, 1.901121758950722815e-02, 1.096109893901688059e-02])
    t2[12] = np.array([2.536453459202126295e-02, 1.876110971042996636e-02, 1.096113773776564992e-02])
    t2[13] = np.array([2.533093717014197818e-02, 1.884376831132510144e-02, 1.096108058630265204e-02])
    t2[14] = np.array([2.550215654212010033e-02, 1.869511372174431293e-02, 1.096121258364996486e-02])
    t2[15] = np.array([2.550372289756071970e-02, 1.871456525411633459e-02, 1.096120805569583169e-02])
    t2[16] = np.array([2.554113093022942343e-02, 1.874114086152754363e-02, 1.095968071188522033e-02])
    t2[17] = np.array([2.561312768447849067e-02, 1.872258153565684036e-02, 1.096107638567272134e-02])
    t2[18] = np.array([2.575036210662052030e-02, 1.864187879031905387e-02, 1.096084126273930842e-02])
    t2[19] = np.array([2.634770116045473745e-02, 1.821423532181074695e-02, 1.096065474091995583e-02])
    t2[20] = np.array([2.651237137704074917e-02, 1.814498685698860692e-02, 1.096149391203860533e-02])
    t2[21] = np.array([2.667922750602190821e-02, 1.807207892487976142e-02, 1.096152694662169171e-02])
    t2[22] = np.array([2.673726393241739746e-02, 1.807667762741265935e-02, 1.096158308361969667e-02])

    z1[0] =  np.array([7.320000000000000284e+01])
    z1[1] =  np.array([7.320000000000000284e+01])
    z1[2] =  np.array([7.320000000000000284e+01])
    z1[3] =  np.array([7.320000000000000284e+01])
    z1[4] =  np.array([9.936768395625240657e+01])
    z1[5] =  np.array([1.020963223363434906e+02])
    z1[6] =  np.array([1.019869662958678020e+02])
    z1[7] =  np.array([1.026600002475595943e+02])
    z1[8] =  np.array([1.017649938828523659e+02])
    z1[9] =  np.array([9.961847197687549738e+01])
    z1[10] = np.array([1.015312971784926219e+02])
    z1[11] = np.array([1.014369485045197337e+02])
    z1[12] = np.array([1.013919954868071471e+02])
    z1[13] = np.array([1.024016607311403249e+02])
    z1[14] = np.array([1.023456549076060753e+02])
    z1[15] = np.array([1.022852547514456063e+02])
    z1[16] = np.array([1.022309265127805986e+02])
    z1[17] = np.array([1.024039419513529339e+02])
    z1[18] = np.array([1.041114563507125013e+02])
    z1[19] = np.array([1.048082852238738809e+02])
    z1[20] = np.array([1.052571272398207896e+02])
    z1[21] = np.array([1.057046252109822007e+02])
    z1[22] = np.array([1.071683413392912030e+02])

    z2[0] =  np.array([7.320000000000000284e+01])
    z2[1] =  np.array([7.320000000000000284e+01])
    z2[2] =  np.array([7.320000000000000284e+01])
    z2[3] =  np.array([8.245868933037945681e+01])
    z2[4] =  np.array([7.320000000000000284e+01])
    z2[5] =  np.array([7.320000000000000284e+01])
    z2[6] =  np.array([7.320000000000000284e+01])
    z2[7] =  np.array([7.469171022997574028e+01])
    z2[8] =  np.array([9.137745963409722094e+01])
    z2[9] =  np.array([9.931299993936795545e+01])
    z2[10] = np.array([1.008305407082570326e+02])
    z2[11] = np.array([1.014105507388036926e+02])
    z2[12] = np.array([1.021273533796530160e+02])
    z2[13] = np.array([1.018916963681577528e+02])
    z2[14] = np.array([1.023448338546086234e+02])
    z2[15] = np.array([1.022640116600902616e+02])
    z2[16] = np.array([1.022305757995129483e+02])
    z2[17] = np.array([1.023557896676853716e+02])
    z2[18] = np.array([1.027373547366691753e+02])
    z2[19] = np.array([1.048276951005909723e+02])
    z2[20] = np.array([1.052762857975955058e+02])
    z2[21] = np.array([1.057551227296440715e+02])
    z2[22] = np.array([1.057906055401705459e+02])

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
    #          4.14638709e+08,   4.36376838e+08,   4.46210776e+08,
    #          4.53570277e+08,   4.63541398e+08,   4.91470667e+08,
    #          5.07437751e+08,   5.21479042e+08,   5.32140529e+08,
    #          5.42932363e+08,   5.54135833e+08,   5.64675228e+08,
    #          5.74393052e+08,   5.83334835e+08,   5.92554869e+08,
    #          6.04979856e+08,   6.19019276e+08,   6.30512378e+08,
    #          6.42469872e+08,   6.56132466e+08])
    # AEP:  array([  3.24409104e+08,   3.28140069e+08,   3.31913944e+08,
    #          3.42543665e+08,   3.63741774e+08,   3.72712274e+08,
    #          3.78913362e+08,   3.87197658e+08,   4.06683380e+08,
    #          4.19722842e+08,   4.31917672e+08,   4.41220523e+08,
    #          4.50869430e+08,   4.60990616e+08,   4.70429740e+08,
    #          4.79211007e+08,   4.87779845e+08,   4.96928803e+08,
    #          5.09231506e+08,   5.23016059e+08,   5.34193429e+08,
    #          5.45560057e+08,   5.58452980e+08])
    # COE:  array([ 31.18268636,  30.89738575,  30.61580955,  30.38976187,
    #         30.05490986,  29.67751574,  29.28418637,  28.93063753,
    #         28.59445666,  28.1822051 ,  27.75806219,  27.32849515,
    #         26.91483991,  26.5089434 ,  26.12372297,  25.74998573,
    #         25.40312924,  25.0692544 ,  24.74255487,  24.41730258,
    #         24.09140355,  23.77404717,  23.46544823])
    # cost:  array([  1.01159473e+10,   1.01386703e+10,   1.01618141e+10,
    #          1.04098204e+10,   1.09322262e+10,   1.10611744e+10,
    #          1.10961695e+10,   1.12018751e+10,   1.16288903e+10,
    #          1.18287152e+10,   1.19891976e+10,   1.20578929e+10,
    #          1.21350785e+10,   1.22203742e+10,   1.22893762e+10,
    #          1.23396766e+10,   1.23911344e+10,   1.24576346e+10,
    #          1.25996885e+10,   1.27706414e+10,   1.28694695e+10,
    #          1.29701705e+10,   1.31043495e+10])
    # tower cost:  array([ 11934557.53274826,  11934023.77952536,  11934646.35407243,
    #         13172336.1509229 ,  15824749.25623689,  16311977.69514944,
    #         16293165.46617116,  16650671.77485725,  18645182.52042643,
    #         19462430.68837095,  20051609.56149949,  20133032.24945986,
    #         20258941.50178483,  20420947.26096756,  20499426.72031866,
    #         20476512.2250969 ,  20470931.17783503,  20544929.64532859,
    #         21009151.39287437,  21611612.40253787,  21824415.26695262,
    #         22042140.80477513,  22429044.21632938])
    # wake loss:  array([ 17.47245106,  17.47245106,  17.47245106,  17.38743694,
    #         16.64503192,  16.47170034,  16.45983404,  16.46967037,
    #         17.25174922,  17.28584615,  17.1744907 ,  17.08571347,
    #         16.9566117 ,  16.80909473,  16.69021127,  16.57089065,
    #         16.38081335,  16.13792584,  15.82670057,  15.50892201,
    #         15.27629781,  15.08394701,  14.88715926])
