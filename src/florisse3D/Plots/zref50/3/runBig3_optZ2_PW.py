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
    4.199999999573020091e+02,    8.399999999146041318e+02,    1.259999999871906084e+03,    1.679999999829208264e+03,
    2.099999999786510216e+03,    4.199999999573020091e+02,    8.399999999146041318e+02,    1.259999999871906084e+03,
    1.679999999829208264e+03,    2.099999999786510216e+03,    4.199999999573020091e+02,    8.399999999146041318e+02,
    1.259999999871906084e+03,    1.679999999829208264e+03,    2.099999999786510216e+03,    4.199999999573020091e+02,
    8.399999999146041318e+02,    1.259999999871906084e+03,    1.679999999829208264e+03,    2.099999999786510216e+03,
    4.199999999573020091e+02,    8.399999999146041318e+02,    1.259999999871906084e+03,    1.679999999829208264e+03,
    2.099999999786510216e+03,    ])

    turbineY = np.array([
    4.199999999573020091e+02,    4.199999999573020091e+02,    4.199999999573020091e+02,    4.199999999573020091e+02,
    4.199999999573020091e+02,    8.399999999146041318e+02,    8.399999999146041318e+02,    8.399999999146041318e+02,
    8.399999999146041318e+02,    8.399999999146041318e+02,    1.259999999871906084e+03,    1.259999999871906084e+03,
    1.259999999871906084e+03,    1.259999999871906084e+03,    1.259999999871906084e+03,    1.679999999829208264e+03,
    1.679999999829208264e+03,    1.679999999829208264e+03,    1.679999999829208264e+03,    1.679999999829208264e+03,
    2.099999999786510216e+03,    2.099999999786510216e+03,    2.099999999786510216e+03,    2.099999999786510216e+03,
    2.099999999786510216e+03,    ])

    d1 = np.zeros((len(shearExp),3))
    t1 = np.zeros((len(shearExp),3))
    z1 = np.zeros((len(shearExp),1))
    d2 = np.zeros((len(shearExp),3))
    t2 = np.zeros((len(shearExp),3))
    z2 = np.zeros((len(shearExp),1))

    d1[0] =  np.array([4.563508916458991926e+00, 4.476885015505256860e+00, 3.870000000000000107e+00])
    d1[1] =  np.array([4.575186597778553299e+00, 4.488328997216983751e+00, 3.870000000000000107e+00])
    d1[2] =  np.array([6.191698038888427291e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d1[3] =  np.array([4.562111903842282601e+00, 4.475968612309970673e+00, 3.870000000000000107e+00])
    d1[4] =  np.array([6.299999999999999822e+00, 5.270909359659539994e+00, 3.870000000000000107e+00])
    d1[5] =  np.array([6.299999999999999822e+00, 5.497471963798219896e+00, 3.870000000000000107e+00])
    d1[6] =  np.array([6.299999999999999822e+00, 5.543625946890704093e+00, 3.870000000000000107e+00])
    d1[7] =  np.array([6.299999999999999822e+00, 5.697141865777520131e+00, 3.870000000000000107e+00])
    d1[8] =  np.array([6.299999999999999822e+00, 5.932476026325388396e+00, 3.870000000000000107e+00])
    d1[9] =  np.array([6.299999999999999822e+00, 6.282987783600289866e+00, 3.870000000000000107e+00])
    d1[10] = np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 3.870000000000000107e+00])
    d1[11] = np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 3.870000000000000107e+00])
    d1[12] = np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 3.870000000000000107e+00])
    d1[13] = np.array([6.299999999999999822e+00, 5.199071954737479295e+00, 3.870000000000000107e+00])
    d1[14] = np.array([6.299999999999999822e+00, 5.187417274005714951e+00, 3.870000000000000107e+00])
    d1[15] = np.array([6.299999999999999822e+00, 5.176464435257150498e+00, 3.870000000000000107e+00])
    d1[16] = np.array([6.299999999999999822e+00, 5.164882672633066107e+00, 3.870000000000000107e+00])
    d1[17] = np.array([6.299999999999999822e+00, 5.317515864713977969e+00, 3.870000000000000107e+00])
    d1[18] = np.array([6.299999999999999822e+00, 5.605551331476839039e+00, 3.870000000000000107e+00])
    d1[19] = np.array([6.299999999999999822e+00, 5.733072774471020594e+00, 3.870000000000000107e+00])
    d1[20] = np.array([6.299999999999999822e+00, 5.752595204465281498e+00, 3.870000000000000107e+00])
    d1[21] = np.array([6.299999999999999822e+00, 5.779192626956920265e+00, 3.870000000000000107e+00])
    d1[22] = np.array([6.299999999999999822e+00, 5.722301176684214674e+00, 3.870000000000000107e+00])

    d2[0] =  np.array([4.562741298549036983e+00, 4.476180282752333817e+00, 3.870000000000000107e+00])
    d2[1] =  np.array([4.564624587277965162e+00, 4.478370720421650297e+00, 3.870000000000000107e+00])
    d2[2] =  np.array([6.192018630259972234e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d2[3] =  np.array([6.299999999999999822e+00, 5.278235724781779936e+00, 3.870000000000000107e+00])
    d2[4] =  np.array([4.567009057303919484e+00, 4.480689585906275418e+00, 3.870000000000000107e+00])
    d2[5] =  np.array([4.566484930790149832e+00, 4.480335346163373700e+00, 3.870000000000000107e+00])
    d2[6] =  np.array([4.582907775962955732e+00, 4.501158780059187237e+00, 3.870000000000000107e+00])
    d2[7] =  np.array([6.192294963478149228e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d2[8] =  np.array([4.550560852941501366e+00, 4.465512084693036066e+00, 3.870000000000000107e+00])
    d2[9] =  np.array([4.554000918781733631e+00, 4.468903060632316127e+00, 3.870000000000000107e+00])
    d2[10] = np.array([4.567580912990472086e+00, 4.481832209651169130e+00, 3.870000000000000107e+00])
    d2[11] = np.array([4.548762505379371213e+00, 4.464074917845145762e+00, 3.870000000000000107e+00])
    d2[12] = np.array([6.191710615890897884e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d2[13] = np.array([6.299999999999999822e+00, 5.123066106674751374e+00, 3.870000000000000107e+00])
    d2[14] = np.array([6.299999999999999822e+00, 5.184679562454054036e+00, 3.870000000000000107e+00])
    d2[15] = np.array([6.299999999999999822e+00, 5.176459616385356455e+00, 3.870000000000000107e+00])
    d2[16] = np.array([6.299999999999999822e+00, 5.165642346260665541e+00, 3.870000000000000107e+00])
    d2[17] = np.array([6.299999999999999822e+00, 5.153699889338295037e+00, 3.870000000000000107e+00])
    d2[18] = np.array([6.299999999999999822e+00, 5.142345477480245286e+00, 3.870000000000000107e+00])
    d2[19] = np.array([6.299999999999999822e+00, 5.182264835584526885e+00, 3.870000000000000107e+00])
    d2[20] = np.array([6.299999999999999822e+00, 5.588442856813764514e+00, 3.870000000000000107e+00])
    d2[21] = np.array([6.299999999999999822e+00, 5.559339543963137231e+00, 3.870000000000000107e+00])
    d2[22] = np.array([6.299999999999999822e+00, 5.779115149265111384e+00, 3.870000000000000107e+00])

    t1[0] =  np.array([2.690818315606698097e-02, 1.730061636990065307e-02, 1.095954256019358690e-02])
    t1[1] =  np.array([2.682543844989018073e-02, 1.726671728127013652e-02, 1.096066643877319398e-02])
    t1[2] =  np.array([1.994597856831350005e-02, 1.989065453103355460e-02, 1.096069007640670424e-02])
    t1[3] =  np.array([2.691400381381708201e-02, 1.730298918921490195e-02, 1.096070465007866097e-02])
    t1[4] =  np.array([2.534226054102211786e-02, 1.852142522879709693e-02, 1.096088290528288677e-02])
    t1[5] =  np.array([2.581947119137625521e-02, 1.817565617693076008e-02, 1.096092317161999168e-02])
    t1[6] =  np.array([2.595097184502789117e-02, 1.811332321159201536e-02, 1.096093474289353345e-02])
    t1[7] =  np.array([2.633859610894635425e-02, 1.793729616697671211e-02, 1.096096804394257850e-02])
    t1[8] =  np.array([2.694896401434225541e-02, 1.772986771315837248e-02, 1.096104668498892355e-02])
    t1[9] =  np.array([2.789330843042837904e-02, 1.748073485029910321e-02, 1.096092284102188029e-02])
    t1[10] = np.array([2.798455656404497116e-02, 1.747332236414744430e-02, 1.096114458576815699e-02])
    t1[11] = np.array([2.803126729087683511e-02, 1.747642482297488004e-02, 1.096118427262388756e-02])
    t1[12] = np.array([2.807775527816831762e-02, 1.747952617230127265e-02, 1.096122430276896299e-02])
    t1[13] = np.array([2.547455228272847705e-02, 1.866237859485521053e-02, 1.096118081404641453e-02])
    t1[14] = np.array([2.550554919706687559e-02, 1.868643121132237267e-02, 1.096120421255849728e-02])
    t1[15] = np.array([2.552386957159198666e-02, 1.870651814461423909e-02, 1.096125658565327843e-02])
    t1[16] = np.array([2.554407359208073908e-02, 1.874115161840853028e-02, 1.096128778656661751e-02])
    t1[17] = np.array([2.589396429027680802e-02, 1.848381788055231470e-02, 1.096133521118634706e-02])
    t1[18] = np.array([2.653821364431347904e-02, 1.805530761422009106e-02, 1.096141930490215216e-02])
    t1[19] = np.array([2.687368297456529942e-02, 1.794037803858645336e-02, 1.096146739231504581e-02])
    t1[20] = np.array([2.696498009897299944e-02, 1.792158047032369958e-02, 1.095778512391471990e-02])
    t1[21] = np.array([2.707263807533681621e-02, 1.789150029304593589e-02, 1.096154712511873339e-02])
    t1[22] = np.array([2.699508687341657664e-02, 1.796234718901949454e-02, 1.096159814007246006e-02])

    t2[0] =  np.array([2.691440893081624777e-02, 1.730329024052065442e-02, 1.095954255770542445e-02])
    t2[1] =  np.array([2.689848946764814702e-02, 1.729676816901235675e-02, 1.096067515034522197e-02])
    t2[2] =  np.array([1.994381752863732052e-02, 1.989060478009408925e-02, 1.096069007675550856e-02])
    t2[3] =  np.array([2.532834177479014945e-02, 1.850666341503100915e-02, 1.096085286352140899e-02])
    t2[4] =  np.array([2.688090545604125345e-02, 1.728912476197055209e-02, 1.096071913666164972e-02])
    t2[5] =  np.array([2.688753220848360756e-02, 1.729162212013814898e-02, 1.096073376288404094e-02])
    t2[6] =  np.array([2.676110632298130815e-02, 1.722673912914090416e-02, 1.096074852780713760e-02])
    t2[7] =  np.array([1.994367825562618282e-02, 1.989150925999893030e-02, 1.096076343964454483e-02])
    t2[8] =  np.array([2.700239383015431982e-02, 1.733870314270171548e-02, 1.096077849291306194e-02])
    t2[9] =  np.array([2.697556358861558784e-02, 1.732758513471478051e-02, 1.096079369128264563e-02])
    t2[10] = np.array([2.687540900335248784e-02, 1.728626900607444297e-02, 1.096080903615116384e-02])
    t2[11] = np.array([2.701134700425820295e-02, 1.734197923399466088e-02, 1.096082452893172229e-02])
    t2[12] = np.array([1.994138886913700975e-02, 1.989343197751152967e-02, 1.096084017103762971e-02])
    t2[13] = np.array([2.528906576917557686e-02, 1.886036131372020863e-02, 1.096116628393367827e-02])
    t2[14] = np.array([2.549993991247336000e-02, 1.868791152542505218e-02, 1.095595142213623245e-02])
    t2[15] = np.array([2.551284142648413758e-02, 1.871362870177305238e-02, 1.096124693514366304e-02])
    t2[16] = np.array([2.554055554231934866e-02, 1.874016950651749211e-02, 1.096128570724090416e-02])
    t2[17] = np.array([2.556711169865120287e-02, 1.876404190309559697e-02, 1.096132861302280201e-02])
    t2[18] = np.array([2.558450255115596664e-02, 1.879116083853116584e-02, 1.096135721316013160e-02])
    t2[19] = np.array([2.571483542631729236e-02, 1.872299160023675052e-02, 1.096140223316814308e-02])
    t2[20] = np.array([2.659208113972304569e-02, 1.807989352787123249e-02, 1.095750742336083078e-02])
    t2[21] = np.array([2.657493463319473420e-02, 1.814014119522246865e-02, 1.096148496618341156e-02])
    t2[22] = np.array([2.712661234864765028e-02, 1.790778236593140033e-02, 1.096160711718107382e-02])

    z1[0] =  np.array([7.320000000000000284e+01])
    z1[1] =  np.array([7.320000000000000284e+01])
    z1[2] =  np.array([7.320000000000000284e+01])
    z1[3] =  np.array([7.320000000000000284e+01])
    z1[4] =  np.array([1.027777149919563300e+02])
    z1[5] =  np.array([1.045191382035517336e+02])
    z1[6] =  np.array([1.048967910089193509e+02])
    z1[7] =  np.array([1.061449994084667026e+02])
    z1[8] =  np.array([1.081043677243911105e+02])
    z1[9] =  np.array([1.110755953534448679e+02])
    z1[10] = np.array([1.112453577034864480e+02])
    z1[11] = np.array([1.112705335537703490e+02])
    z1[12] = np.array([1.112956831368672539e+02])
    z1[13] = np.array([1.024000029107391327e+02])
    z1[14] = np.array([1.023447764416091985e+02])
    z1[15] = np.array([1.022853604137219321e+02])
    z1[16] = np.array([1.022303655150536912e+02])
    z1[17] = np.array([1.034033299040725495e+02])
    z1[18] = np.array([1.056378871050219317e+02])
    z1[19] = np.array([1.066977676662539807e+02])
    z1[20] = np.array([1.068788758394197203e+02])
    z1[21] = np.array([1.071178106349741626e+02])
    z1[22] = np.array([1.066939348194619441e+02])

    z2[0] =  np.array([7.320000000000000284e+01])
    z2[1] =  np.array([7.320000000000000284e+01])
    z2[2] =  np.array([7.320000000000000284e+01])
    z2[3] =  np.array([1.028158140466907327e+02])
    z2[4] =  np.array([7.320000000000000284e+01])
    z2[5] =  np.array([7.320000000000000284e+01])
    z2[6] =  np.array([7.320000000000000284e+01])
    z2[7] =  np.array([7.320000000000000284e+01])
    z2[8] =  np.array([7.320000000000000284e+01])
    z2[9] =  np.array([7.320000000000000284e+01])
    z2[10] = np.array([7.320000000000000284e+01])
    z2[11] = np.array([7.320000000000000284e+01])
    z2[12] = np.array([7.320000000000000284e+01])
    z2[13] = np.array([1.018513528991389734e+02])
    z2[14] = np.array([1.023214201107630572e+02])
    z2[15] = np.array([1.022849679362883109e+02])
    z2[16] = np.array([1.022336724032681730e+02])
    z2[17] = np.array([1.021735615760903642e+02])
    z2[18] = np.array([1.021150376758682228e+02])
    z2[19] = np.array([1.024422015538474682e+02])
    z2[20] = np.array([1.055529079613713321e+02])
    z2[21] = np.array([1.053634550225695818e+02])
    z2[22] = np.array([1.071544192321732965e+02])

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
    #          4.29978230e+08,   4.39295980e+08,   4.48475123e+08,
    #          4.56359366e+08,   4.65316692e+08,   4.75353677e+08,
    #          4.86831737e+08,   4.95223932e+08,   5.03527371e+08,
    #          5.11866030e+08,   5.54078394e+08,   5.64639736e+08,
    #          5.74424299e+08,   5.83338236e+08,   5.93861644e+08,
    #          6.06559400e+08,   6.18452876e+08,   6.34024044e+08,
    #          6.44552150e+08,   6.57777098e+08])
    # AEP:  array([  2.87977299e+08,   2.91289270e+08,   2.94639330e+08,
    #          3.21494228e+08,   3.28932713e+08,   3.36648542e+08,
    #          3.42908254e+08,   3.50476686e+08,   3.59346524e+08,
    #          3.70110110e+08,   3.76955311e+08,   3.83673523e+08,
    #          3.90460153e+08,   4.08801356e+08,   4.17266070e+08,
    #          4.25227343e+08,   4.32891960e+08,   4.42260825e+08,
    #          4.53673061e+08,   4.64308457e+08,   4.77879354e+08,
    #          4.87030425e+08,   4.98349115e+08])
    # COE:  array([ 34.3543856 ,  34.03343844,  33.73998897,  33.32872427,
    #         32.87523079,  32.43698721,  31.99602247,  31.576649  ,
    #         31.14614455,  30.73526804,  30.3081094 ,  29.88917964,
    #         29.48916764,  29.10885953,  28.67108777,  28.24486574,
    #         27.849837  ,  27.47241749,  27.09622119,  26.7206723 ,
    #         26.34620529,  25.97512977,  25.61461742])
    # cost:  array([  9.89328318e+09,   9.91357543e+09,   9.94112776e+09,
    #          1.07149925e+10,   1.08137389e+10,   1.09198645e+10,
    #          1.09717002e+10,   1.10668793e+10,   1.11922588e+10,
    #          1.13754334e+10,   1.14248028e+10,   1.14676869e+10,
    #          1.15143449e+10,   1.18997413e+10,   1.19634721e+10,
    #          1.20104892e+10,   1.20559705e+10,   1.21499740e+10,
    #          1.22928256e+10,   1.24066341e+10,   1.25903076e+10,
    #          1.26506785e+10,   1.27650219e+10])
    # tower cost:  array([ 11934077.77457232,  11934486.71054761,  11986114.49956485,
    #         16101104.52598352,  16443773.56029143,  16835688.80903514,
    #         16926359.47712496,  17255281.02550614,  17731204.69369097,
    #         18526967.51678783,  18578826.6771846 ,  18591745.26983801,
    #         18629125.154367  ,  20408584.22502878,  20491852.51400457,
    #         20481050.41416576,  20472370.66552951,  20727671.9414484 ,
    #         21235998.01606207,  21576485.51297639,  22276361.36883869,
    #         22308129.13030356,  22623583.56904258])
    # wake loss:  array([ 26.74046322,  26.74046322,  26.74046322,  25.23011492,
    #         25.1227581 ,  24.93484583,  24.86003799,  24.67996712,
    #         24.40438736,  23.9757638 ,  23.88184687,  23.80284658,
    #         23.7182914 ,  26.21958173,  26.10047719,  25.97330168,
    #         25.79057338,  25.52796944,  25.2055016 ,  24.92419796,
    #         24.62756607,  24.4389418 ,  24.23738735])
