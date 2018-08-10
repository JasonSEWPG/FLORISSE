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
    rotor_diameter = 70.0

    nRows = 9
    nTurbs = nRows**2

    turbineX = np.array([
    2.800000000171101533e+02,    5.600000000342204203e+02,    8.400000000513305167e+02,    1.120000000068440841e+03,
    1.400000000085550937e+03,    1.680000000102661261e+03,    1.960000000119771357e+03,    2.240000000136881681e+03,
    2.520000000153992005e+03,    2.800000000171101533e+02,    5.600000000342204203e+02,    8.400000000513305167e+02,
    1.120000000068440841e+03,    1.400000000085550937e+03,    1.680000000102661261e+03,    1.960000000119771357e+03,
    2.240000000136881681e+03,    2.520000000153992005e+03,    2.800000000171101533e+02,    5.600000000342204203e+02,
    8.400000000513305167e+02,    1.120000000068440841e+03,    1.400000000085550937e+03,    1.680000000102661261e+03,
    1.960000000119771357e+03,    2.240000000136881681e+03,    2.520000000153992005e+03,    2.800000000171101533e+02,
    5.600000000342204203e+02,    8.400000000513305167e+02,    1.120000000068440841e+03,    1.400000000085550937e+03,
    1.680000000102661261e+03,    1.960000000119771357e+03,    2.240000000136881681e+03,    2.520000000153992005e+03,
    2.800000000171101533e+02,    5.600000000342204203e+02,    8.400000000513305167e+02,    1.120000000068440841e+03,
    1.400000000085550937e+03,    1.680000000102661261e+03,    1.960000000119771357e+03,    2.240000000136881681e+03,
    2.520000000153992005e+03,    2.800000000171101533e+02,    5.600000000342204203e+02,    8.400000000513305167e+02,
    1.120000000068440841e+03,    1.400000000085550937e+03,    1.680000000102661261e+03,    1.960000000119771357e+03,
    2.240000000136881681e+03,    2.520000000153992005e+03,    2.800000000171101533e+02,    5.600000000342204203e+02,
    8.400000000513305167e+02,    1.120000000068440841e+03,    1.400000000085550937e+03,    1.680000000102661261e+03,
    1.960000000119771357e+03,    2.240000000136881681e+03,    2.520000000153992005e+03,    2.800000000171101533e+02,
    5.600000000342204203e+02,    8.400000000513305167e+02,    1.120000000068440841e+03,    1.400000000085550937e+03,
    1.680000000102661261e+03,    1.960000000119771357e+03,    2.240000000136881681e+03,    2.520000000153992005e+03,
    2.800000000171101533e+02,    5.600000000342204203e+02,    8.400000000513305167e+02,    1.120000000068440841e+03,
    1.400000000085550937e+03,    1.680000000102661261e+03,    1.960000000119771357e+03,    2.240000000136881681e+03,
    2.520000000153992005e+03    ])

    turbineY = np.array([
    2.800000000171101533e+02,    2.800000000171101533e+02,    2.800000000171101533e+02,    2.800000000171101533e+02,
    2.800000000171101533e+02,    2.800000000171101533e+02,    2.800000000171101533e+02,    2.800000000171101533e+02,
    2.800000000171101533e+02,    5.600000000342204203e+02,    5.600000000342204203e+02,    5.600000000342204203e+02,
    5.600000000342204203e+02,    5.600000000342204203e+02,    5.600000000342204203e+02,    5.600000000342204203e+02,
    5.600000000342204203e+02,    5.600000000342204203e+02,    8.400000000513305167e+02,    8.400000000513305167e+02,
    8.400000000513305167e+02,    8.400000000513305167e+02,    8.400000000513305167e+02,    8.400000000513305167e+02,
    8.400000000513305167e+02,    8.400000000513305167e+02,    8.400000000513305167e+02,    1.120000000068440841e+03,
    1.120000000068440841e+03,    1.120000000068440841e+03,    1.120000000068440841e+03,    1.120000000068440841e+03,
    1.120000000068440841e+03,    1.120000000068440841e+03,    1.120000000068440841e+03,    1.120000000068440841e+03,
    1.400000000085550937e+03,    1.400000000085550937e+03,    1.400000000085550937e+03,    1.400000000085550937e+03,
    1.400000000085550937e+03,    1.400000000085550937e+03,    1.400000000085550937e+03,    1.400000000085550937e+03,
    1.400000000085550937e+03,    1.680000000102661261e+03,    1.680000000102661261e+03,    1.680000000102661261e+03,
    1.680000000102661261e+03,    1.680000000102661261e+03,    1.680000000102661261e+03,    1.680000000102661261e+03,
    1.680000000102661261e+03,    1.680000000102661261e+03,    1.960000000119771357e+03,    1.960000000119771357e+03,
    1.960000000119771357e+03,    1.960000000119771357e+03,    1.960000000119771357e+03,    1.960000000119771357e+03,
    1.960000000119771357e+03,    1.960000000119771357e+03,    1.960000000119771357e+03,    2.240000000136881681e+03,
    2.240000000136881681e+03,    2.240000000136881681e+03,    2.240000000136881681e+03,    2.240000000136881681e+03,
    2.240000000136881681e+03,    2.240000000136881681e+03,    2.240000000136881681e+03,    2.240000000136881681e+03,
    2.520000000153992005e+03,    2.520000000153992005e+03,    2.520000000153992005e+03,    2.520000000153992005e+03,
    2.520000000153992005e+03,    2.520000000153992005e+03,    2.520000000153992005e+03,    2.520000000153992005e+03,
    2.520000000153992005e+03    ])

    d1 = np.zeros((len(shearExp),3))
    t1 = np.zeros((len(shearExp),3))
    z1 = np.zeros((len(shearExp),1))
    d2 = np.zeros((len(shearExp),3))
    t2 = np.zeros((len(shearExp),3))
    z2 = np.zeros((len(shearExp),1))

    d1[0] =  np.array([3.870000000000000107e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d1[1] =  np.array([6.299999999999999822e+00, 5.593685850239412005e+00, 3.870000000000000107e+00])
    d1[2] =  np.array([6.299999999999999822e+00, 5.780462358125333822e+00, 3.870000000000000107e+00])
    d1[3] =  np.array([6.299999999999999822e+00, 5.924075262595447811e+00, 3.870000000000000107e+00])
    d1[4] =  np.array([6.299999999999999822e+00, 6.102765288608442695e+00, 3.870000000000000107e+00])
    d1[5] =  np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 3.870000000000000107e+00])
    d1[6] =  np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 3.870000000000000107e+00])
    d1[7] =  np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 3.870000000000000107e+00])
    d1[8] =  np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 3.870000000000000107e+00])
    d1[9] =  np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 3.870000000000000107e+00])
    d1[10] = np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 3.870000000000000107e+00])
    d1[11] = np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 3.870000000000000107e+00])
    d1[12] = np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 3.870000000000000107e+00])
    d1[13] = np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 6.030114614803141038e+00])
    d1[14] = np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 6.299999999999999822e+00])
    d1[15] = np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 6.299999999999999822e+00])
    d1[16] = np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 6.299999999999999822e+00])
    d1[17] = np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 6.299999999999999822e+00])
    d1[18] = np.array([4.258793690482757910e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d1[19] = np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 6.299999999999999822e+00])
    d1[20] = np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 6.299999999999999822e+00])
    d1[21] = np.array([4.950949168654006805e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d1[22] = np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 6.299999999999999822e+00])

    d2[0] =  np.array([6.299999999999999822e+00, 5.477871441350679937e+00, 3.870000000000000107e+00])
    d2[1] =  np.array([3.870000000000000107e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d2[2] =  np.array([3.870000000000000107e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d2[3] =  np.array([3.870000000000000107e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d2[4] =  np.array([3.870000000000000107e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d2[5] =  np.array([3.870000000000000107e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d2[6] =  np.array([3.870000000000000107e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d2[7] =  np.array([3.870000000000000107e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d2[8] =  np.array([3.870000000000000107e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d2[9] =  np.array([3.870000000000000107e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d2[10] = np.array([3.870000000000000107e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d2[11] = np.array([3.870000000000000107e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d2[12] = np.array([3.870000000000000107e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d2[13] = np.array([3.870000000000000107e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d2[14] = np.array([3.870000000000000107e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d2[15] = np.array([3.870000000000000107e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d2[16] = np.array([3.974541239212861399e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d2[17] = np.array([4.075835402290427290e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d2[18] = np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 6.299999999999999822e+00])
    d2[19] = np.array([4.287486679458551997e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d2[20] = np.array([4.651037679827773275e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d2[21] = np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 6.299999999999999822e+00])
    d2[22] = np.array([5.076957495940761511e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])

    t1[0] =  np.array([1.039868059214303468e-02, 7.534053539940271937e-03, 6.034160799769684724e-03])
    t1[1] =  np.array([1.527809723264298911e-02, 1.038822594686213303e-02, 6.114456695737755486e-03])
    t1[2] =  np.array([1.547593036395489112e-02, 1.046113151496776486e-02, 6.120220244723952908e-03])
    t1[3] =  np.array([1.563427676271703440e-02, 1.052089818885194525e-02, 6.130873074069385399e-03])
    t1[4] =  np.array([1.585553274580423291e-02, 1.058764302729939613e-02, 6.141088135472582213e-03])
    t1[5] =  np.array([1.611854228297960462e-02, 1.068286464399700701e-02, 6.151899668512742295e-03])
    t1[6] =  np.array([1.613275140593904700e-02, 1.070040395242237083e-02, 6.160378714255784066e-03])
    t1[7] =  np.array([1.614779704683315734e-02, 1.071817864638364981e-02, 6.170430515482990680e-03])
    t1[8] =  np.array([1.616694743058484651e-02, 1.073624916285687014e-02, 6.179904403010343800e-03])
    t1[9] =  np.array([1.618674256688859212e-02, 1.075450838359527823e-02, 6.189520208449383193e-03])
    t1[10] = np.array([1.620712575731072386e-02, 1.077295316966812855e-02, 6.199270586415992802e-03])
    t1[11] = np.array([1.622809885788282364e-02, 1.079158991923683669e-02, 6.209176215382286070e-03])
    t1[12] = np.array([1.624966164966069535e-02, 1.081041717835850603e-02, 6.219221672839072255e-03])
    t1[13] = np.array([1.814696559592217373e-02, 1.175530915834231520e-02, 6.997471063866992323e-03])
    t1[14] = np.array([1.847417086611322437e-02, 1.191422591009017214e-02, 7.142443553595259099e-03])
    t1[15] = np.array([1.852382457983108369e-02, 1.194508943891295707e-02, 7.167388936010016254e-03])
    t1[16] = np.array([1.857447454311505000e-02, 1.197628354447771888e-02, 7.192642859917903336e-03])
    t1[17] = np.array([1.862612412022020270e-02, 1.200780995012328337e-02, 7.218198777402414466e-03])
    t1[18] = np.array([1.326307845134998796e-02, 9.047375197520754822e-03, 6.130703851705976079e-03])
    t1[19] = np.array([1.873201640337164922e-02, 1.207183413098120335e-02, 7.270104473626727570e-03])
    t1[20] = np.array([1.878728152329872322e-02, 1.210448533141777853e-02, 7.296933540047370777e-03])
    t1[21] = np.array([1.315356697212890710e-02, 9.510656050882762558e-03, 6.178689969484110409e-03])
    t1[22] = np.array([1.889997072150799776e-02, 1.217070711753629950e-02, 7.351103704554974583e-03])

    t2[0] =  np.array([1.515151006823104926e-02, 1.034551389932065156e-02, 6.101664694826746185e-03])
    t2[1] =  np.array([1.036790653310131795e-02, 7.523620035655482867e-03, 6.032973523204072583e-03])
    t2[2] =  np.array([1.033794635997486132e-02, 7.513295890939726204e-03, 6.031776055286356200e-03])
    t2[3] =  np.array([1.030877816599090092e-02, 7.502979927415675404e-03, 6.030544214176727230e-03])
    t2[4] =  np.array([1.048052831709767041e-02, 7.582326689470889579e-03, 6.034229629759441105e-03])
    t2[5] =  np.array([1.080738387581940685e-02, 7.726692485026435964e-03, 6.039172466748953493e-03])
    t2[6] =  np.array([1.100127056466422559e-02, 7.812279230524668897e-03, 6.040205457847966017e-03])
    t2[7] =  np.array([1.121821467996942864e-02, 7.905451626462188519e-03, 6.048062085241350319e-03])
    t2[8] =  np.array([1.143865194419314835e-02, 7.996702509062481035e-03, 6.053875548744866615e-03])
    t2[9] =  np.array([1.168372141191897397e-02, 8.096280165448071012e-03, 6.059592954177329960e-03])
    t2[10] = np.array([1.193121601133772559e-02, 8.195535518377794135e-03, 6.065573268887081322e-03])
    t2[11] = np.array([1.215621523621754699e-02, 8.285430121152310284e-03, 6.072667420604981742e-03])
    t2[12] = np.array([1.242686504864443271e-02, 8.391424300698566499e-03, 6.080636895482331890e-03])
    t2[13] = np.array([1.318658135109481919e-02, 8.674116950108997498e-03, 6.098420471215808790e-03])
    t2[14] = np.array([1.352818145843937697e-02, 8.801305392199922431e-03, 6.108752034098162940e-03])
    t2[15] = np.array([1.374725079363933386e-02, 8.885167107032070077e-03, 6.116136858555228963e-03])
    t2[16] = np.array([1.356272996317466929e-02, 8.925102673092384642e-03, 6.122639747704630882e-03])
    t2[17] = np.array([1.340510739271498497e-02, 8.965539700703533191e-03, 6.128635665452982981e-03])
    t2[18] = np.array([1.867848099979352652e-02, 1.203947035319466279e-02, 7.240937233747463467e-03])
    t2[19] = np.array([1.325488479292743008e-02, 9.063081704715713477e-03, 6.144610510766835079e-03])
    t2[20] = np.array([1.317816775633934290e-02, 9.307069197602622912e-03, 6.157043131664379117e-03])
    t2[21] = np.array([1.884309172606814589e-02, 1.213741472904529030e-02, 7.323843023478703376e-03])
    t2[22] = np.array([1.314756530244942483e-02, 9.599907331886398110e-03, 6.193515662530571335e-03])

    z1[0] =  np.array([4.500000000000000000e+01])
    z1[1] =  np.array([9.172433360028018967e+01])
    z1[2] =  np.array([9.304949375023147695e+01])
    z1[3] =  np.array([9.406839859273598847e+01])
    z1[4] =  np.array([9.534970672066542363e+01])
    z1[5] =  np.array([9.679505369478454213e+01])
    z1[6] =  np.array([9.681907976106231217e+01])
    z1[7] =  np.array([9.684400310214088847e+01])
    z1[8] =  np.array([9.687150839985599760e+01])
    z1[9] =  np.array([9.689952331938727070e+01])
    z1[10] = np.array([9.692800718464985721e+01])
    z1[11] = np.array([9.695696658518939159e+01])
    z1[12] = np.array([9.698639498569616535e+01])
    z1[13] = np.array([1.027346335401665840e+02])
    z1[14] = np.array([1.036491079182058428e+02])
    z1[15] = np.array([1.037001096871344998e+02])
    z1[16] = np.array([1.037516679336665248e+02])
    z1[17] = np.array([1.038037752088636836e+02])
    z1[18] = np.array([6.677066038202059417e+01])
    z1[19] = np.array([1.039094113023192278e+02])
    z1[20] = np.array([1.039635664099305217e+02])
    z1[21] = np.array([7.109507413842591461e+01])
    z1[22] = np.array([1.040729353450522154e+02])

    z2[0] =  np.array([9.092752386093600592e+01])
    z2[1] =  np.array([4.500000000000000000e+01])
    z2[2] =  np.array([4.500000000000000000e+01])
    z2[3] =  np.array([4.500000000000000000e+01])
    z2[4] =  np.array([4.626945783602561590e+01])
    z2[5] =  np.array([4.845670805489447019e+01])
    z2[6] =  np.array([4.978001804532508601e+01])
    z2[7] =  np.array([5.118847839686672785e+01])
    z2[8] =  np.array([5.254828004007910636e+01])
    z2[9] =  np.array([5.400866787000087754e+01])
    z2[10] = np.array([5.544459628255889072e+01])
    z2[11] = np.array([5.672733334507725544e+01])
    z2[12] = np.array([5.820987451928311174e+01])
    z2[13] = np.array([6.201847856677024851e+01])
    z2[14] = np.array([6.369358861259883042e+01])
    z2[15] = np.array([6.477190423299732913e+01])
    z2[16] = np.array([6.527359315089627501e+01])
    z2[17] = np.array([6.577459923957216859e+01])
    z2[18] = np.array([1.038549845265818874e+02])
    z2[19] = np.array([6.694304680296920651e+01])
    z2[20] = np.array([6.917287924820807632e+01])
    z2[21] = np.array([1.040179626788484342e+02])
    z2[22] = np.array([7.191319726592772099e+01])

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
    wind_zref = 50.0
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

    prob['ratedPower'] = np.ones(nTurbs)*1543.209877 # in kw

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

    prob['ratedPower'] = np.ones(1)*1543.209877 # in kw

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

        idealAEP[k] = AEP1*41.+AEP2*40.

    print 'ideal AEP: ', repr(idealAEP)
    print 'AEP: ', repr(AEP)
    print 'COE: ', repr(COE)
    print 'cost: ', repr(cost)
    print 'tower cost: ', repr(tower_cost)

    print 'wake loss: ', repr((idealAEP-AEP)/idealAEP*100.)

    # ideal AEP:  array([  3.79123638e+08,   3.83641693e+08,   3.87939637e+08,
    #          3.92287394e+08,   3.98817195e+08,   4.07340805e+08,
    #          4.13851684e+08,   4.20824942e+08,   4.27774260e+08,
    #          4.35251318e+08,   4.43059398e+08,   4.50855788e+08,
    #          4.59305008e+08,   4.81456023e+08,   4.92827261e+08,
    #          5.01899618e+08,   5.09422906e+08,   5.16999761e+08,
    #          5.24156916e+08,   5.33337599e+08,   5.46547442e+08,
    #          5.57534899e+08,   5.69292621e+08])
    # AEP:  array([  3.07206890e+08,   3.11626110e+08,   3.16135590e+08,
    #          3.20436324e+08,   3.25785083e+08,   3.32189707e+08,
    #          3.36459928e+08,   3.40965216e+08,   3.45596870e+08,
    #          3.50475655e+08,   3.55560560e+08,   3.60752212e+08,
    #          3.66214780e+08,   3.87216551e+08,   3.96206633e+08,
    #          4.02946720e+08,   4.09440188e+08,   4.16014837e+08,
    #          4.21240777e+08,   4.29808381e+08,   4.38304424e+08,
    #          4.45010963e+08,   4.54601058e+08])
    # COE:  array([ 63.60348692,  63.14631016,  62.66482486,  62.16597728,
    #         61.66987784,  61.14354666,  60.55619148,  59.95596318,
    #         59.35160913,  58.7361187 ,  58.1080078 ,  57.47302612,
    #         56.83640911,  56.2755433 ,  55.5789561 ,  54.865643  ,
    #         54.16463455,  53.4777533 ,  52.84833015,  52.12425272,
    #         51.48210919,  50.87540938,  50.18886463])
    # cost:  array([  1.95394294e+10,   1.96780390e+10,   1.98105814e+10,
    #          1.99202373e+10,   2.00911262e+10,   2.03112568e+10,
    #          2.03747318e+10,   2.04428980e+10,   2.05117304e+10,
    #          2.05855797e+10,   2.06609158e+10,   2.07335213e+10,
    #          2.08143330e+10,   2.17908218e+10,   2.20207511e+10,
    #          2.21079309e+10,   2.21771782e+10,   2.22475388e+10,
    #          2.22618716e+10,   2.24034407e+10,   2.25648362e+10,
    #          2.26401149e+10,   2.28159109e+10])
    # tower cost:  array([ 21692436.65656537,  22438811.56115383,  23157984.94690884,
    #         23730317.98210882,  24649372.00767979,  25843006.66238177,
    #         26068998.82959853,  26315756.72632734,  26563343.74292838,
    #         26833013.01282367,  27105176.76374039,  27357960.82533768,
    #         27651596.51661097,  33505808.4470419 ,  34692422.23938881,
    #         34988864.89050635,  35185447.3387778 ,  35386538.54697528,
    #         35253815.54294945,  35863242.44721215,  36572872.6227161 ,
    #         36775203.90075533,  37557581.52974909])
    # wake loss:  array([ 18.96920689,  18.77157359,  18.50907712,  18.31592614,
    #         18.31217754,  18.44919483,  18.70036039,  18.9769469 ,
    #         19.21045691,  19.47740517,  19.74878287,  19.98501036,
    #         20.2676275 ,  19.57384838,  19.60537389,  19.71567508,
    #         19.62666314,  19.53287644,  19.63460491,  19.41157317,
    #         19.80487135,  20.18240231,  20.14632885])
