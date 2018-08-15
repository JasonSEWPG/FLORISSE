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
    2.100000000233957849e+02,    4.200000000467915697e+02,    6.300000000701872978e+02,    8.400000000935831395e+02,
    1.050000000116978981e+03,    1.260000000140374823e+03,    1.470000000163770437e+03,    1.680000000187166279e+03,
    1.890000000210562121e+03,    2.100000000233957849e+02,    4.200000000467915697e+02,    6.300000000701872978e+02,
    8.400000000935831395e+02,    1.050000000116978981e+03,    1.260000000140374823e+03,    1.470000000163770437e+03,
    1.680000000187166279e+03,    1.890000000210562121e+03,    2.100000000233957849e+02,    4.200000000467915697e+02,
    6.300000000701872978e+02,    8.400000000935831395e+02,    1.050000000116978981e+03,    1.260000000140374823e+03,
    1.470000000163770437e+03,    1.680000000187166279e+03,    1.890000000210562121e+03,    2.100000000233957849e+02,
    4.200000000467915697e+02,    6.300000000701872978e+02,    8.400000000935831395e+02,    1.050000000116978981e+03,
    1.260000000140374823e+03,    1.470000000163770437e+03,    1.680000000187166279e+03,    1.890000000210562121e+03,
    2.100000000233957849e+02,    4.200000000467915697e+02,    6.300000000701872978e+02,    8.400000000935831395e+02,
    1.050000000116978981e+03,    1.260000000140374823e+03,    1.470000000163770437e+03,    1.680000000187166279e+03,
    1.890000000210562121e+03,    2.100000000233957849e+02,    4.200000000467915697e+02,    6.300000000701872978e+02,
    8.400000000935831395e+02,    1.050000000116978981e+03,    1.260000000140374823e+03,    1.470000000163770437e+03,
    1.680000000187166279e+03,    1.890000000210562121e+03,    2.100000000233957849e+02,    4.200000000467915697e+02,
    6.300000000701872978e+02,    8.400000000935831395e+02,    1.050000000116978981e+03,    1.260000000140374823e+03,
    1.470000000163770437e+03,    1.680000000187166279e+03,    1.890000000210562121e+03,    2.100000000233957849e+02,
    4.200000000467915697e+02,    6.300000000701872978e+02,    8.400000000935831395e+02,    1.050000000116978981e+03,
    1.260000000140374823e+03,    1.470000000163770437e+03,    1.680000000187166279e+03,    1.890000000210562121e+03,
    2.100000000233957849e+02,    4.200000000467915697e+02,    6.300000000701872978e+02,    8.400000000935831395e+02,
    1.050000000116978981e+03,    1.260000000140374823e+03,    1.470000000163770437e+03,    1.680000000187166279e+03,
    1.890000000210562121e+03    ])

    turbineY = np.array([
    2.100000000233957849e+02,    2.100000000233957849e+02,    2.100000000233957849e+02,    2.100000000233957849e+02,
    2.100000000233957849e+02,    2.100000000233957849e+02,    2.100000000233957849e+02,    2.100000000233957849e+02,
    2.100000000233957849e+02,    4.200000000467915697e+02,    4.200000000467915697e+02,    4.200000000467915697e+02,
    4.200000000467915697e+02,    4.200000000467915697e+02,    4.200000000467915697e+02,    4.200000000467915697e+02,
    4.200000000467915697e+02,    4.200000000467915697e+02,    6.300000000701872978e+02,    6.300000000701872978e+02,
    6.300000000701872978e+02,    6.300000000701872978e+02,    6.300000000701872978e+02,    6.300000000701872978e+02,
    6.300000000701872978e+02,    6.300000000701872978e+02,    6.300000000701872978e+02,    8.400000000935831395e+02,
    8.400000000935831395e+02,    8.400000000935831395e+02,    8.400000000935831395e+02,    8.400000000935831395e+02,
    8.400000000935831395e+02,    8.400000000935831395e+02,    8.400000000935831395e+02,    8.400000000935831395e+02,
    1.050000000116978981e+03,    1.050000000116978981e+03,    1.050000000116978981e+03,    1.050000000116978981e+03,
    1.050000000116978981e+03,    1.050000000116978981e+03,    1.050000000116978981e+03,    1.050000000116978981e+03,
    1.050000000116978981e+03,    1.260000000140374823e+03,    1.260000000140374823e+03,    1.260000000140374823e+03,
    1.260000000140374823e+03,    1.260000000140374823e+03,    1.260000000140374823e+03,    1.260000000140374823e+03,
    1.260000000140374823e+03,    1.260000000140374823e+03,    1.470000000163770437e+03,    1.470000000163770437e+03,
    1.470000000163770437e+03,    1.470000000163770437e+03,    1.470000000163770437e+03,    1.470000000163770437e+03,
    1.470000000163770437e+03,    1.470000000163770437e+03,    1.470000000163770437e+03,    1.680000000187166279e+03,
    1.680000000187166279e+03,    1.680000000187166279e+03,    1.680000000187166279e+03,    1.680000000187166279e+03,
    1.680000000187166279e+03,    1.680000000187166279e+03,    1.680000000187166279e+03,    1.680000000187166279e+03,
    1.890000000210562121e+03,    1.890000000210562121e+03,    1.890000000210562121e+03,    1.890000000210562121e+03,
    1.890000000210562121e+03,    1.890000000210562121e+03,    1.890000000210562121e+03,    1.890000000210562121e+03,
    1.890000000210562121e+03    ])

    d1 = np.zeros((len(shearExp),3))
    t1 = np.zeros((len(shearExp),3))
    z1 = np.zeros((len(shearExp),1))
    d2 = np.zeros((len(shearExp),3))
    t2 = np.zeros((len(shearExp),3))
    z2 = np.zeros((len(shearExp),1))

    d1[0] =  np.array([3.870000000000000107e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d1[1] =  np.array([3.870000000000000107e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d1[2] =  np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 3.870000000000000107e+00])
    d1[3] =  np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 3.870000000000000107e+00])
    d1[4] =  np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 3.870000000000000107e+00])
    d1[5] =  np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 3.870000000000000107e+00])
    d1[6] =  np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 3.870000000000000107e+00])
    d1[7] =  np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 3.870000000000000107e+00])
    d1[8] =  np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 3.870000000000000107e+00])
    d1[9] =  np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 3.870000000000000107e+00])
    d1[10] = np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 4.969182209184133114e+00])
    d1[11] = np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 6.256759704848817272e+00])
    d1[12] = np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 6.299999999999999822e+00])
    d1[13] = np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 6.299999999999999822e+00])
    d1[14] = np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 6.299999999999999822e+00])
    d1[15] = np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 6.299999999999999822e+00])
    d1[16] = np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 6.299999999999999822e+00])
    d1[17] = np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 6.299999999999999822e+00])
    d1[18] = np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 6.299999999999999822e+00])
    d1[19] = np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 6.299999999999999822e+00])
    d1[20] = np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 6.299999999999999822e+00])
    d1[21] = np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 6.299999999999999822e+00])
    d1[22] = np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 6.299999999999999822e+00])

    d2[0] =  np.array([6.299999999999999822e+00, 6.283532982822681134e+00, 3.870000000000000107e+00])
    d2[1] =  np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 3.870000000000000107e+00])
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
    d2[16] = np.array([3.870000000000000107e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d2[17] = np.array([3.870000000000000107e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d2[18] = np.array([3.870000000000000107e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d2[19] = np.array([3.870000000000000107e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d2[20] = np.array([3.870000000000000107e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d2[21] = np.array([3.870000000000000107e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d2[22] = np.array([3.870000000000000107e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])

    t1[0] =  np.array([1.039868066560271746e-02, 7.534053439961167081e-03, 6.034130620527143087e-03])
    t1[1] =  np.array([1.036790102076544720e-02, 7.523613948676391092e-03, 6.031140986530564643e-03])
    t1[2] =  np.array([1.608066687408895787e-02, 1.063126837920704872e-02, 6.125096812488609391e-03])
    t1[3] =  np.array([1.609246824244178406e-02, 1.064827875219246864e-02, 6.133757285395315118e-03])
    t1[4] =  np.array([1.610510372939967863e-02, 1.066548436221015188e-02, 6.142834822951016155e-03])
    t1[5] =  np.array([1.611703544687922290e-02, 1.067928194999144889e-02, 6.151479955229965067e-03])
    t1[6] =  np.array([1.613277728063285121e-02, 1.070042878435663487e-02, 6.161097565041490087e-03])
    t1[7] =  np.array([1.614779704683266814e-02, 1.071817864639347875e-02, 6.170430515500142758e-03])
    t1[8] =  np.array([1.616694755628374225e-02, 1.073624977271069904e-02, 6.179905554197834619e-03])
    t1[9] =  np.array([1.618674256661935956e-02, 1.075450838223406327e-02, 6.189520206323717230e-03])
    t1[10] = np.array([1.706052077056069774e-02, 1.121013111753541534e-02, 6.521554823683666131e-03])
    t1[11] = np.array([1.828688154130817672e-02, 1.180326011346799660e-02, 7.050470507837406330e-03])
    t1[12] = np.array([1.837783283937378595e-02, 1.185349340145670435e-02, 7.093539169360264796e-03])
    t1[13] = np.array([1.842551342873833403e-02, 1.188369859967332472e-02, 7.117840531281457474e-03])
    t1[14] = np.array([1.847417410468257110e-02, 1.191422907604382936e-02, 7.142455455270586666e-03])
    t1[15] = np.array([1.852382459037161394e-02, 1.194508945909164392e-02, 7.167389013479657199e-03])
    t1[16] = np.array([1.857447454372813597e-02, 1.197628354563202470e-02, 7.192642864399047958e-03])
    t1[17] = np.array([1.862613436104190376e-02, 1.200781584720546531e-02, 7.218221176022879791e-03])
    t1[18] = np.array([1.867881475803672905e-02, 1.203969077501225275e-02, 7.244127500842015809e-03])
    t1[19] = np.array([1.873252686103245301e-02, 1.207191281027266208e-02, 7.270365423236354804e-03])
    t1[20] = np.array([1.878728221112936983e-02, 1.210448650553917760e-02, 7.296938559901771856e-03])
    t1[21] = np.array([1.884303534559656629e-02, 1.213736274542878864e-02, 7.323347251544467564e-03])
    t1[22] = np.array([1.889710438391621616e-02, 1.217044163937996970e-02, 7.350361611435615380e-03])

    t2[0] =  np.array([1.603936507612090112e-02, 1.059144272918025563e-02, 6.107741041214794463e-03])
    t2[1] =  np.array([1.606961789574988531e-02, 1.061437227444073615e-02, 6.114003055597901082e-03])
    t2[2] =  np.array([1.033006985209757034e-02, 7.512741105004674090e-03, 6.031783806451643942e-03])
    t2[3] =  np.array([1.030119703763889352e-02, 7.502999061039608911e-03, 6.030535766484089802e-03])
    t2[4] =  np.array([1.027995512410049758e-02, 7.487667255742544263e-03, 6.029427716194727185e-03])
    t2[5] =  np.array([1.025240543922914700e-02, 7.482967254835634456e-03, 6.028250405570547386e-03])
    t2[6] =  np.array([1.022570003990966141e-02, 7.472242133158848131e-03, 6.027077146898311699e-03])
    t2[7] =  np.array([1.022582688759751841e-02, 7.476461440035537739e-03, 6.028748973785151913e-03])
    t2[8] =  np.array([1.033669860837892196e-02, 7.527615125270223535e-03, 6.028699236235683136e-03])
    t2[9] =  np.array([1.043985068094396146e-02, 7.576093693295491219e-03, 6.031220410105825498e-03])
    t2[10] = np.array([1.083147507847888498e-02, 7.750674902824748920e-03, 6.041004167069965493e-03])
    t2[11] = np.array([1.139479871092592862e-02, 7.985848612572172059e-03, 6.056025004403451165e-03])
    t2[12] = np.array([1.152242169193920356e-02, 8.039507525734681040e-03, 6.059931482908637729e-03])
    t2[13] = np.array([1.163015402958355975e-02, 8.084971742233451608e-03, 6.063504548484375048e-03])
    t2[14] = np.array([1.174672362525712790e-02, 8.133971471802958070e-03, 6.067998867952092805e-03])
    t2[15] = np.array([1.186500428527816238e-02, 8.183435840440704198e-03, 6.072679052547659030e-03])
    t2[16] = np.array([1.199551313118128390e-02, 8.237513347102360869e-03, 6.077499991639392224e-03])
    t2[17] = np.array([1.211589812689813084e-02, 8.288193646857556357e-03, 6.085522064128802422e-03])
    t2[18] = np.array([1.223267031583233529e-02, 8.336072047237933977e-03, 6.088409143093919014e-03])
    t2[19] = np.array([1.236818911398247962e-02, 8.390971965629824905e-03, 6.094872997059178586e-03])
    t2[20] = np.array([1.248775341210037713e-02, 8.440473881957405097e-03, 6.102805148130188879e-03])
    t2[21] = np.array([1.264588677354046634e-02, 8.504102792470604258e-03, 6.111115623942801368e-03])
    t2[22] = np.array([1.285608841444851706e-02, 8.586565218510965394e-03, 6.120380734996906137e-03])

    z1[0] =  np.array([4.500000000000000000e+01])
    z1[1] =  np.array([4.500000000000000000e+01])
    z1[2] =  np.array([9.672624484661864130e+01])
    z1[3] =  np.array([9.674852218865173370e+01])
    z1[4] =  np.array([9.677149321285433814e+01])
    z1[5] =  np.array([9.679125642244804339e+01])
    z1[6] =  np.array([9.681922641168617361e+01])
    z1[7] =  np.array([9.684400310212855345e+01])
    z1[8] =  np.array([9.687150914166011262e+01])
    z1[9] =  np.array([9.689952331780345673e+01])
    z1[10] = np.array([9.954937232329862695e+01])
    z1[11] = np.array([1.033595905509363462e+02])
    z1[12] = np.array([1.035488216544208342e+02])
    z1[13] = np.array([1.035986873434123936e+02])
    z1[14] = np.array([1.036491160983019171e+02])
    z1[15] = np.array([1.037001097364555591e+02])
    z1[16] = np.array([1.037516679365039352e+02])
    z1[17] = np.array([1.038037920188030228e+02])
    z1[18] = np.array([1.038564829461529655e+02])
    z1[19] = np.array([1.039097417349353378e+02])
    z1[20] = np.array([1.039635694532137933e+02])
    z1[21] = np.array([1.040177175206220284e+02])
    z1[22] = np.array([1.040715133799653671e+02])

    z2[0] =  np.array([9.656526986565511095e+01])
    z2[1] =  np.array([9.670410102744128267e+01])
    z2[2] =  np.array([4.500000000000000000e+01])
    z2[3] =  np.array([4.500000000000000000e+01])
    z2[4] =  np.array([4.500000000000000000e+01])
    z2[5] =  np.array([4.500000000000000000e+01])
    z2[6] =  np.array([4.500000000000000000e+01])
    z2[7] =  np.array([4.517121113778350860e+01])
    z2[8] =  np.array([4.603358417581606687e+01])
    z2[9] =  np.array([4.683341994724375468e+01])
    z2[10] = np.array([4.935844500705672999e+01])
    z2[11] = np.array([5.266411393027567556e+01])
    z2[12] = np.array([5.347588141809205098e+01])
    z2[13] = np.array([5.416685847946650512e+01])
    z2[14] = np.array([5.489349518145457552e+01])
    z2[15] = np.array([5.561678426285779864e+01])
    z2[16] = np.array([5.639067789275797793e+01])
    z2[17] = np.array([5.710024486047326064e+01])
    z2[18] = np.array([5.778360247072507150e+01])
    z2[19] = np.array([5.853580627058917685e+01])
    z2[20] = np.array([5.920336235616605336e+01])
    z2[21] = np.array([6.004413255954054307e+01])
    z2[22] = np.array([6.111208148769705417e+01])

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
    BOS = np.zeros(23)
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
        BOS[k] = prob['BOS']
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
    print 'BOS: ', repr(BOS)
    print 'cost: ', repr(cost)
    print 'tower cost: ', repr(tower_cost)

    print 'wake loss: ', repr((idealAEP-AEP)/idealAEP*100.)

    # ideal AEP:  array([  3.82078602e+08,   3.85716125e+08,   3.90481718e+08,
    #          3.94357589e+08,   3.98327502e+08,   4.02389932e+08,
    #          4.06557796e+08,   4.11090155e+08,   4.16676706e+08,
    #          4.22413040e+08,   4.35180324e+08,   4.51752095e+08,
    #          4.59085652e+08,   4.66203741e+08,   4.73522860e+08,
    #          4.80894843e+08,   4.88102423e+08,   4.95239644e+08,
    #          5.02575663e+08,   5.10371401e+08,   5.18214407e+08,
    #          5.26854300e+08,   5.36524563e+08])
    # AEP:  array([  2.80035989e+08,   2.82808763e+08,   2.86541971e+08,
    #          2.89416950e+08,   2.92362102e+08,   2.95372911e+08,
    #          2.98468928e+08,   3.01685883e+08,   3.05181258e+08,
    #          3.08794370e+08,   3.18574170e+08,   3.31783785e+08,
    #          3.36937963e+08,   3.41888747e+08,   3.46965994e+08,
    #          3.52156349e+08,   3.57405575e+08,   3.62711772e+08,
    #          3.68169541e+08,   3.73824590e+08,   3.79607297e+08,
    #          3.85612738e+08,   3.91891075e+08])
    # COE:  array([ 70.77857607,  70.18949175,  69.57541948,  68.95639699,
    #         68.33483402,  67.71083165,  67.08652737,  66.46305892,
    #         65.8545207 ,  65.23359818,  64.67099754,  64.06256752,
    #         63.32716413,  62.58674053,  61.85116324,  61.12004057,
    #         60.40639161,  59.70147554,  58.99359549,  58.28586611,
    #         57.57663471,  56.87369582,  56.17583829])
    # cost:  array([  1.98205486e+10,   1.98502034e+10,   1.99362778e+10,
    #          1.99571501e+10,   1.99785157e+10,   1.99999454e+10,
    #          2.00232439e+10,   2.00509666e+10,   2.00975655e+10,
    #          2.01437679e+10,   2.06025094e+10,   2.12549211e+10,
    #          2.13373257e+10,   2.13977023e+10,   2.14602504e+10,
    #          2.15238103e+10,   2.15895811e+10,   2.16544280e+10,
    #          2.17196450e+10,   2.17886900e+10,   2.18565107e+10,
    #          2.19312215e+10,   2.20148097e+10])
    # tower cost:  array([ 24760398.07610117,  24848375.18453292,  25264440.25212284,
    #         25287781.53856856,  25311569.19737323,  25333010.70382905,
    #         25364001.1408332 ,  25416050.03331462,  25569324.09441949,
    #         25716611.45983376,  28446895.78417078,  32369469.63730412,
    #         32706137.2577231 ,  32900243.58479853,  33103300.03290809,
    #         33308795.10133406,  33526021.80572877,  33736181.75958945,
    #         33943174.85918789,  34166897.36634266,  34378913.99353539,
    #         34625255.07271547,  34916334.6944499 ])
    # wake loss:  array([ 26.70723034,  26.67955906,  26.61833886,  26.61052856,
    #         26.60258182,  26.59535272,  26.58634752,  26.61320637,
    #         26.75826278,  26.89752895,  26.79490489,  26.55622658,
    #         26.60673199,  26.66537888,  26.72666441,  26.77061232,
    #         26.77652112,  26.76035202,  26.74346015,  26.75440103,
    #         26.74705846,  26.80846723,  26.95747735])
