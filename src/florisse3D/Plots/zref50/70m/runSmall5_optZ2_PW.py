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
    3.499999999949981770e+02,    6.999999999899963541e+02,    1.049999999984994474e+03,    1.399999999979992708e+03,
    1.749999999974990942e+03,    2.099999999969988949e+03,    2.449999999964987182e+03,    2.799999999959985416e+03,
    3.149999999954983650e+03,    3.499999999949981770e+02,    6.999999999899963541e+02,    1.049999999984994474e+03,
    1.399999999979992708e+03,    1.749999999974990942e+03,    2.099999999969988949e+03,    2.449999999964987182e+03,
    2.799999999959985416e+03,    3.149999999954983650e+03,    3.499999999949981770e+02,    6.999999999899963541e+02,
    1.049999999984994474e+03,    1.399999999979992708e+03,    1.749999999974990942e+03,    2.099999999969988949e+03,
    2.449999999964987182e+03,    2.799999999959985416e+03,    3.149999999954983650e+03,    3.499999999949981770e+02,
    6.999999999899963541e+02,    1.049999999984994474e+03,    1.399999999979992708e+03,    1.749999999974990942e+03,
    2.099999999969988949e+03,    2.449999999964987182e+03,    2.799999999959985416e+03,    3.149999999954983650e+03,
    3.499999999949981770e+02,    6.999999999899963541e+02,    1.049999999984994474e+03,    1.399999999979992708e+03,
    1.749999999974990942e+03,    2.099999999969988949e+03,    2.449999999964987182e+03,    2.799999999959985416e+03,
    3.149999999954983650e+03,    3.499999999949981770e+02,    6.999999999899963541e+02,    1.049999999984994474e+03,
    1.399999999979992708e+03,    1.749999999974990942e+03,    2.099999999969988949e+03,    2.449999999964987182e+03,
    2.799999999959985416e+03,    3.149999999954983650e+03,    3.499999999949981770e+02,    6.999999999899963541e+02,
    1.049999999984994474e+03,    1.399999999979992708e+03,    1.749999999974990942e+03,    2.099999999969988949e+03,
    2.449999999964987182e+03,    2.799999999959985416e+03,    3.149999999954983650e+03,    3.499999999949981770e+02,
    6.999999999899963541e+02,    1.049999999984994474e+03,    1.399999999979992708e+03,    1.749999999974990942e+03,
    2.099999999969988949e+03,    2.449999999964987182e+03,    2.799999999959985416e+03,    3.149999999954983650e+03,
    3.499999999949981770e+02,    6.999999999899963541e+02,    1.049999999984994474e+03,    1.399999999979992708e+03,
    1.749999999974990942e+03,    2.099999999969988949e+03,    2.449999999964987182e+03,    2.799999999959985416e+03,
    3.149999999954983650e+03        ])

    turbineY = np.array([
    3.499999999949981770e+02,    3.499999999949981770e+02,    3.499999999949981770e+02,    3.499999999949981770e+02,
    3.499999999949981770e+02,    3.499999999949981770e+02,    3.499999999949981770e+02,    3.499999999949981770e+02,
    3.499999999949981770e+02,    6.999999999899963541e+02,    6.999999999899963541e+02,    6.999999999899963541e+02,
    6.999999999899963541e+02,    6.999999999899963541e+02,    6.999999999899963541e+02,    6.999999999899963541e+02,
    6.999999999899963541e+02,    6.999999999899963541e+02,    1.049999999984994474e+03,    1.049999999984994474e+03,
    1.049999999984994474e+03,    1.049999999984994474e+03,    1.049999999984994474e+03,    1.049999999984994474e+03,
    1.049999999984994474e+03,    1.049999999984994474e+03,    1.049999999984994474e+03,    1.399999999979992708e+03,
    1.399999999979992708e+03,    1.399999999979992708e+03,    1.399999999979992708e+03,    1.399999999979992708e+03,
    1.399999999979992708e+03,    1.399999999979992708e+03,    1.399999999979992708e+03,    1.399999999979992708e+03,
    1.749999999974990942e+03,    1.749999999974990942e+03,    1.749999999974990942e+03,    1.749999999974990942e+03,
    1.749999999974990942e+03,    1.749999999974990942e+03,    1.749999999974990942e+03,    1.749999999974990942e+03,
    1.749999999974990942e+03,    2.099999999969988949e+03,    2.099999999969988949e+03,    2.099999999969988949e+03,
    2.099999999969988949e+03,    2.099999999969988949e+03,    2.099999999969988949e+03,    2.099999999969988949e+03,
    2.099999999969988949e+03,    2.099999999969988949e+03,    2.449999999964987182e+03,    2.449999999964987182e+03,
    2.449999999964987182e+03,    2.449999999964987182e+03,    2.449999999964987182e+03,    2.449999999964987182e+03,
    2.449999999964987182e+03,    2.449999999964987182e+03,    2.449999999964987182e+03,    2.799999999959985416e+03,
    2.799999999959985416e+03,    2.799999999959985416e+03,    2.799999999959985416e+03,    2.799999999959985416e+03,
    2.799999999959985416e+03,    2.799999999959985416e+03,    2.799999999959985416e+03,    2.799999999959985416e+03,
    3.149999999954983650e+03,    3.149999999954983650e+03,    3.149999999954983650e+03,    3.149999999954983650e+03,
    3.149999999954983650e+03,    3.149999999954983650e+03,    3.149999999954983650e+03,    3.149999999954983650e+03,
    3.149999999954983650e+03    ])

    d1 = np.zeros((len(shearExp),3))
    t1 = np.zeros((len(shearExp),3))
    z1 = np.zeros((len(shearExp),1))
    d2 = np.zeros((len(shearExp),3))
    t2 = np.zeros((len(shearExp),3))
    z2 = np.zeros((len(shearExp),1))


    d1[0] =  np.array([6.299999999999999822e+00, 4.248317524691961111e+00, 3.870000000000000107e+00])
    d1[1] =  np.array([6.299999999999999822e+00, 4.568345200861559263e+00, 3.870000000000000107e+00])
    d1[2] =  np.array([6.299999999999999822e+00, 4.887800823271758688e+00, 3.870000000000000107e+00])
    d1[3] =  np.array([6.299999999999999822e+00, 5.287850877705333019e+00, 3.870000000000000107e+00])
    d1[4] =  np.array([6.299999999999999822e+00, 5.701766756012805359e+00, 3.870000000000000107e+00])
    d1[5] =  np.array([6.299999999999999822e+00, 6.076981212003778055e+00, 3.870000000000000107e+00])
    d1[6] =  np.array([6.299999999999999822e+00, 6.266028267909932836e+00, 3.870000000000000107e+00])
    d1[7] =  np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 3.870000000000000107e+00])
    d1[8] =  np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 3.870000000000000107e+00])
    d1[9] =  np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 3.870000000000000107e+00])
    d1[10] = np.array([4.067473673602680861e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d1[11] = np.array([4.262320479231457959e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d1[12] = np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 3.870000000000000107e+00])
    d1[13] = np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 3.870000000000000107e+00])
    d1[14] = np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 3.870000000000000107e+00])
    d1[15] = np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 3.870000000000000107e+00])
    d1[16] = np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 3.870000000000000107e+00])
    d1[17] = np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 3.870000000000000107e+00])
    d1[18] = np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 3.870000000000000107e+00])
    d1[19] = np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 3.870000000000000107e+00])
    d1[20] = np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 3.870000000000000107e+00])
    d1[21] = np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 3.870000000000000107e+00])
    d1[22] = np.array([6.299999999999999822e+00, 6.171976017266565862e+00, 3.870000000000000107e+00])

    d2[0] =  np.array([3.870000000000000107e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d2[1] =  np.array([3.870000000000000107e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d2[2] =  np.array([3.870000000000000107e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d2[3] =  np.array([3.870000000000000107e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d2[4] =  np.array([3.870000000000000107e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d2[5] =  np.array([3.870000000000000107e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d2[6] =  np.array([3.870000000000000107e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d2[7] =  np.array([3.870000000000000107e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d2[8] =  np.array([3.870000000000000107e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d2[9] =  np.array([3.870000000000000107e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d2[10] = np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 3.870000000000000107e+00])
    d2[11] = np.array([6.299999999999999822e+00, 6.299999999999999822e+00, 3.870000000000000107e+00])
    d2[12] = np.array([4.329675738309409105e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d2[13] = np.array([4.838967588908125173e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d2[14] = np.array([6.299999999999999822e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d2[15] = np.array([6.299999999999999822e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d2[16] = np.array([6.299999999999999822e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d2[17] = np.array([6.299999999999999822e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d2[18] = np.array([6.299999999999999822e+00, 3.870000000000000107e+00, 3.870000000000000107e+00])
    d2[19] = np.array([6.299999999999999822e+00, 3.980969264094510240e+00, 3.870000000000000107e+00])
    d2[20] = np.array([6.299999999999999822e+00, 4.612333313889989128e+00, 3.870000000000000107e+00])
    d2[21] = np.array([6.299999999999999822e+00, 5.259511707065152919e+00, 3.870000000000000107e+00])
    d2[22] = np.array([6.299999999999999822e+00, 6.171972876715316580e+00, 3.870000000000000107e+00])

    t1[0] =  np.array([1.401572202586587497e-02, 1.006670571708229329e-02, 6.092298165723616585e-03])
    t1[1] =  np.array([1.425323682782424564e-02, 1.014109215810133843e-02, 6.101816359110207207e-03])
    t1[2] =  np.array([1.455667573896904203e-02, 1.022925910472476801e-02, 6.111280516186148697e-03])
    t1[3] =  np.array([1.497706564103898005e-02, 1.034015230868408526e-02, 6.123609705971878522e-03])
    t1[4] =  np.array([1.541507804830020111e-02, 1.047030977029233222e-02, 6.135362314677703273e-03])
    t1[5] =  np.array([1.583383770329289952e-02, 1.059734179545879229e-02, 6.148879775281026333e-03])
    t1[6] =  np.array([1.608871556937273295e-02, 1.068646249281218590e-02, 6.160582082733988447e-03])
    t1[7] =  np.array([1.614779703901412719e-02, 1.071817861914984085e-02, 6.170430474363153110e-03])
    t1[8] =  np.array([1.616694755628685781e-02, 1.073624977270215726e-02, 6.179905554181607148e-03])
    t1[9] =  np.array([1.618674256708250991e-02, 1.075450838457583241e-02, 6.189520209980104923e-03])
    t1[10] = np.array([1.351398198238734638e-02, 8.965358143781075248e-03, 6.103407045103165923e-03])
    t1[11] = np.array([1.333020507994828879e-02, 9.047236687599572932e-03, 6.111479910585852771e-03])
    t1[12] = np.array([1.624966096511505295e-02, 1.081041403571585992e-02, 6.219215678832507727e-03])
    t1[13] = np.array([1.627179364821957491e-02, 1.082941499288864184e-02, 6.229374865491631684e-03])
    t1[14] = np.array([1.629456213610040380e-02, 1.084865577427902329e-02, 6.239757994618876265e-03])
    t1[15] = np.array([1.631778661069836073e-02, 1.086779005185389141e-02, 6.249609859758511500e-03])
    t1[16] = np.array([1.634183020418528837e-02, 1.088765386146237253e-02, 6.260851745592198661e-03])
    t1[17] = np.array([1.636564005339151764e-02, 1.090630209487086777e-02, 6.269765249162823247e-03])
    t1[18] = np.array([1.639070138313380998e-02, 1.092623274354105958e-02, 6.280544125541080019e-03])
    t1[19] = np.array([1.641724709432192467e-02, 1.094776656211436487e-02, 6.293797816724141722e-03])
    t1[20] = np.array([1.644343450641830989e-02, 1.096723201792258739e-02, 6.304399415341099648e-03])
    t1[21] = np.array([1.647054294782944808e-02, 1.098886669257803650e-02, 6.316540280009441677e-03])
    t1[22] = np.array([1.627947965569331509e-02, 1.094913308429615838e-02, 6.323019125568709979e-03])

    t2[0] =  np.array([1.039868075585684108e-02, 7.534053539940310101e-03, 6.034160799769682990e-03])
    t2[1] =  np.array([1.036790653660789145e-02, 7.523620013754236426e-03, 6.032973523204073450e-03])
    t2[2] =  np.array([1.033794660793812929e-02, 7.513295933810769725e-03, 6.031788920487569580e-03])
    t2[3] =  np.array([1.071571467400131775e-02, 7.682217984454486545e-03, 6.039072117668840313e-03])
    t2[4] =  np.array([1.121978024117374975e-02, 7.897802930016524503e-03, 6.047388649137678401e-03])
    t2[5] =  np.array([1.173881258141511695e-02, 8.107125490745475255e-03, 6.055473871128763327e-03])
    t2[6] =  np.array([1.224569997340450161e-02, 8.303157083319896367e-03, 6.064266242609811161e-03])
    t2[7] =  np.array([1.274449654204902124e-02, 8.491257859729019197e-03, 6.074821528339044603e-03])
    t2[8] =  np.array([1.331430364559677525e-02, 8.698858369547333991e-03, 6.086395547741092586e-03])
    t2[9] =  np.array([1.384591040315387434e-02, 8.892245200866219501e-03, 6.096948904547257880e-03])
    t2[10] = np.array([1.619785841947181046e-02, 1.077200595379744150e-02, 6.198926940945553681e-03])
    t2[11] = np.array([1.622807367057315861e-02, 1.079142883486960969e-02, 6.209108982760547929e-03])
    t2[12] = np.array([1.330667088171305461e-02, 9.078950710575460145e-03, 6.117109149184341878e-03])
    t2[13] = np.array([1.320547086763566511e-02, 9.403799845334839466e-03, 6.133199090752985624e-03])
    t2[14] = np.array([1.338403844504746504e-02, 1.029563881211177992e-02, 6.177978307866022374e-03])
    t2[15] = np.array([1.338298698764890313e-02, 1.030834707575896403e-02, 6.185676776551912379e-03])
    t2[16] = np.array([1.338345888708498618e-02, 1.032003019349964253e-02, 6.191794317478882190e-03])
    t2[17] = np.array([1.338269894528257981e-02, 1.033285564053555358e-02, 6.199282562532959853e-03])
    t2[18] = np.array([1.338358861546712526e-02, 1.034510434225722510e-02, 6.206252510010998487e-03])
    t2[19] = np.array([1.371075198551221543e-02, 1.029656703181307600e-02, 6.219902459794583727e-03])
    t2[20] = np.array([1.436509618438311549e-02, 1.040675235007696223e-02, 6.245607963248931381e-03])
    t2[21] = np.array([1.512612966246548382e-02, 1.061176337212555928e-02, 6.285310536613587448e-03])
    t2[22] = np.array([1.627947544598604482e-02, 1.094913716420056922e-02, 6.323030198577923519e-03])

    z1[0] =  np.array([8.283060566970584659e+01])
    z1[1] =  np.array([8.484575873143478475e+01])
    z1[2] =  np.array([8.696004681564738803e+01])
    z1[3] =  np.array([8.967988841826270630e+01])
    z1[4] =  np.array([9.254513127436511866e+01])
    z1[5] =  np.array([9.518806908037265657e+01])
    z1[6] =  np.array([9.657310082602916168e+01])
    z1[7] =  np.array([9.684400306908904099e+01])
    z1[8] =  np.array([9.687150914168105942e+01])
    z1[9] =  np.array([9.689952332052806128e+01])
    z1[10] = np.array([6.577527951301993880e+01])
    z1[11] = np.array([6.681935642079909599e+01])
    z1[12] = np.array([9.698639117809889854e+01])
    z1[13] = np.array([9.701625909579306040e+01])
    z1[14] = np.array([9.704666450829394364e+01])
    z1[15] = np.array([9.707698289048640561e+01])
    z1[16] = np.array([9.710877958899837381e+01])
    z1[17] = np.array([9.713896667129124296e+01])
    z1[18] = np.array([9.717109444970687093e+01])
    z1[19] = np.array([9.720558432152132866e+01])
    z1[20] = np.array([9.723797635300439879e+01])
    z1[21] = np.array([9.727245855718837220e+01])
    z1[22] = np.array([9.634456796004329249e+01])

    z2[0] =  np.array([4.500000000000000000e+01])
    z2[1] =  np.array([4.500000000000000000e+01])
    z2[2] =  np.array([4.500000000000000000e+01])
    z2[3] =  np.array([4.756064159459681662e+01])
    z2[4] =  np.array([5.076100064820892754e+01])
    z2[5] =  np.array([5.382590399327084896e+01])
    z2[6] =  np.array([5.665444621646624768e+01])
    z2[7] =  np.array([5.932156414640863318e+01])
    z2[8] =  np.array([6.222037974416934247e+01])
    z2[9] =  np.array([6.481739435645478409e+01])
    z2[10] = np.array([9.692201428715117117e+01])
    z2[11] = np.array([9.695683375585232966e+01])
    z2[12] = np.array([6.721139447448193494e+01])
    z2[13] = np.array([7.036443566907327352e+01])
    z2[14] = np.array([8.039889430544016591e+01])
    z2[15] = np.array([8.040745794328370266e+01])
    z2[16] = np.array([8.041536333801059300e+01])
    z2[17] = np.array([8.042390312298486776e+01])
    z2[18] = np.array([8.043310583114421775e+01])
    z2[19] = np.array([8.128777165185881870e+01])
    z2[20] = np.array([8.538476557324212024e+01])
    z2[21] = np.array([8.984019295579317088e+01])
    z2[22] = np.array([9.634455105291112886e+01])

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

    """OpenMDAO"""

    COE = np.zeros(23)
    AEP = np.zeros(23)
    idealAEP = np.zeros(23)
    cost = np.zeros(23)
    tower_cost = np.zeros(23)



    # for k in range(23):
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


    # ideal AEP:  array([  3.75302501e+08,   3.79214876e+08,   3.83570146e+08,
    #          3.91944070e+08,   4.02189471e+08,   4.13096114e+08,
    #          4.23407435e+08,   4.33030228e+08,   4.43003473e+08,
    #          4.53103559e+08,   4.59878960e+08,   4.68169114e+08,
    #          4.76680767e+08,   4.89609566e+08,   5.16585214e+08,
    #          5.24604202e+08,   5.32634669e+08,   5.40696885e+08,
    #          5.48435919e+08,   5.57871213e+08,   5.75112439e+08,
    #          5.93125481e+08,   6.12218301e+08])
    # AEP:  array([  3.19923444e+08,   3.24576206e+08,   3.29676031e+08,
    #          3.36980184e+08,   3.45567472e+08,   3.54649875e+08,
    #          3.62471569e+08,   3.68941857e+08,   3.75447333e+08,
    #          3.82209579e+08,   3.87231368e+08,   3.93671949e+08,
    #          4.01035340e+08,   4.09684404e+08,   4.25305760e+08,
    #          4.32481195e+08,   4.39742917e+08,   4.47090973e+08,
    #          4.54453261e+08,   4.63011223e+08,   4.76807936e+08,
    #          4.92220508e+08,   5.10828264e+08])
    # COE:  array([ 59.64522476,  59.30154206,  58.93576276,  58.57777555,
    #         58.15658917,  57.66448165,  57.10205046,  56.48523728,
    #         55.85478798,  55.19788356,  54.53639608,  53.86230653,
    #         53.15340991,  52.50130369,  51.91788904,  51.17450615,
    #         50.44715666,  49.73442888,  49.04442999,  48.38812811,
    #         47.77199125,  47.1607075 ,  46.54483319])
    # cost:  array([  1.90819057e+10,   1.92478695e+10,   1.94297084e+10,
    #          1.97395496e+10,   2.00970255e+10,   2.04507012e+10,
    #          2.06978698e+10,   2.08397684e+10,   2.09705312e+10,
    #          2.10971598e+10,   2.11182033e+10,   2.12040792e+10,
    #          2.13163958e+10,   2.15089653e+10,   2.20809772e+10,
    #          2.21320116e+10,   2.21837798e+10,   2.22358142e+10,
    #          2.22884011e+10,   2.24042464e+10,   2.27780646e+10,
    #          2.32134674e+10,   2.37764163e+10])
    # tower cost:  array([ 18034083.76835743,  18966233.51424982,  19990797.55853888,
    #         21740751.65476598,  23753763.21608059,  25729595.02630559,
    #         27037235.94703647,  27682023.60292311,  28244566.7699696 ,
    #         28775963.49347209,  28698549.56700625,  28999723.57504175,
    #         29450297.76843689,  30359500.33693677,  33442196.51148397,
    #         33492932.99028545,  33545139.60379894,  33595482.69446328,
    #         33649105.43458862,  34073121.97303188,  36000741.02755602,
    #         38301138.44859688,  41380058.71817537])
    # wake loss:  array([ 14.75584551,  14.40836683,  14.05065417,  14.02340041,
    #         14.07843906,  14.14833907,  14.39177983,  14.79997613,
    #         15.24957343,  15.64630828,  15.7971114 ,  15.91244762,
    #         15.86920065,  16.32426485,  17.66977676,  17.56047833,
    #         17.4400498 ,  17.31208631,  17.13648841,  17.00392278,
    #         17.09309272,  17.01241568,  16.56109219])
