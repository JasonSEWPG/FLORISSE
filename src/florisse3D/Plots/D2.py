import numpy as np
import matplotlib.pyplot as plt
from numpy import array

if __name__=="__main__":



    """Optimizations with Yaw coupled"""
    yawSpaceCOE = np.array([ 81.88629162,  73.21823851,  68.83250209,  65.61626217,
        63.87546503,  61.89125291,  61.15146622,  60.01989887,
        59.73347052,  58.9445006 ,  58.62114894,  58.15324368,
        58.01845345,  57.83597405,  57.50259471,  57.44054883,  57.23763116])
    yawSpaceAEP = np.array([  2.18391562e+08,   2.55385837e+08,   2.71870053e+08,
         2.86971100e+08,   2.96798153e+08,   3.13651354e+08,
         3.18516686e+08,   3.28738137e+08,   3.31877830e+08,
         3.39480208e+08,   3.42383233e+08,   3.47381379e+08,
         3.47900249e+08])
    yawSpaceheights = np.array([[ 117.58595613,   73.2       ],
       [ 117.58595613,   73.2       ],
       [ 117.58595613,   73.2       ],
       [ 113.69299078,   73.2       ],
       [ 104.13032293,  104.13032293],
       [ 104.13032293,  104.13032293],
       [ 104.13032293,  104.13032293],
       [ 104.13032293,  104.13032293],
       [ 104.13032293,  104.13032293],
       [ 104.13032293,  104.13032293],
       [ 104.13032293,  104.13032293],
       [ 104.13032293,  104.13032293],
       [ 104.13032293,  104.13032293]])

    yawShearCOE = np.array([ 68.77056799,  68.82055745,  68.80868109,  68.35494743,
        68.22177819,  68.50249552,  68.01516143,  67.29921548,
        66.82574983,  66.49249005,  65.99854078,  64.9560053 ,
        65.80006639,  64.20455639,  63.56926949,  63.09944655,
        63.14769018,  62.55352872,  62.20758301,  61.67656216,
        60.61048326,  60.86888438,  61.83500338])
    yawShearAEP = np.array([  2.73667653e+08,   2.71657401e+08,   2.72307808e+08,
         2.75565221e+08,   2.78197158e+08,   2.72330760e+08,
         2.77012815e+08,   2.85540759e+08,   2.91579117e+08,
         2.89867916e+08,   2.93687302e+08,   3.04231008e+08,
         2.98257224e+08,   3.08592129e+08,   3.13623470e+08,
         3.16804230e+08,   3.12698623e+08,   3.13235410e+08,
         3.18650888e+08,   3.21746814e+08,   3.31372967e+08,
         3.25494560e+08,   3.37314048e+08])
    yawShearZ = np.array([[ 117.61257748,   73.2       ],
       [ 117.59912676,   73.2       ],
       [ 117.58595613,   73.2       ],
       [ 117.57305988,   73.2       ],
       [ 117.56043249,   73.2       ],
       [ 110.58791006,  105.11292702],
       [ 110.56891751,  110.56891751],
       [ 117.5241106 ,  105.08023243],
       [ 117.51250655,  110.53186469],
       [ 117.50114606,  110.51379171],
       [ 117.49002448,  110.49601283],
       [ 117.47913733,  117.47913733],
       [ 117.46848022,  117.46848022],
       [ 117.4580489 ,  117.4580489 ],
       [ 117.44783922,  117.44783922],
       [ 117.43784716,  117.43784716],
       [ 117.42806881,  117.42806881],
       [ 117.41850035,  117.41850035],
       [ 117.40913808,  117.40913808],
       [ 117.39997837,  117.39997837],
       [ 117.39101772,  117.39101772],
       [ 117.3822527 ,  117.3822527 ],
       [ 117.37367998,  117.37367998]])
    """shear exponent"""
    shear_ex = np.array([0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.3])
    shear_ex3 = np.array([0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.19,0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.3])

    # COEdt = array([ 81.45263581,  81.4525218 ,  81.4524095 ,  81.45229885,
    #     81.45218991,  81.45208516,  81.4519818 ,  81.45187981,
    #     81.45177913,  81.45165764,  81.4515595 ,  81.45146258,
    #     81.45136685,  81.45127227,  81.45117883,  81.45110162,
    #     81.45100878,  81.45091706,  81.45082643,  81.45073688,
    #     81.45064837,  81.4505609 ,  81.45047446])
    COEdt = array([ 78.11753862,  78.11731938,  78.1171034 ,  78.11689063,
        78.11668113,  78.11647967,  78.11628092,  78.11608478,
        78.11589117,  78.11565753,  78.1154688 ,  78.11528241,
        78.11509831,  78.11491644,  78.11473675,  78.11458825,
        78.11440971,  78.11423333,  78.11405905,  78.11388683,
        78.11371663,  78.11354842,  78.11338217])
    AEPdt =  array([  2.48818899e+08,   2.48818899e+08,   2.48818899e+08,
         2.48818899e+08,   2.48818899e+08,   2.48818899e+08,
         2.48818899e+08,   2.48818899e+08,   2.48818899e+08,
         2.48818899e+08,   2.48818899e+08,   2.48818899e+08,
         2.48818899e+08,   2.48818899e+08,   2.48818899e+08,
         2.48818899e+08,   2.48818899e+08,   2.48818899e+08,
         2.48818899e+08,   2.48818899e+08,   2.48818899e+08,
         2.48818899e+08,   2.48818899e+08])

    # COEXYdt =  array([ 78.75358224,  78.26826421,  78.28203798,  78.6906634 ,
    #     78.51708069,  79.06391954,  78.5406008 ,  78.5649045 ,
    #     78.26314163,  78.66032115,  79.05816334,  79.01377933,
    #     79.01570726,  79.14235756,  79.07603554,  78.4401699 ,
    #     78.72549075,  78.31185125,  79.12523769,  79.09860859,
    #     79.14852951,  79.14618279,  78.51894438])

    COEXYdt = array([ 75.53810759,  75.07416729,  75.08722745,  75.47766282,
        75.31166267,  75.83419469,  75.33394541,  75.35707753,
        75.06858005,  75.44808081,  75.82821516,  75.78570451,
        75.78745632,  75.90840869,  75.84493515,  75.23714054,
        75.50974018,  75.11433404,  75.89161468,  75.86608167,
        75.91370907,  75.91138475,  75.3118472 ])
    AEPXYdt = array([  2.58064241e+08,   2.59799527e+08,   2.59749558e+08,
         2.58286808e+08,   2.58905628e+08,   2.56964596e+08,
         2.58820830e+08,   2.58733660e+08,   2.59815398e+08,
         2.58392683e+08,   2.56983131e+08,   2.57139249e+08,
         2.57132118e+08,   2.56685881e+08,   2.56918873e+08,
         2.59177166e+08,   2.58158490e+08,   2.59637148e+08,
         2.56744495e+08,   2.56837863e+08,   2.56662017e+08,
         2.56669968e+08,   2.58893062e+08])

    """start from grid"""
    # COEXYZdtSAME =  array([ 77.70870556,  77.78415534,  76.46461289,  77.16840282,
    #     76.44220922,  75.25059973,  74.17944405,  73.65485099,
    #     74.32510581,  72.93446076,  72.57036018,  71.79781517,
    #     70.82835498,  69.84800466,  68.94653113,  68.77363269,
    #     67.89230313,  67.60692781,  66.72408013,  66.06342008,
    #     65.37360994,  64.21647126,  64.15794101])
    COEXYZdtSAME = array([ 74.57934611,  74.71496979,  73.45959062,  74.13486992,
        73.49308207,  72.48459216,  71.59929751,  71.14774432,
        71.79056897,  70.44914426,  70.24214317,  69.64244383,
        68.85338098,  68.07735636,  67.2006935 ,  67.03242524,
        66.17547029,  65.89796688,  65.03964992,  64.39739938,
        63.72685966,  62.60202431,  62.54532012])
    AEPXYZdtSAME = array([  2.70297635e+08,   2.73597498e+08,   2.79086017e+08,
         2.76390540e+08,   2.80342190e+08,   2.87595452e+08,
         2.94563362e+08,   2.97738188e+08,   2.94791459e+08,
         3.00904915e+08,   3.04440287e+08,   3.09885544e+08,
         3.16431167e+08,   3.23477098e+08,   3.28115198e+08,
         3.29018416e+08,   3.33710275e+08,   3.35257740e+08,
         3.40140175e+08,   3.43888262e+08,   3.47891422e+08,
         3.54820304e+08,   3.55179522e+08])

    """start from optimized XY"""
    COEXYZdtSAME2 = array([ 78.3767004 ,  77.42545257,  76.96463989,  76.30378255,
        76.1724442 ,  75.58092761,  74.09043024,  74.18181665,
        73.26125908,  72.85085126,  72.68436541,  71.85451152,
        70.77431384,  69.56157314,  68.7742278 ,  68.33181978,
        67.69119913,  66.70034969,  66.96945159,  66.32523844,
        65.01629099,  65.09445876,  63.94922392])
    AEPXYZdtSAME2 = array([  2.67864387e+08,   2.74936980e+08,   2.77116484e+08,
         2.79773752e+08,   2.81393223e+08,   2.86194903e+08,
         2.94913641e+08,   2.95433302e+08,   2.99461706e+08,
         3.01281873e+08,   3.03918948e+08,   3.09791456e+08,
         3.16695615e+08,   3.24937331e+08,   3.29017393e+08,
         3.31354648e+08,   3.34800063e+08,   3.40274006e+08,
         3.38768807e+08,   3.42393022e+08,   3.50001693e+08,
         3.49538781e+08,   3.56461213e+08])

    """start from optimized XYZdt"""
    COEXYZdtSAME3 = array([ 72.93621494,  72.6861249 ,  73.30180957,  72.54675087,
        71.25745471,  71.23776417,  69.75637229,  69.60262763,
        69.60881241,  67.42674342,  68.48377407,  67.59865398,
        66.60208726,  66.58198354,  66.24542718,  64.83924876,
        64.47880509,  63.65602641,  63.26185397,  62.57743428,  62.571697  ])
    AEPXYZdtSAME3 = array([  2.73102920e+08,   2.77164466e+08,   2.74987575e+08,
         2.78151284e+08,   2.84519685e+08,   2.86766228e+08,
         2.95801426e+08,   2.97090353e+08,   2.97040702e+08,
         3.13327109e+08,   3.15958187e+08,   3.24961641e+08,
         3.31362099e+08,   3.31469198e+08,   3.33322062e+08,
         3.41301465e+08,   3.43408460e+08,   3.48318620e+08,
         3.50721977e+08,   3.54974819e+08,   3.55081556e+08])


    COEXYZdt = array([ 72.01145694,  71.74742891,  71.91698975,  71.28161109,
        71.12051912,  71.50771123,  70.16476432,  70.07984294,
        70.37275484,  70.34286495,  70.37928496,  69.07477523,
        68.97432682,  67.79741567,  66.60208859,  66.58198354,
        66.24542718,  64.83924981,  64.47888633,  63.6560277 ,
        63.26234891,  62.57743428,  61.74158465])
    AEPXYZdt = array([  2.84222073e+08,   2.85402344e+08,   2.84659377e+08,
         2.87427833e+08,   2.88133541e+08,   2.86421598e+08,
         2.92420928e+08,   2.92803891e+08,   2.91464529e+08,
         2.91549744e+08,   2.92697690e+08,   3.02087142e+08,
         3.13586920e+08,   3.23411661e+08,   3.31362092e+08,
         3.31469198e+08,   3.33322062e+08,   3.41301459e+08,
         3.43407982e+08,   3.48318612e+08,   3.50718938e+08,
         3.54974819e+08,   3.20800239e+08])
    diff = array([  5.29550675e+01,   5.29490683e+01,   5.29434077e+01,
         5.29380803e+01,   5.29330814e+01,   5.29284061e+01,
         5.29240497e+01,   5.29200077e+01,   5.29162758e+01,
         5.29128498e+01,   4.79016366e+01,   3.62529282e+01,
         7.30758658e+00,   1.92354748e+00,   3.84545729e-11,
         8.07744982e-11,   1.81046289e-11,   3.91224830e-11,
         5.78381787e-11,   7.97228950e-12,   1.82228277e-07,
         4.47499815e-11,   1.93056348e+01])

    COEXYZdt2 = array([ 71.42091593,  71.30160304,  71.52016331,  71.21241189,
        70.61924292,  70.88984209,  71.304625  ,  71.28183133,
        71.29341052,  68.98561281,  67.79418982,  66.60209154,
        66.58198338,  66.24543709,  64.83924937,  64.4788047 ,
        63.65603519,  63.26191237,  62.57714399,  62.55959124])
    AEPXYZdt2 = array([  2.85104724e+08,   2.87354133e+08,   2.84657869e+08,
         2.85996962e+08,   2.88620584e+08,   2.87409255e+08,
         2.93048548e+08,   2.97125610e+08,   2.97039855e+08,
         3.14804322e+08,   3.24962048e+08,   3.31362075e+08,
         3.31469198e+08,   3.33321935e+08,   3.41301461e+08,
         3.43408462e+08,   3.48318565e+08,   3.50721620e+08,
         3.54976644e+08,   3.55089728e+08])
    diffXYZdt2 = array([  5.29550675e+01,   5.29490683e+01,   5.29434077e+01,
         5.29380803e+01,   5.29330814e+01,   5.29284061e+01,
         8.37003444e+00,   1.02389208e-10,   2.28794761e-12,
         5.40472585e+00,   2.11315410e-10,   1.36781864e-07,
         4.15282386e-09,   1.11900149e-04,   1.78346227e-11,
         8.18387491e-08,   8.11105551e-07,   5.75539616e-12,
         1.39124268e-11,   3.04001446e-09])

    AEPdt /= 10.**6
    AEPXYdt /= 10.**6
    AEPXYZdtSAME /= 10.**6
    AEPXYZdt /= 10.**6

    COEsame = np.zeros(len(shear_ex))
    AEPsame = np.zeros(len(shear_ex))
    for i in range(len(shear_ex)):
        if diff[i] < 1.:
            COEsame[i] = COEXYZdt[i]
            AEPsame[i] = AEPXYZdt[i]
        else:
            COEsame[i] = COEXYZdtSAME[i]
            AEPsame[i] = AEPXYZdtSAME[i]

        if COEsame[i] < COEXYZdt[i]:
            COEXYZdt[i] = COEsame[i]

    # COEsame[len(shear_ex)-1] = COEXYZdt[len(shear_ex)-1]
    # AEPsame[len(shear_ex)-1] = AEPXYZdt[len(shear_ex)-1]
    # plt.plot(shear_ex, COEXYZdtSAME,'ob')
    # plt.plot(shear_ex, COEXYZdtSAME2, 'or')
    # plt.plot(shear_ex3, COEXYZdtSAME3, 'oy')
    # plt.plot(shear_ex, COEXYZdt, 'ow')
    # plt.show()



    # plt.figure(1)
    # plt.plot(shear_ex, COEdt, 'c', label='Original Grid')
    # plt.plot(shear_ex, COEXYdt, 'k', label='Optimized Layout')
    # plt.plot(shear_ex, COEsame, 'r', label='1 Height Group')
    # plt.plot(shear_ex, COEXYZdt, 'b', label='2 Height Groups')
    # plt.plot(shear_ex, COEdt, 'oc')
    # plt.plot(shear_ex, COEXYdt, 'ok')
    # plt.plot(shear_ex, COEXYZdt, 'ob')
    # plt.plot(shear_ex, COEsame, 'or')
    #
    # plt.legend(loc=3)
    # plt.title('Wind Farm COE vs. Shear Exponent', fontsize=15)
    # plt.xlabel(r'Shear Exponent ($\alpha$)', fontsize=15)
    # plt.ylabel('COE ($/MWhr)', fontsize=15)

    dt = np.zeros(10)
    XYdt = np.zeros(10)
    same = np.zeros(10)
    XYZdt = np.zeros(10)
    dif = np.zeros(10)
    yaw = np.zeros(10)
    shear = np.zeros(10)

    for i in range(10):
        dt[i] = COEdt[2*i+1]
        XYdt[i] = COEXYdt[2*i+1]
        same[i] = COEXYZdtSAME[2*i+1]
        XYZdt[i] = COEXYZdt[2*i+1]
        dif[i] = diff[2*i+1]
        shear[i] = shear_ex[2*i+1]
        if dif[i] < 0.1:
            same[i] = XYZdt[i]

    z_shear = array([[ 126.15506754,   73.2       ],
       [ 126.14906835,   73.2       ],
       [ 126.14340772,   73.2       ],
       [ 126.13808035,   73.2       ],
       [ 126.13308143,   73.2       ],
       [ 126.1284061 ,   73.2       ],
       [ 126.12404971,   73.2       ],
       [ 126.12000774,   73.2       ],
       [ 126.11627583,   73.2       ],
       [ 126.11284977,   73.2       ],
       [ 126.10972547,   78.20808882],
       [ 126.10689896,   89.85397077],
       [ 126.10436641,  118.79677983],
       [ 126.10212412,  124.17857664],
       [ 126.10016849,  126.10016849],
       [ 126.09849604,  126.09849604],
       [ 126.0971034 ,  126.0971034 ],
       [ 126.0959873 ,  126.0959873 ],
       [ 126.09514459,  126.09514459],
       [ 126.09457221,  126.09457221],
       [ 126.09426663,  126.09426644],
       [ 126.09422667,  126.09422667],
       [ 123.79078144,  104.48514664]])

    z = np.zeros((10,2))
    for i in range(10):
        dt[i] = COEdt[2*i]
        XYdt[i] = COEXYdt[2*i]
        same[i] = COEXYZdtSAME[2*i]
        XYZdt[i] = COEXYZdt[2*i]
        dif[i] = diff[2*i]
        shear[i] = shear_ex[2*i]
        z[i] = z_shear[2*i]
        yaw[i] = yawShearCOE[2*i]
        if dif[i] < 0.1:
            same[i] = XYZdt[i]
    z1 = z[:,0]
    z2 = z[:,1]

    benefit = (same-XYZdt)/same*100.

    # plt.plot(shear,z1,'b', label='Height Group 1')
    # plt.plot(shear,z2,'r', label='Height Group 2')
    # plt.plot(shear,z1,'ob')
    # plt.plot(shear,z2,'or')
    # plt.axis([0.078,0.262,0,130])
    # plt.xlabel(r'Shear Exponent $\alpha$',fontsize=15)
    # plt.ylabel('Turbine Height (m)', fontsize=15)
    # plt.legend(loc=3)
    # plt.show()

    #
    # l = len(shear_ex)-1
    # dt = np.zeros(l)
    # XYdt = np.zeros(l)
    # same = np.zeros(l)
    # XYZdt = np.zeros(l)
    # dif = np.zeros(l)
    # shear = np.zeros(l)
    #
    # for i in range(l):
    #     dt[i] = COEdt[i]
    #     XYdt[i] = COEXYdt[i]
    #     same[i] = COEXYZdtSAME[i]
    #     XYZdt[i] = COEXYZdt[i]
    #     dif[i] = diff[i]
    #     shear[i] = shear_ex[i]
    #     if dif[i] < 0.1:
    #         same[i] = XYZdt[i]

    # print shear
    # print dt
    #
    plt.figure(1)
    plt.plot(shear, dt, 'c', label='Original Grid')
    plt.plot(shear, XYdt, 'k', label='Optimized Layout')
    plt.plot(shear, same, 'r', label='1 Height Group')
    plt.plot(shear, XYZdt, 'b', label='2 Height Groups')
    plt.plot(shear, yaw, 'm', label='2 Height Groups with yaw')
    plt.plot(shear, dt, 'oc')
    plt.plot(shear, XYdt, 'ok')
    plt.plot(shear, XYZdt, 'ob')
    plt.plot(shear, same, 'or')
    plt.plot(shear, yaw, 'om')
    plt.axis([0.079,0.262,58,80])

    plt.legend(loc=3)
    # plt.title('Wind Farm COE vs. Shear Exponent', fontsize=15)
    plt.xlabel(r'Shear Exponent ($\alpha$)', fontsize=15)
    plt.ylabel('COE ($/MWhr)', fontsize=15)

    # plt.figure(2)
    # benefit = benefit*-1.
    # plt.plot(shear, benefit,linewidth=2)
    # plt.plot(shear, benefit,'ob')
    # plt.xlabel(r'Shear Exponent ($\alpha$)', fontsize=15)
    # plt.ylabel('COE Decrease (%)', fontsize=15)
    # plt.axis([0.078,0.262,-4,0.2])
    #
    #
    # plt.show()
    #
    # # plt.figure(2)
    # # plt.plot(shear_ex, AEPdt, 'c', label='Original Grid')
    # # plt.plot(shear_ex, AEPXYdt, 'k', label='Optimized Layout')
    # # plt.plot(shear_ex, AEPsame, 'r', label='1 Height Group')
    # # plt.plot(shear_ex, AEPXYZdt, 'b', label='2 Height Groups')
    # # plt.plot(shear_ex, AEPdt, 'oc')
    # # plt.plot(shear_ex, AEPXYdt, 'ok')
    # # plt.plot(shear_ex, AEPXYZdt, 'ob')
    # # plt.plot(shear_ex, AEPsame, 'or')
    # # plt.title('Wind Farm AEP vs. Shear Exponent')
    # # plt.xlabel(r'Shear Exponent ($\alpha$)', fontsize=15)
    # # plt.ylabel('AEP (GWhrs)', fontsize=15)
    # #
    # plt.figure(4)
    # plt.plot(shear, dif,'k')
    # plt.plot(shear, dif,'ok')
    # # plt.title('Difference in Height between \n Height Groups vs. Shear Exponent')
    # plt.xlabel(r'Shear Exponent ($\alpha$)', fontsize=15)
    # plt.ylabel('Height Difference (m)', fontsize=15)
    # plt.axis([0.078,0.262,-0.5,60])
    # plt.show()
    # #
    # #
    # # #
    """spacing"""
    D = 126.4
    R = 1/2.
    A = np.pi*R**2*25.
    print A
    spacing = np.array([2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0,8.5,9.0,9.5,10.0])
    area = (4.*spacing)**2
    print area
    density = A/area
    print density
    spacingSAME2 = np.array([2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0,9.0,10.0])

    spacingSAME = np.array([2.0,2.5,3.0,3.5,4.0,5.0,5.5,6.0,6.5,7.0,7.5,8.0,8.5,9.0,9.5,10.0])


    # COEdt = array([ 103.28319704,   89.71764439,   81.4524095 ,   76.0049225 ,
    #      72.20093401,   69.46741984,   67.43221966,   65.85488169,
    #      64.63411473,   63.64763946,   62.85668712,   62.21672817,
    #      61.67944119,   61.23186106,   60.86247103,   60.55219726,
    #      60.28808344])
    COEdt = array([ 98.98151412,  86.01644758,  78.1171034 ,  72.91074881,
        69.27515883,  66.66264705,  64.71754325,  63.21003253,
        62.04328229,  61.10047778,  60.34454049,  59.73291219,
        59.21941063,  58.79164464,  58.43860732,  58.14206921,  57.88964757])
    AEPdt = array([  1.92919698e+08,   2.24221308e+08,   2.48818899e+08,
         2.68211324e+08,   2.83648733e+08,   2.95886535e+08,
         3.05706603e+08,   3.13777631e+08,   3.20322641e+08,
         3.25814531e+08,   3.30355832e+08,   3.34123903e+08,
         3.37354463e+08,   3.40093724e+08,   3.42388182e+08,
         3.44339501e+08,   3.46018128e+08])


    # COEXYdt = array([ 103.28311334,   87.32386708,   78.28203701,   74.04863279,
    #      70.83873905,   68.46501359,   66.61592382,   65.14173513,
    #      64.2504091 ,   63.32380145,   62.61957038,   62.02755199,
    #      61.57413082,   61.16765004,   60.82651031,   60.50686264,
    #      60.25738229])
    COEXYdt = array([ 98.98135316,  83.72880606,  75.08722745,  71.041221  ,
        67.97341802,  65.70476961,  63.93753273,  62.52860157,
        61.67673134,  60.79114142,  60.11806841,  59.55227249,
        59.11892262,  58.73043518,  58.40439684,  58.09889902,  57.86046215])
    AEPXYdt = array([  1.92919698e+08,   2.30830716e+08,   2.59749563e+08,
         2.75935124e+08,   2.89618727e+08,   3.00643910e+08,
         3.09831730e+08,   3.17569102e+08,   3.22437619e+08,
         3.27659666e+08,   3.31742916e+08,   3.35255289e+08,
         3.37996012e+08,   3.40491375e+08,   3.42614230e+08,
         3.44627506e+08,   3.46215353e+08])

    """start from grid"""
    COEXYZdtSAME = array([ 101.22616829,   85.63034162,   77.9119994 ,   72.73023583,
         69.65477937,   0., 65.5562663 ,   64.20843677,   63.35744793,
         62.43197481,   61.70721174,   61.15894544,   60.70345538,
         60.29836118,   59.92977624,   59.63399862,   59.41374138])
    AEPXYZdtSAME = array([  2.06431658e+08,   2.46917811e+08,   2.73460194e+08,
         2.94730099e+08,   3.08994521e+08, 0.,   3.30298098e+08,
         3.37960705e+08,   3.42984524e+08,   3.48620387e+08,
         3.53164989e+08,   3.56682376e+08,   3.59658287e+08,
         3.62346973e+08,   3.64828508e+08,   3.66844578e+08,
         3.68360422e+08])

    """start from optimized XY"""
    COEXYZdtSAME2 = array([ 101.22612951,   84.84060268,   76.98408909,   72.73080971,
         69.65373445,   67.33456513,   65.54654391,   64.15941339,
         63.10905987,   62.33186638,   61.67772054,   61.1019099 ,
         60.69136874,   59.94628956,   59.41199762])
    AEPXYZdtSAME2 = array([  2.06431791e+08,   2.49394619e+08,   2.77040438e+08,
         2.94727560e+08,   3.08999602e+08,   3.20704420e+08,
         3.30352126e+08,   3.38246117e+08,   3.44479167e+08,
         3.49241139e+08,   3.53352423e+08,   3.57052314e+08,
         3.59737931e+08,   3.64716603e+08,   3.68372473e+08])

    """start from optimized XYZdt"""
    COEXYZdtSAME3 = array([ 95.61875902,  79.67774676,  73.30178301,  68.80802107,
        65.84367873,  63.72046981,  61.91331284,  61.67470239,
        60.76963333,  59.97541805,  59.26603084,  58.85134152,
        58.39543369,  58.02323618,  57.59274578,  57.54589847,  57.1611293 ])
    AEPXYZdtSAME3 = array([  2.06421905e+08,   2.51155059e+08,   2.74987683e+08,
         2.94696887e+08,   3.09321506e+08,   3.20721417e+08,
         3.31107791e+08,   3.38278226e+08,   3.43879529e+08,
         3.48949839e+08,   3.53606706e+08,   3.56387009e+08,
         3.59494560e+08,   3.62071995e+08,   3.65099582e+08,
         3.65432111e+08,   3.68186350e+08])


    COEXYZdt = array([ 88.26852849,  77.09434726,  71.91698975,  68.20810932,
        66.03375447,  64.30941945,  62.91512361,  61.67470239,
        60.76963333,  59.97541805,  59.26603084,  58.85134152,
        58.39543369,  58.02323618,  57.59274578,  57.54589847,  57.1611293 ])
    COE = array([ 98.98135316,  83.72880606,  75.08722675,  71.041221  ,
        67.97341802,  65.70476961,  63.93753273,  62.52860157,
        61.67673134,  60.79114142,  60.11806841,  59.55227249,
        59.11892262,  58.73043518,  58.40439684,  58.09889902,  57.86046215])
    AEPXYZdt = array([  2.30734453e+08,   2.63897472e+08,   2.84659377e+08,
         3.01661195e+08,   3.11449135e+08,   3.17309987e+08,
         3.23585898e+08,   3.38278226e+08,   3.43879529e+08,
         3.48949839e+08,   3.53606706e+08,   3.56387009e+08,
         3.59494560e+08,   3.62071995e+08,   3.65099582e+08,
         3.65432111e+08,   3.68186350e+08])
    diff = array([  5.53250083e+01,   5.29434077e+01,   5.29434077e+01,
         5.29434077e+01,   5.16313791e+01,   4.76398316e+01,
         4.57559798e+01,   2.42437181e-11,   2.80664381e-11,
         2.27373675e-13,   1.00897068e-11,   8.93578544e-11,
         2.71000999e-11,   6.21014351e-12,   7.43227702e-12,
         1.27613475e-11,   8.80930884e-11])

    """start from optimized SAME"""
    COEXYZdt2 = array([ 88.28798558,  76.89232881,  71.94036489,  68.39429162,
        66.87604914,  64.71480338,  62.87795523,  61.67470228,
        60.76963367,  59.9754108 ,  59.26603564,  58.851342  ,
        58.39543351,  58.02318026,  57.59274576,  57.54589892,  57.16112898])
    AEPXYZdt2 =  array([  2.30814541e+08,   2.63053088e+08,   2.82840856e+08,
         2.98944025e+08,   3.09322874e+08,   3.20730177e+08,
         3.31108086e+08,   3.38278227e+08,   3.43879527e+08,
         3.48949886e+08,   3.53606674e+08,   3.56387006e+08,
         3.59494561e+08,   3.62072385e+08,   3.65099582e+08,
         3.65432108e+08,   3.68186353e+08])
    diffXYZdt2 =  array([  5.53904004e+01,   5.29434077e+01,   5.29434077e+01,
         5.29434077e+01,   6.84963197e-12,   1.57740487e-11,
         2.41584530e-12,   1.42108547e-13,   1.70530257e-13,
         1.42108547e-13,   0.00000000e+00,   2.41584530e-13,
         2.84217094e-14,   1.70530257e-13,   2.27373675e-13,
         2.41868747e-11,   4.26325641e-14])


    diff =  array([ 88.26852849,  77.09434726,  71.91698975,  68.20810932,
     66.03375447,  64.30767053,  62.90584714,  61.68060919,
     60.76963338,  59.9754228 ,  59.22772038])

    DIFF =  array([  5.53250083e+01,   5.29434077e+01,   5.29434077e+01,
          5.29434077e+01,   5.16313791e+01,   4.76398316e+01,
          4.57559798e+01,   1.13544729e-11,   1.17950094e-12,
          5.25801624e-13,   5.26227950e-11])

    same = array([ 97.16351701,  82.25255559,  74.84372926,  69.88471818,
        66.93921376,  64.77180096, 63.046665, 61.72751939,  60.91229563,
        60.02632176,  59.32678852])

    # space_same = array([2.0,2.5,3.0,3.5,4.0,4.5,5.5,6.0,6.5,7.0])
    # space = array([2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0])
    zs = array([[ 128.52500827,   73.2       ],
       [ 126.14340766,   73.2       ],
       [ 126.14340772,   73.2       ],
       [ 126.14340766,   73.2       ],
       [ 124.83137908,   73.2       ],
       [ 120.83983157,   73.2       ],
       [ 118.95597978,   73.2       ],
       [ 110.35342496,  110.35342496],
       [ 110.35342496,  110.35342496],
       [ 110.35342496,  110.35342496],
       [ 110.35342496,  110.35342496],
       [ 110.35342496,  110.35342496],
       [ 110.35342496,  110.35342496]])
    n = 11
    space = np.zeros(n)
    dt = np.zeros(n)
    XYdt = np.zeros(n)
    XYZdt = np.zeros(n)
    yaw = np.zeros(n)
    z = np.zeros((n,2))
    for i in range(n):
        space[i] = density[i]
        dt[i] = COEdt[i]
        XYdt[i] = COEXYdt[i]
        XYZdt[i] = COEXYZdt[i]
        yaw[i] = yawSpaceCOE[i]
        z[i] = zs[i]

    benefit = (same-XYZdt)/same*100.
    # plt.figure(3)
    # benefit = benefit*-1.
    # plt.plot(space,benefit,linewidth=2)
    # plt.plot(space,benefit,'ob')
    # plt.xlabel('Turbine Density (Total Rotor Area/Farm Area)', fontsize=15)
    # plt.ylabel('COE Decrease (%)', fontsize=15)
    # plt.show()
    # z1 = z[:,0]
    # z2 = z[:,1]
    # plt.plot(space,z1,'b', label='Height Group 1')
    # plt.plot(space,z2,'r', label='Height Group 2')
    # plt.plot(space,z1,'ob')
    # plt.plot(space,z2,'or')
    # plt.axis([0.024,0.309,0,130])
    # plt.xlabel('Turbine Density (Total Rotor Area/Farm Area)', fontsize=15)
    # plt.ylabel('Turbine Height (m)', fontsize=15)
    # plt.legend(loc=3)
    # plt.show()
    # print len(space_same)
    plt.figure(2)
    plt.plot(space, dt, 'c', label='Original Grid')
    plt.plot(space, XYdt, 'k', label='Optimized Layout')
    plt.plot(space, same, 'r', label='1 Height Group')
    plt.plot(space, diff, 'b', label='2 Height Groups')
    plt.plot(space, yaw, 'm', label='2 Height Groups with coupled yaw')
    plt.plot(space, dt, 'oc')
    plt.plot(space, XYdt, 'ok')
    plt.plot(space, diff, 'ob')
    plt.plot(space, same, 'or')
    plt.plot(space, yaw, 'om')
    plt.legend(loc=4)
    plt.xlabel('Turbine Density (Total Rotor Area/Farm Area)', fontsize=15)
    plt.ylabel('COE ($/MWhr)', fontsize=15)
    plt.axis([0.024,0.309,50,100])
    #
    # plt.figure(3)
    # plt.plot(space, DIFF,'k')
    # plt.plot(space,DIFF,'ok')
    # plt.xlabel('Turbine Density (Total Rotor Area/Farm Area)', fontsize=15)
    # plt.ylabel('Height Difference (m)', fontsize=15)
    # plt.axis([0.024,0.309,-0.5,60])
    #
    #
    #
    # print space
    # #
    plt.show()
    #
    #
    # # for i in range(len(shear)):
    # #     if DIFF[i] < 0.1:
    # #         same[i] = diff[i]
    #
    # # plt.figure(3)
    # # plt.plot(shear_same, same)
    # # plt.plot(shear,diff)
    # plt.show()
    #
    # #
    # #
    # #
    # # COE = np.zeros(len(spacing))
    # # AEP = np.zeros(len(spacing))
    # # DIFF = np.zeros(len(spacing))
    # # space = np.zeros(len(spacing))
    # # for i in range(len(spacing)):
    # #     space[i] = (4*(spacing[i]))**2
    # #     if spacing_XYZdt_COE[i] >= spacing_XYZdtSAME3_COE[i]:
    # #         COE[i] = spacing_XYZdtSAME3_COE[i]
    # #         AEP[i] = spacing_XYZdtSAME3_AEP[i]
    # #         DIFF[i] = 0
    # #     else:
    # #         COE[i] = spacing_XYZdt_COE[i]
    # #         AEP[i] = spacing_XYZdt_AEP[i]
    # #         DIFF[i] = 44.3859
    # #
    # # DIFF[6] = 0
    # #
    # # spacing = space
    # #
    # #
    # # COEsame = np.zeros(len(spacing))
    # # AEPsame = np.zeros(len(spacing))
    # # for i in range(len(spacing)):
    # #     if diff[i] < 0.1:
    # #         COEsame[i] = COEXYZdt[i]
    # #         AEPsame[i] = AEPXYZdt[i]
    # #     else:
    # #         COEsame[i] = COEXYZdtSAME[i]
    # #         AEPsame[i] = AEPXYZdtSAME[i]
    # #
    # #
    # plt.figure(4)
    # plt.plot(spacing, COEdt, 'c', label='Original Grid')
    # plt.plot(spacing, COEXYdt, 'k', label='Optimized Layout')
    # plt.plot(spacing, COEsame, 'r', label='1 Height Group')
    # # plt.plot(spacingSAME2, COEXYZdtSAME2, 'm', label='1 Height Group 2222')
    # # plt.plot(spacing, COEXYZdtSAME3, 'y', label='1 Height Group 3333')
    # plt.plot(spacing, COEXYZdt, 'b', label='2 Height Groups')
    # plt.plot(spacing, COEdt, 'oc')
    # plt.plot(spacing, COEXYdt, 'ok')
    # # plt.plot(spacingSAME2, COEXYZdtSAME2, 'om')
    # # plt.plot(spacing, COEXYZdtSAME3, 'y')
    # plt.plot(spacing, COEXYZdt, 'ob')
    # # plt.plot(spacing, COEXYZdt2, 'ow')
    # plt.plot(spacing, COEsame, 'or')
    # plt.legend()
    # plt.title('Wind Farm COE vs. Farm Size', fontsize=15)
    # plt.xlabel('Farm Size (square rotor diameters)', fontsize=15)
    # plt.ylabel('COE ($/MWhr)', fontsize=15)
    #
    # plt.show()
    # # plt.figure(6)
    # # plt.plot(spacing, diff,'k')
    # # plt.plot(spacing, diff, 'ok')
    # #
    # # plt.show()
    # # plt.axis([0,403,60,110])
    # #
    # # AEP /= 10.**6
    # # spacing_XYZdtSAME3_AEP /= 10.**6
    # # spacing_XYdt_AEP /= 10.**6
    # # spacing_dt_AEP /= 10.**6
    # #
    # # plt.figure(5)
    # # plt.plot(spacing, spacing_dt_AEP, 'c')
    # # plt.plot(spacing, spacing_XYdt_AEP, 'k')
    # # plt.plot(spacing, spacing_XYZdtSAME3_AEP, 'r')
    # # plt.plot(spacing, AEP, 'b')
    # # plt.plot(spacing, spacing_dt_AEP, 'oc')
    # # plt.plot(spacing, spacing_XYdt_AEP, 'ok')
    # # plt.plot(spacing, AEP, 'ob')
    # # plt.plot(spacing, spacing_XYZdtSAME3_AEP, 'or')
    # # plt.title('Wind Farm AEP vs. Farm Size')
    # # plt.xlabel('Farm Size (square rotor diameters)')
    # # plt.ylabel('AEP (GWhrs)')
    # # plt.axis([0,403,175,350])
    # # plt.legend()
    # #
    # # plt.figure(6)
    # # plt.plot(spacing, DIFF,'k')
    # # plt.plot(spacing, DIFF,'ok')
    # # plt.title('Difference in Height between \n Height Groups vs. Farm Size')
    # # plt.xlabel('Farm Size (square rotor diameters)')
    # # plt.ylabel('Height Difference (m)')
    # # plt.axis([0,403,-0.5,50])
    # # plt.show()
    # #
    # #
    # # # plt.figure(5)
    # # # plt.plot(spacing, spacing_XYZdt_AEP, 'r', label='XYZdt')
    # # # plt.plot(spacing, spacing_XYZdtSAME_AEP, 'b', label='XYZdtSAME')
    # # # plt.plot(spacing, spacing_XYdt_AEP, 'k', label='XYdt')
    # # # plt.plot(spacing, spacing_dt_AEP, 'c', label='dt')
    # # # plt.legend()
    # # #
    # # # plt.figure(6)
    # # # plt.plot(spacing, spacing_XYZdt_diff, 'b')
    # # plt.show()
