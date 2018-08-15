import numpy as np
import matplotlib.pyplot as plt
import matplotlib

if __name__=="__main__":
    f, ax = plt.subplots(2, 2, figsize=(12,9.25),sharex=True)#,sharex=True,sharey=True)
    shearExp = np.array([0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,\
                        0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.3])
    font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 18}

    matplotlib.rc('font', **font)

    gridfile = 'src/florisse3D/Plots/zref50/3/Ashear_grid.txt'
    gridfile = 'Ashear_grid.txt'
    optgrid = open(gridfile)
    optimizedgrid = np.loadtxt(optgrid)
    grid = optimizedgrid[:,1]

    #
    gridfile = 'src/florisse3D/Plots/zref50/3/Ashear_grid_1.txt'
    gridfile = 'Ashear_grid_1.txt'
    optgrid = open(gridfile)
    optimizedgrid = np.loadtxt(optgrid)
    grid_1group = optimizedgrid[:,1]
    #

    gridfile = 'src/florisse3D/Plots/zref50/3/Ashear_grid_2.txt'
    gridfile = 'Ashear_grid_2.txt'
    optgrid = open(gridfile)
    optimizedgrid = np.loadtxt(optgrid)
    grid_2group = optimizedgrid[:,1]

    # ax.plot(shearExp, grid, 'ob', label='grid')
    ax[0][0].plot(shearExp, grid, 'sb', label='baseline',markersize=8)
    ax[0][0].plot(shearExp, grid_1group, 'or', label='1 group')
    ax[0][0].plot(shearExp, grid_2group, 'ok', label='2 groups',markersize=4)
    # ax.plot(shearExp, grid_1group, '*r', label='1 optimized height')
    # ax.plot(shearExp, grid_2group, '^k', label='2 height groups')

    ax[0][0].legend(loc=4)
    # plt.axis([0.075,0.305,55,90])
    # handles, labels = ax.get_legend_handles_labels()
    # lgd = ax.legend(handles, labels, loc='1', bbox_to_anchor=(1.0,-0.1))
    ax[0][0].set_ylabel('COE ($/MWh)')
    ax[0][0].set_ylim(0.,100.)
    # ax[0][0].title('Small Farm: Small Rotor')
    # plt.title('Varied Wind Shear Exponent')
    plt.tight_layout()

    file = 'src/florisse3D/Plots/zref50/3/A2heights.txt'
    file = 'A2heights.txt'
    opt = open(file)
    optimized = np.loadtxt(opt)
    group20_2 = optimized[:,0]
    group20_1 = optimized[:,1]

    ax[0][1].plot(shearExp, group20_1, 'b',linewidth=3,label='hub height, group 1')
    ax[0][1].plot(shearExp, group20_2, 'r',linewidth=3,label='hub height, group 2')
    ax[0][1].plot([0,1],[90.,90.],'--k')#, label='90 m')
    ax[0][1].text(0.18,82,'90 m')
    # ax.plot(shearExp, group20_1-35., 'b',label='blade tip, group 1')
    # ax.plot(shearExp, group20_2+35., 'r',label='blade tip, group 1')
    ax[0][1].set_ylabel('Optimized Hub Height (m)')
    ax[0][1].set_ylim(0.,120.)
    # ax[0][1].title('Small Farm: Small Rotor: Hub Heights')
    ax[0][1].legend(loc=4)
    # plt.tight_layout()
    # ax[0][1].set_axis([0.075,0.305,0.,120.])

    """SMALL"""
    """baseline"""
    ideal_AEPb = np.array([  4.10465416e+08,   4.17767591e+08,   4.25199673e+08,
             4.32763970e+08,   4.40462837e+08,   4.48298666e+08,
             4.56273895e+08,   4.64391003e+08,   4.72652514e+08,
             4.80948195e+08,   4.88928743e+08,   4.97051265e+08,
             5.05318286e+08,   5.13678226e+08,   5.21896202e+08,
             5.30260376e+08,   5.38570749e+08,   5.46787472e+08,
             5.55060792e+08,   5.63261342e+08,   5.71475307e+08,
             5.79223386e+08,   5.86466197e+08])
    AEPb = np.array([  2.47328309e+08,   2.51728278e+08,   2.56206521e+08,
             2.60764432e+08,   2.65403429e+08,   2.70124953e+08,
             2.74930473e+08,   2.79821484e+08,   2.84799505e+08,
             2.89842411e+08,   2.94877917e+08,   3.00003005e+08,
             3.05219268e+08,   3.10511615e+08,   3.15808394e+08,
             3.21199404e+08,   3.26643798e+08,   3.32134365e+08,
             3.37710839e+08,   3.43302754e+08,   3.48966346e+08,
             3.54602247e+08,   3.60197381e+08])
    tower_costb = np.array([ 33623682.97330754,  33646352.1523455 ,  33668986.58628966,
            33692420.35947319,  33715837.93248764,  33738210.63789596,
            33762974.20537763,  33786984.82206847,  33811270.62580765,
            33836158.45814328,  33861108.45444167,  33887214.47070372,
            33906855.3459703 ,  33938727.26815102,  33964188.70571121,
            33991083.7832114 ,  34018397.29465535,  34045842.4089272 ,
            34067155.95865495,  34100699.76286625,  34128750.13412264,
            34156351.28385241,  34185597.63273675])
    wake_lossb = np.array([ 39.74442187,  39.74442187,  39.74442187,  39.74442187,
            39.74442187,  39.74442187,  39.74442187,  39.74442187,
            39.74442187,  39.7352119 ,  39.68897899,  39.64344805,
            39.59861015,  39.55133791,  39.488275  ,  39.42609737,
            39.34988143,  39.25713706,  39.1578645 ,  39.05089368,
            38.93588363,  38.77970824,  38.58173198])

    """1 group"""
    ideal_AEP1 = np.array([  3.80576626e+08,   3.86008242e+08,   3.92173440e+08,
             4.04116312e+08,   4.16402492e+08,   4.28884508e+08,
             4.35041409e+08,   4.41293494e+08,   4.47619159e+08,
             4.54243329e+08,   4.65862102e+08,   4.77988174e+08,
             4.90109059e+08,   5.03889445e+08,   5.15644136e+08,
             5.31555908e+08,   5.44851134e+08,   5.57670876e+08,
             5.70932761e+08,   5.81635575e+08,   5.91783957e+08,
             6.03740446e+08,   6.13692251e+08])
    AEP1 = np.array([  2.29318646e+08,   2.32591498e+08,   2.36306373e+08,
             2.43502620e+08,   2.50905729e+08,   2.58426840e+08,
             2.62136716e+08,   2.65903946e+08,   2.69715512e+08,
             2.73706944e+08,   2.80707903e+08,   2.88014537e+08,
             2.95622664e+08,   3.04317709e+08,   3.11778714e+08,
             3.22034421e+08,   3.30840468e+08,   3.39490643e+08,
             3.48583727e+08,   3.56420394e+08,   3.64333007e+08,
             3.73631577e+08,   3.81371102e+08])
    tower_cost1 = np.array([ 16146499.25736432,  16846755.85285272,  17694712.55348704,
            20180332.09994264,  22499349.8059318 ,  24617179.1323556 ,
            24632057.50928923,  24648212.98341035,  24659378.52771848,
            24733524.99454877,  26193714.31553246,  27639445.55458517,
            29131409.60355898,  30972794.38871579,  32106897.10679964,
            34367728.9668173 ,  35844180.65151103,  37123443.08465368,
            38490330.64327099,  39276288.65290488,  40072555.02028491,
            41322042.61672229,  41835351.10951237])
    wake_loss1 = np.array([ 39.74442187,  39.74442187,  39.74442187,  39.74442187,
            39.74442187,  39.74442187,  39.74442187,  39.74442187,
            39.74442187,  39.74442187,  39.74442187,  39.74442187,
            39.68226899,  39.6062546 ,  39.53606905,  39.41664165,
            39.2787411 ,  39.12347634,  38.94487211,  38.72101206,
            38.43479498,  38.11387348,  37.85629502])

    """2 groups"""
    ideal_AEP2 = np.array([  3.82078602e+08,   3.85716125e+08,   3.90481718e+08,
             3.94357589e+08,   3.98327502e+08,   4.02389932e+08,
             4.06557796e+08,   4.11090155e+08,   4.16676706e+08,
             4.22413040e+08,   4.35180324e+08,   4.51752095e+08,
             4.59085652e+08,   4.66203741e+08,   4.73522860e+08,
             4.80894843e+08,   4.88102423e+08,   4.95239644e+08,
             5.02575663e+08,   5.10371401e+08,   5.18214407e+08,
             5.26854300e+08,   5.36524563e+08])
    AEP2 = np.array([  2.80035989e+08,   2.82808763e+08,   2.86541971e+08,
             2.89416950e+08,   2.92362102e+08,   2.95372911e+08,
             2.98468928e+08,   3.01685883e+08,   3.05181258e+08,
             3.08794370e+08,   3.18574170e+08,   3.31783785e+08,
             3.36937963e+08,   3.41888747e+08,   3.46965994e+08,
             3.52156349e+08,   3.57405575e+08,   3.62711772e+08,
             3.68169541e+08,   3.73824590e+08,   3.79607297e+08,
             3.85612738e+08,   3.91891075e+08])
    tower_cost2 = np.array([ 24760398.07610117,  24848375.18453292,  25264440.25212284,
            25287781.53856856,  25311569.19737323,  25333010.70382905,
            25364001.1408332 ,  25416050.03331462,  25569324.09441949,
            25716611.45983376,  28446895.78417078,  32369469.63730412,
            32706137.2577231 ,  32900243.58479853,  33103300.03290809,
            33308795.10133406,  33526021.80572877,  33736181.75958945,
            33943174.85918789,  34166897.36634266,  34378913.99353539,
            34625255.07271547,  34916334.6944499 ])
    wake_loss2 =  np.array([ 26.70723034,  26.67955906,  26.61833886,  26.61052856,
            26.60258182,  26.59535272,  26.58634752,  26.61320637,
            26.75826278,  26.89752895,  26.79490489,  26.55622658,
            26.60673199,  26.66537888,  26.72666441,  26.77061232,
            26.77652112,  26.76035202,  26.74346015,  26.75440103,
            26.74705846,  26.80846723,  26.95747735])

    ax[1][0].plot(shearExp,wake_lossb,'sb',label='baseline',markersize=8)
    ax[1][0].plot(shearExp,wake_loss1,'or', label='1 group')
    ax[1][0].plot(shearExp,wake_loss2,'ok', label='2 groups',markersize=4)
    # ax[1][0].title('Small Farm: Small Rotor')
    # ax[1][0].set_xlabel('Wind Shear Exponent')
    ax[1][0].set_ylabel('% Wake Loss')
    ax[1][0].set_ylim(0.,60.)
    ax[1][0].set_xlim(0.075,0.305)
    ax[1][0].set_xlabel('Wind Shear Exponent')
    # ax[1][0].legend(loc=4)
    # plt.savefig('small3_wl.pdf', transparent=True)


    ax[1][1].plot(shearExp,ideal_AEPb/1000000.,'s',markerfacecolor='none',markeredgecolor='blue',markersize=8)
    ax[1][1].plot(shearExp,ideal_AEP1/1000000.,'o',markerfacecolor='none',markeredgecolor='red')
    ax[1][1].plot(shearExp,ideal_AEP2/1000000.,'o',markerfacecolor='none',markeredgecolor='black',label='ideal AEP',markersize=4)
    ax[1][1].plot(shearExp,AEPb/1000000.,'sb',markersize=8)
    ax[1][1].plot(shearExp,AEP1/1000000.,'or')
    ax[1][1].plot(shearExp,AEP2/1000000.,'ok',label='true AEP',markersize=4)
    # ax[1][1].title('Small Farm: Small Rotor')
    # ax[1][1].set_xlabel('Wind Shear Exponent')
    ax[1][1].set_xlim(0.075,0.305)
    ax[1][1].set_ylim(0.,700.)
    ax[1][1].set_ylabel('AEP (GWh)')
    ax[1][1].set_xlabel('Wind Shear Exponent')
    ax[1][1].legend(loc=4)
    # plt.savefig('small3_AEP.pdf', transparent=True)
    #
    # plt.figure(3)
    # plt.plot(shearExp,tower_costb,'ob',label='baseline')
    # plt.plot(shearExp,tower_cost1,'or', label='1 group')
    # plt.plot(shearExp,tower_cost2,'ok', label='2 groups')
    # plt.title('Small Farm: Small Rotor')
    # plt.xlabel('Wind Shear Exponent')
    # plt.ylabel('Tower Cost')
    # plt.xlim(0.075,0.305)
    # plt.legend(loc=4)
    # plt.savefig('small3_tc.pdf', transparent=True)

    plt.suptitle('Small Wind Farm: Small Rotor',fontsize=18,y=0.98)
    f.tight_layout(rect=[0, 0.0, 1, 0.95])
    plt.savefig('small_farm_small_rotor.pdf', transparent=True)

    plt.show()
