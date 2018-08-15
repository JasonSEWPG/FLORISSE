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
    gridfile = 'Bshear_grid.txt'
    optgrid = open(gridfile)
    optimizedgrid = np.loadtxt(optgrid)
    grid = optimizedgrid[:,1]

    #
    gridfile = 'src/florisse3D/Plots/zref50/3/Ashear_grid_1.txt'
    gridfile = 'Bshear_grid_1.txt'
    optgrid = open(gridfile)
    optimizedgrid = np.loadtxt(optgrid)
    grid_1group = optimizedgrid[:,1]
    #

    gridfile = 'src/florisse3D/Plots/zref50/3/Ashear_grid_2.txt'
    gridfile = 'Bshear_grid_2.txt'
    optgrid = open(gridfile)
    optimizedgrid = np.loadtxt(optgrid)
    grid_2group = optimizedgrid[:,1]

    # ax.plot(shearExp, grid, 'ob', label='grid')
    ax[0][0].plot(shearExp, grid, 'sb', label='baseline',markersize=8)
    ax[0][0].plot(shearExp, grid_1group, 'or', label='1 group')
    ax[0][0].plot(shearExp, grid_2group, 'ok', label='2 groups',markersize=4)
    # ax.plot(shearExp, grid_1group, '*r', label='1 optimized height')
    # ax.plot(shearExp, grid_2group, '^k', label='2 height groups')

    ax[0][0].legend(loc=1)
    # plt.axis([0.075,0.305,55,90])
    # handles, labels = ax.get_legend_handles_labels()
    # lgd = ax.legend(handles, labels, loc='1', bbox_to_anchor=(1.0,-0.1))
    ax[0][0].set_ylabel('COE ($/MWh)')
    ax[0][0].set_ylim(0.,100.)
    # ax[0][0].title('Small Farm: Small Rotor')
    # plt.title('Varied Wind Shear Exponent')
    plt.tight_layout()

    file = 'src/florisse3D/Plots/zref50/3/A2heights.txt'
    file = 'B2heights.txt'
    opt = open(file)
    optimized = np.loadtxt(opt)
    group20_2 = optimized[:,0]
    group20_1 = optimized[:,1]

    ax[0][1].plot(shearExp, group20_1, 'b',linewidth=3,label='hub height, group 1')
    ax[0][1].plot(shearExp, group20_2, 'r',linewidth=3,label='hub height, group 2')
    ax[0][1].plot([0,1],[90.,90.],'--k')#, label='90 m')
    ax[0][1].text(0.23,82,'90 m')
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
    ideal_AEPb = np.array([  4.13075682e+08,   4.20424294e+08,   4.27903638e+08,
             4.35516040e+08,   4.43263866e+08,   4.51149525e+08,
             4.59175471e+08,   4.67344198e+08,   4.75658246e+08,
             4.83800955e+08,   4.91832254e+08,   5.00006429e+08,
             5.08326023e+08,   5.16615858e+08,   5.24886095e+08,
             5.33303459e+08,   5.41507934e+08,   5.49776909e+08,
             5.57992195e+08,   5.66244895e+08,   5.74368896e+08,
             5.81891933e+08,   5.89056286e+08])
    AEPb = np.array([  3.02617331e+08,   3.08000891e+08,   3.13480223e+08,
             3.19057033e+08,   3.24733055e+08,   3.30510052e+08,
             3.36389823e+08,   3.42374194e+08,   3.48465028e+08,
             3.54549288e+08,   3.60649531e+08,   3.66858297e+08,
             3.73177517e+08,   3.79516718e+08,   3.85881815e+08,
             3.92360147e+08,   3.98823180e+08,   4.05371865e+08,
             4.11932639e+08,   4.18553587e+08,   4.25182113e+08,
             4.31607559e+08,   4.37970007e+08])
    tower_costb = np.array([ 16492859.33752908,  16490308.83580766,  16494292.34217262,
            16563078.61774057,  16496893.32803011,  16495077.98298375,
            16492718.8893107 ,  16501024.28209037,  16501712.79659994,
            16499834.30227974,  16505608.78475723,  16506640.37648823,
            16507491.11055629,  16509335.82618657,  16511679.58521711,
            16512600.23719032,  16565298.90901314,  16515479.55554846,
            16517040.92356223,  16518800.18715913,  16522365.26213906,
            16521856.30499885,  16525955.68387577])
    wake_lossb = np.array([ 26.74046322,  26.74046322,  26.74046322,  26.74046322,
            26.74046322,  26.74046322,  26.74046322,  26.74046322,
            26.74046322,  26.71587675,  26.67224888,  26.62928403,
            26.58697373,  26.53792715,  26.48275144,  26.42835138,
            26.34952233,  26.26611649,  26.17591372,  26.08258538,
            25.97403581,  25.82685291,  25.64886964])

    """1 group"""
    ideal_AEP1 = np.array([  3.93091892e+08,   3.97612765e+08,   4.02185631e+08,
             4.13273138e+08,   4.29062587e+08,   4.47555165e+08,
             4.64750622e+08,   4.79363732e+08,   4.93325610e+08,
             5.05203032e+08,   5.21307420e+08,   5.32125254e+08,
             5.42365210e+08,   5.54812018e+08,   5.64669867e+08,
             5.74404282e+08,   5.83332604e+08,   5.93712602e+08,
             6.07991228e+08,   6.21406045e+08,   6.32213803e+08,
             6.44596787e+08,   6.58800398e+08])
    AEP1 = np.array([  2.87977299e+08,   2.91289270e+08,   2.94639330e+08,
             3.02761987e+08,   3.14329263e+08,   3.27876841e+08,
             3.40474153e+08,   3.51179650e+08,   3.61783823e+08,
             3.70805422e+08,   3.83127527e+08,   3.91453354e+08,
             3.99502107e+08,   4.09381258e+08,   4.17289979e+08,
             4.25210999e+08,   4.32886977e+08,   4.42105142e+08,
             4.54785563e+08,   4.66698861e+08,   4.76252792e+08,
             4.87015477e+08,   4.99231102e+08])
    tower_cost1 = np.array([ 11934468.25109019,  11931396.67772063,  11934116.88809882,
            12857900.97971862,  14307956.85420625,  15967114.30597654,
            17341935.94767909,  18090877.15011676,  18828012.36772105,
            19159437.05945804,  20027207.90054901,  20130533.60216072,
            20181559.77200128,  20510097.78913343,  20496887.01028877,
            20479498.40713717,  20469410.46119757,  20704995.14833817,
            21412020.99131322,  21929255.97242977,  22039540.59727028,
            22309754.1202873 ,  22744234.77917372])
    wake_loss1 = np.array([ 26.74046322,  26.74046322,  26.74046322,  26.74046322,
            26.74046322,  26.74046322,  26.74046322,  26.74046322,
            26.66429325,  26.60269271,  26.50641206,  26.43586234,
            26.34075717,  26.21261889,  26.10018646,  25.97356735,
            25.79071128,  25.53549636,  25.19866374,  24.89631143,
            24.6690298 ,  24.44649326,  24.22118996])

    """2 groups"""
    ideal_AEP2 = np.array([  3.93091892e+08,   3.97612765e+08,   4.02185631e+08,
             4.29978230e+08,   4.39295980e+08,   4.48475123e+08,
             4.56359366e+08,   4.65316692e+08,   4.75353677e+08,
             4.86831737e+08,   4.95223932e+08,   5.03527371e+08,
             5.11866030e+08,   5.54078394e+08,   5.64639736e+08,
             5.74424299e+08,   5.83338236e+08,   5.93861644e+08,
             6.06559400e+08,   6.18452876e+08,   6.34024044e+08,
             6.44552150e+08,   6.57777098e+08])
    AEP2 = np.array([  2.87977299e+08,   2.91289270e+08,   2.94639330e+08,
             3.21494228e+08,   3.28932713e+08,   3.36648542e+08,
             3.42908254e+08,   3.50476686e+08,   3.59346524e+08,
             3.70110110e+08,   3.76955311e+08,   3.83673523e+08,
             3.90460153e+08,   4.08801356e+08,   4.17266070e+08,
             4.25227343e+08,   4.32891960e+08,   4.42260825e+08,
             4.53673061e+08,   4.64308457e+08,   4.77879354e+08,
             4.87030425e+08,   4.98349115e+08])
    tower_cost2 = np.array([ 11934077.77457232,  11934486.71054761,  11986114.49956485,
            16101104.52598352,  16443773.56029143,  16835688.80903514,
            16926359.47712496,  17255281.02550614,  17731204.69369097,
            18526967.51678783,  18578826.6771846 ,  18591745.26983801,
            18629125.154367  ,  20408584.22502878,  20491852.51400457,
            20481050.41416576,  20472370.66552951,  20727671.9414484 ,
            21235998.01606207,  21576485.51297639,  22276361.36883869,
            22308129.13030356,  22623583.56904258])
    wake_loss2 = np.array([ 26.74046322,  26.74046322,  26.74046322,  25.23011492,
            25.1227581 ,  24.93484583,  24.86003799,  24.67996712,
            24.40438736,  23.9757638 ,  23.88184687,  23.80284658,
            23.7182914 ,  26.21958173,  26.10047719,  25.97330168,
            25.79057338,  25.52796944,  25.2055016 ,  24.92419796,
            24.62756607,  24.4389418 ,  24.23738735])

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

    plt.suptitle('Small Wind Farm: Big Rotor',fontsize=18,y=0.98)
    f.tight_layout(rect=[0, 0.0, 1, 0.95])
    plt.savefig('small_farm_big_rotor.pdf', transparent=True)

    plt.show()
