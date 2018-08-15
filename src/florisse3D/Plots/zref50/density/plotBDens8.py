import numpy as np
import matplotlib.pyplot as plt
import matplotlib

if __name__=="__main__":
    f, ax = plt.subplots(2, 2, figsize=(12,9.25),sharex=True)#,sharex=True,sharey=True)
    density = np.array([0.025,0.05,0.075,0.1,0.125 ,0.15,0.175,0.2,0.225,0.25])
    font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 18}

    matplotlib.rc('font', **font)

    gridfile = 'src/florisse3D/Plots/zref50/3/Ashear_grid.txt'
    gridfile = '0.08B.txt'
    optgrid = open(gridfile)
    optimizedgrid = np.loadtxt(optgrid)
    grid = optimizedgrid[:,1]

    gridfile = 'src/florisse3D/Plots/zref50/3/Ashear_grid_1.txt'
    gridfile = '0.08B_1.txt'
    optgrid = open(gridfile)
    optimizedgrid = np.loadtxt(optgrid)
    grid_1group = optimizedgrid[:,1]

    gridfile = 'src/florisse3D/Plots/zref50/3/Ashear_grid_2.txt'
    gridfile = '0.08B_2.txt'
    optgrid = open(gridfile)
    optimizedgrid = np.loadtxt(optgrid)
    grid_2group = optimizedgrid[:,1]

    ax[0][0].plot(density, grid, 'sb', label='baseline',markersize=8)
    ax[0][0].plot(density, grid_1group, 'or', label='1 group')
    ax[0][0].plot(density, grid_2group, 'ok', label='2 groups',markersize=4)

    ax[0][0].legend(loc=1)
    ax[0][0].set_xlim(0.02,0.255)
    ax[0][0].set_ylabel('COE ($/MWh)')
    ax[0][0].set_ylim(0.,130.)

    ax2 = ax[0][0].twiny()
    ax2.set_xlim(ax[0][0].get_xlim())
    ax2.set_xticks([0.0122718463,0.0490873852,0.0766990394,0.136353478,0.1963495408])
    ax2.set_xticklabels(['10','5','4','3','2.5'])
    ax2.set_xlabel('Grid Spacing (D)')

    file = 'src/florisse3D/Plots/zref50/3/A2heights.txt'
    file = '0.08B2heights.txt'
    opt = open(file)
    optimized = np.loadtxt(opt)
    group20_2 = optimized[:,0]
    group20_1 = optimized[:,1]

    ax[0][1].plot(density, group20_1, 'b',linewidth=3,label='hub height, group 1')
    ax[0][1].plot(density, group20_2, 'r',linewidth=3,label='hub height, group 2')
    ax[0][1].plot([0,1],[90.,90.],'--k')#, label='90 m')
    ax[0][1].text(0.08,82,'90 m')

    ax[0][1].set_ylabel('Optimized Hub Height (m)')
    ax[0][1].set_ylim(0.,120.)
    ax[0][1].legend(loc=4)
    ax[0][1].set_xlim(0.02,0.255)

    ax2 = ax[0][1].twiny()
    ax2.set_xlim(ax[0][1].get_xlim())
    ax2.set_xticks([0.0122718463,0.0490873852,0.0766990394,0.136353478,0.1963495408])
    ax2.set_xticklabels(['10','5','4','3','2.5'])
    ax2.set_xlabel('Grid Spacing (D)')

    """SMALL"""
    """baseline"""
    ideal_AEPb = np.array([  4.13075682e+08,   4.13075682e+08,   4.13075682e+08,
             4.13075682e+08,   4.13075682e+08,   4.13075682e+08,
             4.13075682e+08,   4.13075682e+08,   4.13075682e+08,
             4.13075682e+08])
    AEPb = np.array([  3.80991344e+08,   3.52981114e+08,   3.29922960e+08,
             3.10407179e+08,   2.93677628e+08,   2.79139808e+08,
             2.66428644e+08,   2.55145204e+08,   2.45079975e+08,
             2.36063944e+08])
    tower_costb = np.array([ 16492236.58706436,  16492363.89379455,  16490598.38689725,
            16490739.96337217,  16492199.34043626,  16491737.56327738,
            16486073.55162448,  16491681.45067947,  16485880.46631331,
            16489191.3384954 ])
    wake_lossb = np.array([  7.76718151,  14.54807708,  20.13014215,  24.85464721,
            28.90464369,  32.42405202,  35.50125177,  38.23281903,
            40.66947403,  42.85213235])


    """1 group"""
    ideal_AEP1 = np.array([  3.93091892e+08,   3.93091892e+08,   3.93091892e+08,
             3.93091892e+08,   3.93091892e+08,   3.93091892e+08,
             3.93091892e+08,   3.93091892e+08,   3.93091892e+08,
             3.93091892e+08])
    AEP1 = np.array([  3.62559731e+08,   3.35904581e+08,   3.13961935e+08,
             2.95390289e+08,   2.79470081e+08,   2.65635572e+08,
             2.53539350e+08,   2.42801780e+08,   2.33223487e+08,
             2.24643634e+08])
    tower_cost1 = np.array([ 11930698.86867209,  11932747.27659467,  11930094.52985454,
            11932203.28129075,  11933779.38995646,  11933170.12322754,
            11933425.82216983,  11932597.29428656,  11933977.89150421,
            11933053.95662389])
    wake_loss1 = np.array([  7.76718151,  14.54807708,  20.13014215,  24.85464721,
            28.90464369,  32.42405202,  35.50125177,  38.23281903,
            40.66947403,  42.85213235])


    """2 groups"""
    ideal_AEP2 = np.array([  3.93091892e+08,   3.93091892e+08,   3.93091892e+08,
             3.93091892e+08,   3.93091892e+08,   3.93091892e+08,
             3.93091892e+08,   4.10386764e+08,   4.12927389e+08,
             4.12927389e+08])
    AEP2 = np.array([  3.62559731e+08,   3.35904581e+08,   3.13961935e+08,
             2.95390289e+08,   2.79470081e+08,   2.65635572e+08,
             2.53539350e+08,   2.63934135e+08,   2.59843899e+08,
             2.51457719e+08])
    tower_cost2 = np.array([ 11929691.82889178,  11932146.81105111,  11932944.92212378,
            11932629.69374506,  11932338.04184561,  11933114.26973423,
            11934668.62117205,  16644940.6809602 ,  17962879.79808424,
            17961442.82661904])
    wake_loss2 = np.array([  7.76718151,  14.54807708,  20.13014215,  24.85464721,
            28.90464369,  32.42405202,  35.50125177,  35.68648938,
            37.07273806,  39.10364734])


    ax[1][0].plot(density,wake_lossb,'sb',label='baseline',markersize=8)
    ax[1][0].plot(density,wake_loss1,'or', label='1 group')
    ax[1][0].plot(density,wake_loss2,'ok', label='2 groups',markersize=4)
    ax[1][0].set_ylabel('% Wake Loss')
    ax[1][0].set_ylim(0.,70.)
    ax[1][0].set_xlabel('Turbine Density')
    ax[1][0].set_xlim(0.02,0.255)

    ax2 = ax[1][0].twiny()
    ax2.set_xlim(ax[1][0].get_xlim())
    ax2.set_xticks([0.0122718463,0.0490873852,0.0766990394,0.136353478,0.1963495408])
    ax2.set_xticklabels(['','','','',''])

    ax[1][1].plot(density,ideal_AEPb/1000000.,'s',markerfacecolor='none',markeredgecolor='blue',markersize=8)
    ax[1][1].plot(density,ideal_AEP1/1000000.,'o',markerfacecolor='none',markeredgecolor='red')
    ax[1][1].plot(density,ideal_AEP2/1000000.,'o',markerfacecolor='none',markeredgecolor='black',label='ideal AEP',markersize=4)
    ax[1][1].plot(density,AEPb/1000000.,'sb',markersize=8)
    ax[1][1].plot(density,AEP1/1000000.,'or')
    ax[1][1].plot(density,AEP2/1000000.,'ok',label='true AEP',markersize=4)

    ax[1][1].set_xlim(0.02,0.255)
    ax[1][1].set_ylim(0.,500.)
    ax[1][1].set_ylabel('AEP (GWh)')
    ax[1][1].set_xlabel('Turbine Density')
    ax[1][1].legend(loc=3)

    ax2 = ax[1][1].twiny()
    ax2.set_xlim(ax[1][1].get_xlim())
    ax2.set_xticks([0.0122718463,0.0490873852,0.0766990394,0.136353478,0.1963495408])
    ax2.set_xticklabels(['','','','',''])

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

    plt.suptitle('0.08 Shear Exponent: Big Rotor',fontsize=18,y=0.98)
    f.tight_layout(rect=[0, 0.0, 1, 0.95])
    plt.savefig('8_big_rotor.pdf', transparent=True)

    plt.show()
