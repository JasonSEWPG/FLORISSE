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
    gridfile = '0.25B.txt'
    optgrid = open(gridfile)
    optimizedgrid = np.loadtxt(optgrid)
    grid = optimizedgrid[:,1]

    gridfile = 'src/florisse3D/Plots/zref50/3/Ashear_grid_1.txt'
    gridfile = '0.25B_1.txt'
    optgrid = open(gridfile)
    optimizedgrid = np.loadtxt(optgrid)
    grid_1group = optimizedgrid[:,1]

    gridfile = 'src/florisse3D/Plots/zref50/3/Ashear_grid_2.txt'
    gridfile = '0.25B_2.txt'
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
    file = '0.25B2heights.txt'
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
    ideal_AEPb = np.array([  5.49776909e+08,   5.49776909e+08,   5.49776909e+08,
             5.49776909e+08,   5.49776909e+08,   5.49776909e+08,
             5.49776909e+08,   5.49776909e+08,   5.49776909e+08,
             5.49776909e+08])
    AEPb = np.array([  5.10665330e+08,   4.73161866e+08,   4.42222177e+08,
             4.15884654e+08,   3.93307289e+08,   3.73687774e+08,
             3.56535221e+08,   3.41337571e+08,   3.27817031e+08,
             3.15785417e+08])
    tower_costb = np.array([ 16511815.23088944,  16516952.20634281,  16511380.71842201,
            16510017.19359521,  16507216.48756092,  16514565.28713956,
            16516773.13689355,  16516907.52232907,  16516816.66449152,
            16515854.18970476])
    wake_lossb = np.array([  7.11408173,  13.93566025,  19.56334111,  24.35392479,
            28.46056596,  32.02919799,  35.14910957,  37.9134399 ,
            40.37271737,  42.5611712 ])

    """1 group"""
    ideal_AEP1 = np.array([  5.90695923e+08,   5.92768057e+08,   5.91955705e+08,
             5.91913401e+08,   5.93344280e+08,   5.91963398e+08,
             5.92066395e+08,   5.91939392e+08,   5.92070682e+08,
             5.91906645e+08])
    AEP1 = np.array([  5.50723791e+08,   5.14648661e+08,   4.80581317e+08,
             4.52053597e+08,   4.28505784e+08,   4.05932485e+08,
             3.87363694e+08,   3.70786390e+08,   3.56008362e+08,
             3.42660710e+08])
    tower_cost1 = np.array([ 20304452.40988584,  20567263.11006819,  20461727.66336127,
            20456445.24365976,  20645497.7119213 ,  20458809.27157985,
            20469865.82406609,  20459018.89832142,  20471386.94725976,
            20453877.09614052])
    wake_loss1 = np.array([  6.76695575,  13.17874596,  18.81464908,  23.62842331,
            27.78125642,  31.42608368,  34.5742814 ,  37.36075096,
            39.8706316 ,  42.10899427])



    """2 groups"""
    ideal_AEP2 = np.array([  4.78418545e+08,   4.77544443e+08,   4.61677296e+08,
             4.61691234e+08,   4.66908396e+08,   4.67022058e+08,
             4.67022070e+08,   4.67022070e+08,   4.69662010e+08,
             4.69662008e+08])
    AEP2 = np.array([  4.41258109e+08,   4.08599492e+08,   3.74621551e+08,
             3.54897726e+08,   3.44499103e+08,   3.29567419e+08,
             3.16390777e+08,   3.04630632e+08,   2.98304205e+08,
             2.88832768e+08])
    tower_cost2 = np.array([ 17959156.51984414,  17928756.97442041,  16428854.65012237,
            16682397.9270203 ,  18002114.60921539,  18032889.4912218 ,
            18033087.59745981,  18034071.44617735,  18951335.87714211,
            18951865.22630659])
    wake_loss2 = np.array([  7.76734867,  14.43738939,  18.85640588,  23.13093678,
            26.21698263,  29.43215135,  32.25357058,  34.77168409,
            36.48534512,  38.50199454])


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
    ax[1][1].set_ylim(0.,650.)
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

    plt.suptitle('0.25 Shear Exponent: Big Rotor',fontsize=18,y=0.98)
    f.tight_layout(rect=[0, 0.0, 1, 0.95])
    plt.savefig('25_big_rotor.pdf', transparent=True)

    plt.show()
