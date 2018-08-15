import numpy as np
import matplotlib.pyplot as plt
import matplotlib

if __name__=="__main__":
    f, ax = plt.subplots(2, 2, figsize=(12,9.25),sharex=True)#,sharex=True,sharey=True)
    density = np.array([0.024842,0.049684,0.074526,0.099368,0.12421 ,0.149052,0.173894,0.198736,0.223578,0.24842])
    font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 18}

    matplotlib.rc('font', **font)

    gridfile = 'src/florisse3D/Plots/zref50/3/Ashear_grid.txt'
    gridfile = '0.25A.txt'
    optgrid = open(gridfile)
    optimizedgrid = np.loadtxt(optgrid)
    grid = optimizedgrid[:,1]

    gridfile = 'src/florisse3D/Plots/zref50/3/Ashear_grid_1.txt'
    gridfile = '0.25A_1.txt'
    optgrid = open(gridfile)
    optimizedgrid = np.loadtxt(optgrid)
    grid_1group = optimizedgrid[:,1]

    gridfile = 'src/florisse3D/Plots/zref50/3/Ashear_grid_2.txt'
    gridfile = '0.25A_2.txt'
    optgrid = open(gridfile)
    optimizedgrid = np.loadtxt(optgrid)
    grid_2group = optimizedgrid[:,1]

    ax[0][0].plot(density, grid, 'sb', label='baseline',markersize=8)
    ax[0][0].plot(density, grid_1group, 'or', label='1 group')
    ax[0][0].plot(density, grid_2group, 'ok', label='2 groups',markersize=4)

    ax[0][0].legend(loc=4)
    ax[0][0].set_xlim(0.02,0.255)
    ax[0][0].set_ylabel('COE ($/MWh)')
    ax[0][0].set_ylim(0.,130.)

    ax2 = ax[0][0].twiny()
    ax2.set_xlim(ax[0][0].get_xlim())
    ax2.set_xticks([0.0099401955,0.039760782,0.0621262219,0.1104466167,0.1590431281])
    ax2.set_xticklabels(['10','5','4','3','2.5'])
    ax2.set_xlabel('Grid Spacing (D)')

    file = 'src/florisse3D/Plots/zref50/3/A2heights.txt'
    file = '0.25A2heights.txt'
    opt = open(file)
    optimized = np.loadtxt(opt)
    group20_2 = optimized[:,0]
    group20_1 = optimized[:,1]

    ax[0][1].plot(density, group20_1, 'b',linewidth=3,label='hub height, group 1')
    ax[0][1].plot(density, group20_2, 'r',linewidth=3,label='hub height, group 2')
    ax[0][1].plot([0,1],[90.,90.],'--k')#, label='90 m')
    ax[0][1].text(0.18,82,'90 m')

    ax[0][1].set_ylabel('Optimized Hub Height (m)')
    ax[0][1].set_ylim(0.,120.)
    ax[0][1].legend(loc=4)
    ax[0][1].set_xlim(0.02,0.255)

    ax2 = ax[0][1].twiny()
    ax2.set_xlim(ax[0][1].get_xlim())
    ax2.set_xticks([0.0099401955,0.039760782,0.0621262219,0.1104466167,0.1590431281])
    ax2.set_xticklabels(['10','5','4','3','2.5'])
    ax2.set_xlabel('Grid Spacing (D)')

    """SMALL"""
    """baseline"""
    ideal_AEPb = np.array([  5.46787472e+08,   5.46787472e+08,   5.46787472e+08,
             5.46787472e+08,   5.46787472e+08,   5.46787472e+08,
             5.46787472e+08,   5.46787472e+08,   5.46787472e+08,
             5.46787472e+08])
    AEPb = np.array([  4.81620357e+08,   4.25717195e+08,   3.81884581e+08,
             3.46005733e+08,   3.16439608e+08,   2.91694854e+08,
             2.70586470e+08,   2.52483305e+08,   2.36798079e+08,
             2.23015768e+08])
    tower_costb = np.array([ 34044251.48583033,  34044937.02829164,  34043903.91558909,
            34037154.16248824,  34042268.40320115,  34042268.29379238,
            34045341.85062581,  34037043.20908744,  34044078.20149913,
            34044904.77379616])
    wake_lossb = np.array([ 11.91817988,  22.14210878,  30.15849848,  36.72025224,
            42.12749482,  46.6529742 ,  50.51341085,  53.82423368,
            56.69284844,  59.213446  ])

    """1 group"""
    ideal_AEP1 = np.array([  5.58363605e+08,   5.58890788e+08,   5.59417771e+08,
             5.58103426e+08,   5.57824431e+08,   5.57545380e+08,
             5.56491788e+08,   5.55873576e+08,   5.54637839e+08,
             5.53328891e+08])
    AEP1 = np.array([  4.93004299e+08,   4.36342770e+08,   3.91779684e+08,
             3.53983046e+08,   3.23548048e+08,   2.98044643e+08,
             2.75861822e+08,   2.57061960e+08,   2.40499894e+08,
             2.25904441e+08])
    tower_cost1 = np.array([ 37329594.86997971,  37486974.25359991,  37644764.85234769,
            37252566.01350669,  37169244.22076634,  37086038.58766221,
            36774120.28064054,  36592414.47397576,  36231326.10356545,
            35852551.29064404])
    wake_loss1 = np.array([ 11.70550983,  21.92700622,  29.96652876,  36.57393431,
            41.99822921,  46.54342879,  50.42841088,  53.75531938,
            56.63839057,  59.17356852])
    """2 groups"""
    ideal_AEP2 = np.array([  5.55331246e+08,   5.25638440e+08,   5.10473138e+08,
             4.98598099e+08,   4.92029733e+08,   4.87835279e+08,
             4.84781681e+08,   4.82447047e+08,   4.80724174e+08,
             4.79549636e+08])
    AEP2 = np.array([  4.90914779e+08,   4.31845149e+08,   3.99681294e+08,
             3.72641498e+08,   3.51516572e+08,   3.33748691e+08,
             3.18229103e+08,   3.04499233e+08,   2.92249416e+08,
             2.81231958e+08])
    tower_cost2 = np.array([ 36701004.52910343,  34990570.76757029,  34846126.35261185,
            33968516.21880025,  33514435.38904266,  33244880.65202157,
            33049852.14075491,  32904629.53458659,  32799182.26219534,
            32728267.69264033])
    wake_loss2 = np.array([ 11.59964742,  17.84368936,  21.70375597,  25.26215023,
            28.55785986,  31.58578207,  34.35620301,  36.88442376,
            39.20642402,  41.35498448])

    ax[1][0].plot(density,wake_lossb,'sb',label='baseline',markersize=8)
    ax[1][0].plot(density,wake_loss1,'or', label='1 group')
    ax[1][0].plot(density,wake_loss2,'ok', label='2 groups',markersize=4)
    ax[1][0].set_ylabel('% Wake Loss')
    ax[1][0].set_ylim(0.,70.)
    ax[1][0].set_xlabel('Turbine Density')
    ax[1][0].set_xlim(0.02,0.255)

    ax2 = ax[1][0].twiny()
    ax2.set_xlim(ax[1][0].get_xlim())
    ax2.set_xticks([0.0099401955,0.039760782,0.0621262219,0.1104466167,0.1590431281])
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
    ax2.set_xticks([0.0099401955,0.039760782,0.0621262219,0.1104466167,0.1590431281])
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

    plt.suptitle('0.25 Shear Exponent: Small Rotor',fontsize=18,y=0.98)
    f.tight_layout(rect=[0, 0.0, 1, 0.95])
    plt.savefig('25_small_rotor.pdf', transparent=True)

    plt.show()
