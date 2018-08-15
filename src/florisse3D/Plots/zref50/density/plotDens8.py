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
    gridfile = '0.08A.txt'
    optgrid = open(gridfile)
    optimizedgrid = np.loadtxt(optgrid)
    grid = optimizedgrid[:,1]

    gridfile = 'src/florisse3D/Plots/zref50/3/Ashear_grid_1.txt'
    gridfile = '0.08A_1.txt'
    optgrid = open(gridfile)
    optimizedgrid = np.loadtxt(optgrid)
    grid_1group = optimizedgrid[:,1]

    gridfile = 'src/florisse3D/Plots/zref50/3/Ashear_grid_2.txt'
    gridfile = '0.08A_2.txt'
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
    file = '0.08A2heights.txt'
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
    ideal_AEPb = np.array([  4.10465416e+08,   4.10465416e+08,   4.10465416e+08,
             4.10465416e+08,   4.10465416e+08,   4.10465416e+08,
             4.10465416e+08,   4.10465416e+08,   4.10465416e+08,
             4.10465416e+08])
    AEPb = np.array([  3.58207866e+08,   3.16672039e+08,   2.84192609e+08,
             2.57606823e+08,   2.35698687e+08,   2.17356706e+08,
             2.01695721e+08,   1.88228210e+08,   1.76518509e+08,
             1.66256916e+08])
    tower_costb = np.array([ 33621778.16560055,  33619017.96889926,  33622963.37465718,
            33623714.70958167,  33619834.25433736,  33622739.76675946,
            33623351.42002705,  33620687.01778759,  33620750.78567463,
            33620119.94539715])
    wake_lossb = np.array([ 12.73129192,  22.8504943 ,  30.76332417,  37.24030983,
            42.57769887,  47.0462803 ,  50.86170166,  54.14273576,
            56.99552224,  59.49551172])


    """1 group"""
    ideal_AEP1 = np.array([  3.80579113e+08,   3.80579113e+08,   3.80578227e+08,
             3.80582640e+08,   3.80576001e+08,   3.80580637e+08,
             3.80580841e+08,   3.80581951e+08,   3.80583046e+08,
             3.80583046e+08])
    AEP1 = np.array([  3.32126475e+08,   2.93614905e+08,   2.63499713e+08,
             2.38852486e+08,   2.18535497e+08,   2.01531604e+08,
             1.87010949e+08,   1.74524471e+08,   1.63667751e+08,
             1.54153215e+08])
    tower_cost1 = np.array([ 16146275.48298173,  16146275.48298173,  16146651.31035441,
            16148632.19578885,  16146390.66366807,  16147943.52396904,
            16148086.72960529,  16148317.3898887 ,  16148827.51330551,
            16148827.51330551])
    wake_loss1 = np.array([ 12.73129192,  22.8504943 ,  30.76332417,  37.24030983,
            42.57769887,  47.0462803 ,  50.86170166,  54.14273576,
            56.99552224,  59.49551172])

    """2 groups"""
    ideal_AEP2 = np.array([  3.74639363e+08,   3.77571735e+08,   3.80371187e+08,
             3.81621223e+08,   3.82138289e+08,   3.82139180e+08,
             3.82139180e+08,   3.82136519e+08,   3.83002841e+08,
             3.83000340e+08])
    AEP2 = np.array([  3.35936841e+08,   3.14176753e+08,   3.00147650e+08,
             2.85772314e+08,   2.72251264e+08,   2.59293945e+08,
             2.47728578e+08,   2.37329108e+08,   2.28717285e+08,
             2.20180756e+08])
    tower_cost2 = np.array([ 17059478.13083974,  19882028.76527821,  22923403.52468141,
            24247834.90495812,  24827352.58421196,  24829210.65989877,
            24828566.7220996 ,  24822946.54247962,  25220119.01638192,
            25214546.44079829])
    wake_loss2 = np.array([ 10.33060742,  16.79018205,  21.09085544,  25.11624175,
            28.75582678,  32.1467259 ,  35.1732063 ,  37.89415679,
            40.28313625,  42.51160291])

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
    ax[1][1].set_ylim(0.,500.)
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

    plt.suptitle('0.08 Shear Exponent: Small Rotor',fontsize=18,y=0.98)
    f.tight_layout(rect=[0, 0.0, 1, 0.95])
    plt.savefig('8_small_rotor.pdf', transparent=True)

    plt.show()
