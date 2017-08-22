import numpy as np
import matplotlib.pyplot as plt
import matplotlib

if __name__=="__main__":
    gridfile = 'src/florisse3D/Plots/zref50/70m/Bdensity_grid.txt'
    optgrid = open(gridfile)
    optimizedgrid = np.loadtxt(optgrid)
    grid = optimizedgrid[:,1]

    gridfile = 'src/florisse3D/Plots/zref50/70m/Bdensity_grid_1.txt'
    optgrid = open(gridfile)
    optimizedgrid = np.loadtxt(optgrid)
    grid_1group = optimizedgrid[:,1]

    gridfile = 'src/florisse3D/Plots/zref50/70m/Bdensity_grid_2.txt'
    optgrid = open(gridfile)
    optimizedgrid = np.loadtxt(optgrid)
    grid_2group = optimizedgrid[:,1]



    density = np.array([0.024842,0.049684,0.0745261,0.0993681,0.12421,0.149052,0.173894,0.198736,0.223578,0.24842])

    fig = plt.figure(1)

    font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 18}

    matplotlib.rc('font', **font)
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()
    ax1.plot(density, grid, 'ob', label='grid')
    ax1.plot(density, grid_1group, 'or', label='1 group, grid')
    ax1.plot(density, grid_2group, 'ok', label='2 groups, grid')
    ax1.legend(loc=2)
    ax1.set_xlim([0.,0.275])
    # ax1.set_ylim([46.,129.])

    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks([0.0099401955,0.039760782,0.0621262219,0.1104466167,0.2485048876])
    ax2.set_xticklabels(['10','5','4','3','2'])
    ax2.set_xlabel('Grid Spacing (D)')

    # handles, labels = ax.get_legend_handles_labels()
    # lgd = ax.legend(handles, labels, loc='1', bbox_to_anchor=(1.0,-0.1))
    ax1.set_xlabel('Turbine Density')
    ax1.set_ylabel('COE ($/MWh)')

    plt.title('Varied Turbine Density: Grid', y=1.15)
    plt.tight_layout()
    plt.show()
