import numpy as np
import matplotlib.pyplot as plt
import matplotlib

if __name__=="__main__":
    gridfile = 'src/florisse3D/Plots/zref50/126.4m/Bdensity_grid.txt'
    optgrid = open(gridfile)
    optimizedgrid = np.loadtxt(optgrid)
    grid = optimizedgrid[:,1]

    gridfile = 'src/florisse3D/Plots/zref50/126.4m/Bdensity_grid_1.txt'
    optgrid = open(gridfile)
    optimizedgrid = np.loadtxt(optgrid)
    grid_1group = optimizedgrid[:,1]
    #
    gridfile = 'src/florisse3D/Plots/zref50/126.4m/Bdensity_grid_2.txt'
    optgrid = open(gridfile)
    optimizedgrid = np.loadtxt(optgrid)
    grid_2group = optimizedgrid[:,1]


    density = np.array([0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25,0.275,0.3])

    font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 18}

    matplotlib.rc('font', **font)

    fig = plt.figure(1)
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()
    ax1.plot(density[0:10], grid[0:10], 'ob', label='grid')
    ax1.plot(density[0:10], grid_1group[0:10], 'or', label='1 group, grid')
    ax1.plot(density[0:10], grid_2group[0:10], 'ok', label='2 groups, grid')
    ax1.legend(loc=2)
    ax1.set_xlim([0.,0.275])
    # ax1.set_ylim([46,129])
    # handles, labels = ax.get_legend_handles_labels()
    # lgd = ax.legend(handles, labels, loc='1', bbox_to_anchor=(1.0,-0.1))

    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks([0.0122718463,0.0490873852,0.0766990394,0.136353478,0.1963495408])
    ax2.set_xticklabels(['10','5','4','3','2.5'])
    ax2.set_xlabel('Grid Spacing (D)')

    ax1.set_xlabel('Turbine Density')
    ax1.set_ylabel('COE ($/MWh)')

    plt.title('Varied Turbine Density: Grid', y=1.15)
    plt.tight_layout()
    plt.show()
