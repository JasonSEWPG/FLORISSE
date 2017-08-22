import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
    gridfile = 'src/florisse3D/Plots/multiStartResults/smallRotor/Bdensity_grid.txt'
    optgrid = open(gridfile)
    optimizedgrid = np.loadtxt(optgrid)
    grid = optimizedgrid[:,1]

    gridfile_yaw = 'src/florisse3D/Plots/multiStartResults/smallRotor/Bdensity_gridYAW.txt'
    optgrid_yaw = open(gridfile_yaw)
    optimizedgrid_yaw = np.loadtxt(optgrid_yaw)
    grid_yaw = optimizedgrid_yaw[:,1]


    gridfile = 'src/florisse3D/Plots/multiStartResults/smallRotor/Bdensity_grid_1group.txt'
    optgrid = open(gridfile)
    optimizedgrid = np.loadtxt(optgrid)
    grid_1group = optimizedgrid[:,1]

    gridfile_yaw = 'src/florisse3D/Plots/multiStartResults/smallRotor/Bdensity_grid_1groupYAW.txt'
    optgrid_yaw = open(gridfile_yaw)
    optimizedgrid_yaw = np.loadtxt(optgrid_yaw)
    grid_1group_yaw = optimizedgrid_yaw[:,1]
    #
    gridfile = 'src/florisse3D/Plots/multiStartResults/smallRotor/Bdensity_grid_2group.txt'
    optgrid = open(gridfile)
    optimizedgrid = np.loadtxt(optgrid)
    grid_2group = optimizedgrid[:,1]

    gridfile_yaw = 'src/florisse3D/Plots/multiStartResults/smallRotor/Bdensity_grid_2groupYAW.txt'
    optgrid_yaw = open(gridfile_yaw)
    optimizedgrid_yaw = np.loadtxt(optgrid_yaw)
    grid_2group_yaw = optimizedgrid_yaw[:,1]


    density = np.array([0.024842,0.049684,0.0745261,0.0993681,0.12421,0.149052,0.173894,0.198736,0.223578,0.24842])

    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.plot(density, grid, 'ob', label='grid')
    ax.plot(density, grid_yaw, 'oc', label='grid_yaw')
    ax.plot(density, grid_1group, 'or', label='1 group')
    ax.plot(density, grid_1group_yaw, 'om', label='1 group, yaw')
    ax.plot(density, grid_2group, 'ok', label='2 group')
    ax.plot(density, grid_2group_yaw, 'ow', label='2 group, yaw')
    plt.legend(loc=2)
    plt.axis([0.07,0.255,60,160])
    # handles, labels = ax.get_legend_handles_labels()
    # lgd = ax.legend(handles, labels, loc='1', bbox_to_anchor=(1.0,-0.1))
    plt.xlabel('Turbine Density')
    plt.ylabel('COE')
    plt.title('Optimized COE vs Turbine Density: Grid')
    plt.show()
