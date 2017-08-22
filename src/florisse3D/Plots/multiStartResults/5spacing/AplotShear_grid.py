import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
    gridfile = 'src/florisse3D/Plots/multiStartResults/5spacing/Ashear_grid.txt'
    optgrid = open(gridfile)
    optimizedgrid = np.loadtxt(optgrid)
    grid = optimizedgrid[:,1]

    gridfile_yaw = 'src/florisse3D/Plots/multiStartResults/5spacing/Ashear_grid_YAW.txt'
    optgrid_yaw = open(gridfile_yaw)
    optimizedgrid_yaw = np.loadtxt(optgrid_yaw)
    grid_yaw = optimizedgrid_yaw[:,1]

    gridfile = 'src/florisse3D/Plots/multiStartResults/5spacing/Ashear_grid_1.txt'
    optgrid = open(gridfile)
    optimizedgrid = np.loadtxt(optgrid)
    grid_1group = optimizedgrid[:,1]

    gridfile_yaw = 'src/florisse3D/Plots/multiStartResults/5spacing/Ashear_grid_1_YAW.txt'
    optgrid_yaw = open(gridfile_yaw)
    optimizedgrid_yaw = np.loadtxt(optgrid_yaw)
    grid_1group_yaw = optimizedgrid_yaw[:,1]

    gridfile = 'src/florisse3D/Plots/multiStartResults/5spacing/Ashear_grid_2.txt'
    optgrid = open(gridfile)
    optimizedgrid = np.loadtxt(optgrid)
    grid_2group = optimizedgrid[:,1]
    #
    gridfile_yaw = 'src/florisse3D/Plots/multiStartResults/5spacing/Ashear_grid_2_YAW.txt'
    optgrid_yaw = open(gridfile_yaw)
    optimizedgrid_yaw = np.loadtxt(optgrid_yaw)
    grid_2group_yaw = optimizedgrid_yaw[:,1]

    shearExp = np.array([0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,\
                        0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.3])

    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.plot(shearExp, grid, 'ob', label='grid')
    ax.plot(shearExp, grid_yaw, 'oc', label='grid_yaw')
    ax.plot(shearExp, grid_1group, 'or', label='1 group')
    ax.plot(shearExp, grid_1group_yaw, 'om', label='1 group, yaw')
    ax.plot(shearExp, grid_2group, 'ok', label='2 group')
    # ax.plot(shearExp, grid_2group_yaw, 'ow', label='2 group, yaw')
    plt.legend(loc=4)
    plt.axis([0.075,0.305,60,80])
    # handles, labels = ax.get_legend_handles_labels()
    # lgd = ax.legend(handles, labels, loc='1', bbox_to_anchor=(1.0,-0.1))
    plt.xlabel('Wind Shear Exponent')
    plt.ylabel('COE')
    plt.title('Optimized COE vs Wind Shear: Grid')
    plt.show()
