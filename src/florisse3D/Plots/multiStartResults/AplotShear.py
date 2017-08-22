import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
    gridfile = 'src/florisse3D/Plots/multiStartResults/AshearExp_grid.txt'
    optgrid = open(gridfile)
    optimizedgrid = np.loadtxt(optgrid)
    grid = optimizedgrid[:,1]

    gridfile_yaw = 'src/florisse3D/Plots/multiStartResults/AshearExp_grid_YAW.txt'
    optgrid_yaw = open(gridfile_yaw)
    optimizedgrid_yaw = np.loadtxt(optgrid_yaw)
    grid_yaw = optimizedgrid_yaw[:,1]


    layoutfile = 'src/florisse3D/Plots/multiStartResults/AshearExp_layout.txt'
    optlayout = open(layoutfile)
    optimizedlayout = np.loadtxt(optlayout)
    layout = optimizedlayout[:,1]
    #
    layout_yaw_file = 'src/florisse3D/Plots/multiStartResults/AshearExp_layout_YAW.txt'
    optlayout_yaw = open(layout_yaw_file)
    optimizedlayout_yaw = np.loadtxt(optlayout_yaw)
    layout_yaw = optimizedlayout_yaw[:,1]

    group1file = 'src/florisse3D/Plots/multiStartResults/AshearExp_1group.txt'
    opt1group = open(group1file)
    optimized1group= np.loadtxt(opt1group)
    group1 = optimized1group[:,1]
    #
    #
    group2file = 'src/florisse3D/Plots/multiStartResults/AshearExp_2group.txt'
    opt2group = open(group2file)
    optimized2group= np.loadtxt(opt2group)
    group2 = optimized2group[:]
    #
    group1_yaw_file = 'src/florisse3D/Plots/multiStartResults/AshearExp_1group_YAW.txt'
    opt1group_yaw = open(group1_yaw_file)
    optimized1group_yaw= np.loadtxt(opt1group_yaw)
    group1_yaw = optimized1group_yaw[:,1]
    #
    group2_yaw_file = 'src/florisse3D/Plots/multiStartResults/AshearExp_2groups_YAW.txt'
    opt2group_yaw = open(group2_yaw_file)
    optimized2group_yaw= np.loadtxt(opt2group_yaw)
    group2_yaw = optimized2group_yaw[:]


    shearExp = np.array([0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,\
                        0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.3])

    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.plot(shearExp, grid, 'ob', label='grid')
    ax.plot(shearExp, grid_yaw, 'oc', label='grid_yaw')
    ax.plot(shearExp, layout, 'or', label='layout')
    ax.plot(shearExp, layout_yaw, 'om', label='layout, yaw')
    ax.plot(shearExp, group1, 'og', label='1 group')
    # ax.plot(shearExp, group2, 'ok', label='2 groups')
    ax.plot(shearExp, group1_yaw, 'oy', label='1 group, yaw')
    # ax.plot(shearExp, group2_yaw, 'ow', label='2 groups, yaw')
    # plt.legend()
    plt.axis([0.075,0.305,83,102])
    # handles, labels = ax.get_legend_handles_labels()
    # lgd = ax.legend(handles, labels, loc='1', bbox_to_anchor=(1.0,-0.1))
    plt.xlabel('Wind Shear Exponent')
    plt.ylabel('COE')
    plt.title('Optimized COE vs Wind Shear')
    plt.show()
