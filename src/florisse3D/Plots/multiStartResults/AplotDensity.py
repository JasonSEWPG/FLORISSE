import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":

    gridfile = 'src/florisse3D/Plots/multiStartResults/Adensity_grid.txt'
    optgrid = open(gridfile)
    optimizedgrid = np.loadtxt(optgrid)
    grid = optimizedgrid[:]


    # layoutfile = 'src/florisse3D/Plots/multiStartResults/Adensity_layout.txt'
    # optlayout = open(layoutfile)
    # optimizedlayout = np.loadtxt(optlayout)
    # layout = optimizedlayout[:]
    #
    group1file = 'src/florisse3D/Plots/multiStartResults/Adensity_1group.txt'
    opt1group = open(group1file)
    optimized1group= np.loadtxt(opt1group)
    group1 = optimized1group[:]
    #
    #
    group2file = 'src/florisse3D/Plots/multiStartResults/Adensity_2group.txt'
    opt2group = open(group2file)
    optimized2group= np.loadtxt(opt2group)
    group2 = optimized2group[:]
    # #
    # #
    group1_yaw_file = 'src/florisse3D/Plots/multiStartResults/Adensity_1groupyaw.txt'
    opt1group_yaw = open(group1_yaw_file)
    optimized1group_yaw= np.loadtxt(opt1group_yaw)
    group1_yaw = optimized1group_yaw#[:,0]
    #
    group2_yaw_file = 'src/florisse3D/Plots/multiStartResults/Adensity_2groupyaw.txt'
    opt2group_yaw = open(group2_yaw_file)
    optimized2group_yaw= np.loadtxt(opt2group_yaw)
    group2_yaw = optimized2group_yaw[:]


    density = np.array([0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25,0.275,0.3])

    plt.plot(density, grid, 'ob', label='grid')
    # plt.plot(density, layout, 'or', label='layout')
    plt.plot(density, group1, 'og', label='1 group')
    plt.plot(density, group2, 'ok', label='2 groups')
    plt.plot(density, group1_yaw, 'oy', label='1 group, yaw')
    plt.plot(density, group2_yaw, 'ow', label='2 groups, yaw')
    # # plt.plot(density, group5, 'oy', label='5 groups')

    # plt.axis([0.02,0.305,50,95])
    plt.legend(loc=2)
    plt.xlabel('Turbine Density')
    plt.ylabel('COE')
    plt.title('Optimized COE vs Turbine Density')
    plt.show()

    # layoutfile = 'layout.txt'
    # o = open(layoutfile)
    # layout= np.loadtxt(o)
    # x = layout[:,0]
    # y = layout[:,1]
    #
    # plt.plot(x,y,'ob')
    # plt.show()
