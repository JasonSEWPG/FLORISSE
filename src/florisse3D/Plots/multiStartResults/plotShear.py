import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
    gridfile = 'src/florisse3D/Plots/multiStartResults/shearExp_grid.txt'
    optgrid = open(gridfile)
    optimizedgrid = np.loadtxt(optgrid)
    grid = optimizedgrid[:,0]


    layoutfile = 'src/florisse3D/Plots/multiStartResults/shearExp_layout.txt'
    optlayout = open(layoutfile)
    optimizedlayout = np.loadtxt(optlayout)
    layout = optimizedlayout[:,0]

    group1file = 'src/florisse3D/Plots/multiStartResults/shearExp_1group.txt'
    opt1group = open(group1file)
    optimized1group= np.loadtxt(opt1group)
    group1 = optimized1group[:,0]


    group2file = 'src/florisse3D/Plots/multiStartResults/shearExp_2group.txt'
    opt2group = open(group2file)
    optimized2group= np.loadtxt(opt2group)
    group2 = optimized2group[:,0]


    group25file = 'src/florisse3D/Plots/multiStartResults/shearExp_25group.txt'
    opt25group = open(group25file)
    optimized25group= np.loadtxt(opt25group)
    group25 = optimized25group[:,0]

    group1_yaw_file = 'src/florisse3D/Plots/multiStartResults/shearExp_1group_YAW.txt'
    opt1group_yaw = open(group1_yaw_file)
    optimized1group_yaw= np.loadtxt(opt1group_yaw)
    group1_yaw = optimized1group_yaw[:]

    group2_yaw_file = 'src/florisse3D/Plots/multiStartResults/shearExp_2group_YAW.txt'
    opt2group_yaw = open(group2_yaw_file)
    optimized2group_yaw= np.loadtxt(opt2group_yaw)
    group2_yaw = optimized2group_yaw[:,0]


    shearExp = np.array([0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,\
                        0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.3])

    plt.plot(shearExp, grid, 'ob', label='grid')
    plt.plot(shearExp, layout, 'or', label='layout')
    plt.plot(shearExp, group1, 'og', label='1 group')
    plt.plot(shearExp, group2, 'ok', label='2 groups')
    plt.plot(shearExp, group1_yaw, 'oy', label='1 group, yaw')
    plt.plot(shearExp, group2_yaw, 'ow', label='2 groups, yaw')
    # plt.plot(shearExp, group25, 'oy', label='25 groups')

    plt.axis([0.075,0.305,50,75])
    plt.legend(loc=3)
    plt.xlabel('Wind Shear Exponent')
    plt.ylabel('COE')
    plt.title('Optimized COE vs Wind Shear')
    plt.show()
