import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
    file = 'src/florisse3D/Plots/multiStartResults/smallRotor/B1group0.txt'
    opt = open(file)
    optimized = np.loadtxt(opt)
    group10 = optimized[:,1]

    file = 'src/florisse3D/Plots/multiStartResults/smallRotor/B2group0.txt'
    opt = open(file)
    optimized = np.loadtxt(opt)
    group20_1 = optimized[:,1]
    group20_2 = optimized[:,2]

    # file = 'src/florisse3D/Plots/multiStartResults/smallRotor/B1group1.txt'
    # opt = open(file)
    # optimized = np.loadtxt(opt)
    # group11 = optimized[:,1]
    #
    # file = 'src/florisse3D/Plots/multiStartResults/smallRotor/B2group1.txt'
    # opt = open(file)
    # optimized = np.loadtxt(opt)
    # group21_1 = optimized[:,1]
    # group21_2 = optimized[:,2]

    density = np.array([0.0248420225,0.049684045,0.0745260675,0.09936809,0.1242101126,
            0.1490521351,0.1738941576,0.1987361801,0.2235782026,0.2484202251])

    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.plot(density, group10, 'b')
    plt.xlabel('Turbine Density')
    plt.ylabel('Turbine Hub Height')
    plt.title('One Height Group, No Layout Optimization')
    plt.axis([0.02,0.25,0.,110.])

    fig = plt.figure(2)
    ax = fig.add_subplot(111)
    ax.plot(density, group20_1, 'b',linewidth=3,label='hub height, group 1')
    ax.plot(density, group20_2, 'r',linewidth=3,label='hub height, group 2')
    ax.plot(density, group20_1+35., 'b',label='blade tip, group 1')
    ax.plot(density, group20_2-35., 'r',label='blade tip, group 1')
    plt.xlabel('Turbine Density')
    plt.ylabel('Turbine Hub Height')
    plt.title('Two Height Groups, No Layout Optimization')
    plt.legend(loc=4)
    plt.axis([0.02,0.25,0.,110.])
    #
    # fig = plt.figure(3)
    # ax = fig.add_subplot(111)
    # ax.plot(density, group11, 'b')
    # plt.xlabel('Turbine Density')
    # plt.ylabel('Turbine Hub Height')
    # plt.title('One Height Group with yaw, No Layout Optimization')
    # plt.axis([0.075,0.305,0.,110.])
    #
    # fig = plt.figure(4)
    # ax = fig.add_subplot(111)
    # ax.plot(density, group21_1, 'b',linewidth=3,label='hub height, group 1')
    # ax.plot(density, group21_2, 'r',linewidth=3,label='hub height, group 2')
    # ax.plot(density, group21_1+35., 'b',label='blade tip, group 1')
    # ax.plot(density, group21_2-35., 'r',label='blade tip, group 1')
    # plt.xlabel('Turbine Density')
    # plt.ylabel('Turbine Hub Height')
    # plt.title('Two Height Groups with yaw, No Layout Optimization')
    # plt.legend(loc=4)
    # plt.axis([0.075,0.305,0.,110.])
    # ax.plot(density, grid_yaw, 'oc', label='grid_yaw')
    # ax.plot(density, grid_1group, 'or', label='1 group')
    # ax.plot(density, grid_1group_yaw, 'om', label='1 group, yaw')
    # ax.plot(density, grid_2group, 'ok', label='2 group')
    # ax.plot(density, grid_2group_yaw, 'ow', label='2 group, yaw')
    # plt.legend(loc=4)
    # plt.axis([0.075,0.305,75,115])
    # plt.xlabel('Wind Turbine Density')
    # plt.ylabel('COE')
    # plt.title('Optimized COE vs Wind Shear: Grid')
    plt.show()
