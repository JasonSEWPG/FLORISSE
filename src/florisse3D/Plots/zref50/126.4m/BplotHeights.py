import numpy as np
import matplotlib.pyplot as plt
import matplotlib
if __name__=="__main__":
    file = 'src/florisse3D/Plots/zref50/126.4m/B1heights.txt'
    opt = open(file)
    optimized = np.loadtxt(opt)
    group10 = optimized[:]

    file = 'src/florisse3D/Plots/zref50/126.4m/B2heights.txt'
    opt = open(file)
    optimized = np.loadtxt(opt)
    group20_1 = optimized[:,0]
    group20_2 = optimized[:,1]

    density = np.array([0.025,0.05,0.075,0.1,
        0.125,0.15,0.175,0.2,0.225,0.25,0.275,0.3])

    font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 18}

    matplotlib.rc('font', **font)

    # fig = plt.figure(1)
    # ax = fig.add_subplot(111)
    # ax.plot(density, group10, 'b',linewidth=3, label='hub height')
    # ax.plot([0,1],[90.,90.],'--k', label='90 m')
    # plt.xlabel('Turbine Density')
    # plt.ylabel('Optimized Hub Height (m)')
    # plt.title('126.4 m Rotor Diameter, 1 Group')
    # plt.legend(loc=4)
    # plt.tight_layout()
    # plt.axis([0.014842,0.25842,0.,120.])

    fig = plt.figure(2)
    ax = fig.add_subplot(111)
    ax.plot(density, group20_1, 'b',linewidth=3,label='hub height, group 1')
    ax.plot(density, group20_2, 'r',linewidth=3,label='hub height, group 2')
    ax.plot([0,1],[90.,90.],'--k', label='90 m')
    # ax.plot(density, group20_1-35., 'b',label='blade tip, group 1')
    # ax.plot(density, group20_2+35., 'r',label='blade tip, group 1')
    plt.xlabel('Turbine Density')
    plt.ylabel('Optimized Hub Height (m)')
    plt.title('Varied Turbine Density: Hub Heights')
    plt.legend(loc=4)
    plt.tight_layout()
    plt.axis([0.014842,0.25842,0.,120.])
    #
    # fig = plt.figure(3)
    # ax = fig.add_subplot(111)
    # ax.plot(density, group11, 'b')
    # plt.xlabel('Shear Exponent')
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
    # plt.xlabel('Shear Exponent')
    # plt.ylabel('Turbine Hub Height')
    # plt.title('Two Height Groups with yaw, No Layout Optimization')
    # plt.legend(loc=4)
    # plt.axis([0.075,0.305,0.,110.])

    plt.show()
