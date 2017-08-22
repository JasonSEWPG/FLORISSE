import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
    file = 'src/florisse3D/Plots/multiStartResults/5spacing/A1group0.txt'
    opt = open(file)
    optimized = np.loadtxt(opt)
    group10 = optimized[:,1]

    file = 'src/florisse3D/Plots/multiStartResults/5spacing/A2group0.txt'
    opt = open(file)
    optimized = np.loadtxt(opt)
    group20_1 = optimized[:,1]
    group20_2 = optimized[:,2]
    #
    # file = 'src/florisse3D/Plots/multiStartResults/5spacing/A1group1.txt'
    # opt = open(file)
    # optimized = np.loadtxt(opt)
    # group11 = optimized[:,1]
    #
    # file = 'src/florisse3D/Plots/multiStartResults/5spacing/A2group1.txt'
    # opt = open(file)
    # optimized = np.loadtxt(opt)
    # group21_1 = optimized[:,1]
    # group21_2 = optimized[:,2]

    shearExp = np.array([0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,\
                        0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.3])

    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.plot(shearExp, group10, 'b')
    plt.xlabel('Shear Exponent')
    plt.ylabel('Turbine Hub Height')
    plt.title('One Height Group, No Layout Optimization')
    plt.axis([0.075,0.305,0.,110.])

    fig = plt.figure(2)
    ax = fig.add_subplot(111)
    ax.plot(shearExp, group20_1, 'b',linewidth=3,label='hub height, group 1')
    ax.plot(shearExp, group20_2, 'r',linewidth=3,label='hub height, group 2')
    ax.plot(shearExp, group20_1-35., 'b',label='blade tip, group 1')
    ax.plot(shearExp, group20_2+35., 'r',label='blade tip, group 1')
    plt.xlabel('Shear Exponent')
    plt.ylabel('Turbine Hub Height')
    plt.title('Two Height Groups, No Layout Optimization')
    plt.legend(loc=4)
    plt.axis([0.075,0.305,0.,110.])
    #
    # fig = plt.figure(3)
    # ax = fig.add_subplot(111)
    # ax.plot(shearExp, group11, 'b')
    # plt.xlabel('Shear Exponent')
    # plt.ylabel('Turbine Hub Height')
    # plt.title('One Height Group with yaw, No Layout Optimization')
    # plt.axis([0.075,0.305,0.,110.])
    #
    # fig = plt.figure(4)
    # ax = fig.add_subplot(111)
    # ax.plot(shearExp, group21_1, 'b',linewidth=3,label='hub height, group 1')
    # ax.plot(shearExp, group21_2, 'r',linewidth=3,label='hub height, group 2')
    # ax.plot(shearExp, group21_1+35., 'b',label='blade tip, group 1')
    # ax.plot(shearExp, group21_2-35., 'r',label='blade tip, group 1')
    # plt.xlabel('Shear Exponent')
    # plt.ylabel('Turbine Hub Height')
    # plt.title('Two Height Groups with yaw, No Layout Optimization')
    # plt.legend(loc=4)
    # plt.axis([0.075,0.305,0.,110.])

    plt.show()
