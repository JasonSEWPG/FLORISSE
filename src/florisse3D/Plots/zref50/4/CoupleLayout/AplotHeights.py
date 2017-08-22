import numpy as np
import matplotlib.pyplot as plt
import matplotlib

if __name__=="__main__":
    file = 'src/florisse3D/Plots/zref50/4/CoupleLayout/A1heights.txt'
    opt = open(file)
    optimized = np.loadtxt(opt)
    group10 = optimized[:]

    file = 'src/florisse3D/Plots/zref50/4/CoupleLayout/A2heights.txt'
    opt = open(file)
    optimized = np.loadtxt(opt)
    group20_1 = optimized[:,0]
    group20_2 = optimized[:,1]

    shearExp = np.array([0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,\
                        0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.3])

    font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 15}

    matplotlib.rc('font', **font)

    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.plot(shearExp, group10, 'b',linewidth=3, label='hub height')
    ax.plot([0,1],[90.,90.],'--k', label='90 m')
    plt.xlabel('Shear Exponent')
    plt.ylabel('Optimized Hub Height (m)')
    plt.title('70 m Rotor Diameter, 1 Group')
    plt.legend(loc=4)
    plt.tight_layout()
    plt.axis([0.075,0.305,0.,120.])

    fig = plt.figure(2)
    ax = fig.add_subplot(111)
    ax.plot(shearExp, group20_1, 'b',linewidth=3,label='hub height, group 1')
    ax.plot(shearExp, group20_2, 'r',linewidth=3,label='hub height, group 2')
    ax.plot([0,1],[90.,90.],'--k', label='90 m')
    # ax.plot(shearExp, group20_1-35., 'b',label='blade tip, group 1')
    # ax.plot(shearExp, group20_2+35., 'r',label='blade tip, group 1')
    plt.xlabel('Shear Exponent')
    plt.ylabel('Optimized Hub Height (m)')
    plt.title('70 m Rotor Diameter, 2 Groups')
    plt.legend(loc=4)
    plt.tight_layout()
    plt.axis([0.075,0.305,0.,120.])

    plt.show()
