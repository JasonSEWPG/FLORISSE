import numpy as np
import matplotlib.pyplot as plt
import matplotlib

if __name__=="__main__":

    file = 'src/florisse3D/Plots/zref50/4/CoupleLayout/B2heights.txt'
    opt = open(file)
    optimized = np.loadtxt(opt)
    group20_1 = optimized[:,0]
    group20_2 = optimized[:,1]

    shearExp = np.array([0.09,0.11,0.13,0.15,0.17,0.19,\
                        0.21,0.23,0.25,0.27,0.29])

    font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 18}

    matplotlib.rc('font', **font)

    fig = plt.figure(2)
    ax = fig.add_subplot(111)
    ax.plot(shearExp[1:4], group20_1[1:4], 'b',linewidth=3,label='hub height, group 1')
    ax.plot(shearExp, group20_2, 'r',linewidth=3,label='hub height, group 2')
    ax.plot(shearExp[0:2], group20_1[0:2], 'b',linewidth=1.5)
    ax.plot(shearExp[3:11], group20_1[3:11], 'b',linewidth=1.5)
    ax.plot([0,1],[90.,90.],'--k', label='90 m')
    # ax.plot(shearExp, group20_1-35., 'b',label='blade tip, group 1')
    # ax.plot(shearExp, group20_2+35., 'r',label='blade tip, group 1')
    plt.xlabel('Shear Exponent')
    plt.ylabel('Optimized Hub Height (m)')
    plt.title('Farm 2: Hub Heights')
    plt.legend(loc=4)
    plt.tight_layout()
    plt.axis([0.075,0.305,0.,120.])

    plt.show()
