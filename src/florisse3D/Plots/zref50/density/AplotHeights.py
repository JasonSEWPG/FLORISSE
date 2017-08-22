import numpy as np
import matplotlib.pyplot as plt
import matplotlib

if __name__=="__main__":

    file = 'src/florisse3D/Plots/zref50/density/A2heights.txt'
    opt = open(file)
    optimized = np.loadtxt(opt)
    group20_2 = optimized[:,0]
    group20_1 = optimized[:,1]

    density = np.array([0.024842,0.049684,0.074526,0.099368,0.12421, \
    0.149052,0.173894,0.198736,0.223578,0.24842])

    font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 18}

    matplotlib.rc('font', **font)


    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax2 = ax.twiny()
    ax.plot(density, group20_2, 'b',linewidth=3,label='hub height, group 1')
    ax.plot(density, group20_1, 'r',linewidth=3,label='hub height, group 2')

    ax.plot([0,1],[90.,90.],'--k')#, label='90 m')
    ax.set_xlim([0.,0.275])
    ax.set_ylim([0,125])
    ax.set_yticks([20,40,60,80,100,120])
    # ax.text(0.24,82,'90 m')
    ax.text(0.12,82,'90 m')
    # ax.text(0.001,82,'90 m')

    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks([0.0099401955,0.039760782,0.0621262219,0.1104466167,0.1590431281])
    ax2.set_xticklabels(['10','5','4','3','2.5'])
    ax2.set_xlabel('Grid Spacing (D)')

    # ax.plot(shearExp, group20_1-35., 'b',label='blade tip, group 1')
    # ax.plot(shearExp, group20_2+35., 'r',label='blade tip, group 1')
    ax.set_xlabel('Turbine Density')
    ax.set_ylabel('Hub Height (m)')
    plt.title('0.15 Shear Exponent: Small Rotor: Hub Heights', y=1.15)
    ax.legend(loc=3)
    plt.tight_layout()
    # plt.axis([0.0,0.255,0.,120.])

    plt.show()
