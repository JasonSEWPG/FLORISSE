import numpy as np
import matplotlib.pyplot as plt
import matplotlib

if __name__=="__main__":

    file = 'src/florisse3D/Plots/zref50/density/0.25B2heights.txt'
    opt = open(file)
    optimized = np.loadtxt(opt)
    group20_1 = optimized[:,0]
    group20_2 = optimized[:,1]

    density = np.array([0.025,0.05,0.075,0.1,0.125 ,0.15,0.175,0.2,0.225,0.25])

    font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 18}

    matplotlib.rc('font', **font)

    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax2 = ax.twiny()
    ax.plot(density, group20_1, 'b',linewidth=3,label='hub height, group 1')
    ax.plot(density[4:10], group20_2[4:10], 'r',linewidth=3,label='hub height, group 2')
    ax.plot(density[0:5], group20_2[0:5], 'r',linewidth=1.5)
    # ax.plot(shearExp, group20_2, 'r',linewidth=3,label='hub height, group 2')
    # ax.plot(density[0:3], group20_2[0:3], 'r',linewidth=1.5)
    # ax.plot(density[13:23], group20_2[13:23], 'r',linewidth=1.5)
    # ax.plot(density[2:14], group20_2[2:14], 'r',linewidth=3,label='hub height, group 2')
    ax.plot([0,1],[90.,90.],'--k')#, label='90 m')
    # ax.plot(shearExp, group20_1-35., 'b',label='blade tip, group 1')
    # ax.plot(shearExp, group20_2+35., 'r',label='blade tip, group 1')
    plt.xlabel('Shear Exponent')
    plt.ylabel('Optimized Hub Height (m)')
    plt.title('Farm 3: Heights')
    ax.legend(loc=4)
    plt.tight_layout()
    ax.set_xlim([0.,0.275])
    ax.set_ylim([0,125])
    ax.set_yticks([20,40,60,80,100,120])
    ax.text(0.1,82,'90 m')


    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks([0.0122718463,0.0490873852,0.0766990394,0.136353478,0.1963495408])
    ax2.set_xticklabels(['10','5','4','3','2.5'])
    ax2.set_xlabel('Grid Spacing (D)')

    ax.set_xlabel('Turbine Density')
    ax.set_ylabel('Hub Height (m)')

    plt.title('0.25 Shear Exponent: Large Rotor: Hub Heights', y=1.15)
    plt.tight_layout()
    plt.show()


    plt.show()
