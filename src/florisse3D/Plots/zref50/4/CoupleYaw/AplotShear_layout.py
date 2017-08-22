import numpy as np
import matplotlib.pyplot as plt
import matplotlib

if __name__=="__main__":
    layoutfile = 'src/florisse3D/Plots/zref50/4/CoupleYaw/Ashear_layout.txt'
    optlayout = open(layoutfile)
    optimizedlayout = np.loadtxt(optlayout)
    layout = optimizedlayout[:,1]

    #
    layoutfile = 'src/florisse3D/Plots/zref50/4/CoupleYaw/Ashear_layout_1.txt'
    optlayout = open(layoutfile)
    optimizedlayout = np.loadtxt(optlayout)
    layout_1group = optimizedlayout[:,1]
    #

    layoutfile = 'src/florisse3D/Plots/zref50/4/CoupleYaw/Ashear_layout_2.txt'
    optlayout = open(layoutfile)
    optimizedlayout = np.loadtxt(optlayout)
    layout_2group = optimizedlayout[:,1]
    #


    shearExp = np.array([0.09,0.11,0.13,0.15,0.17,0.19,0.21,0.23,0.25,0.27,0.29])

    font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 15}

    matplotlib.rc('font', **font)

    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.plot(shearExp, layout, 'ob', label='layout')
    ax.plot(shearExp, layout_1group, 'or', label='1 group, layout')
    ax.plot(shearExp, layout_2group, 'ok', label='2 groups, layout')

    plt.legend(loc=1)
    plt.axis([0.075,0.305,45,75])
    # handles, labels = ax.get_legend_handles_labels()
    # lgd = ax.legend(handles, labels, loc='1', bbox_to_anchor=(1.0,-0.1))
    plt.xlabel('Wind Shear Exponent')
    plt.ylabel('COE ($/MWh)')
    plt.title('70 m Rotor Diameter, Grid')
    plt.tight_layout()
    plt.show()
