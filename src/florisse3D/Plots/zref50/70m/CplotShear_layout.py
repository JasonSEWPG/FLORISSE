import numpy as np
import matplotlib.pyplot as plt
import matplotlib

if __name__=="__main__":
    layoutfile = 'src/florisse3D/Plots/zref50/70m/Cshear_layout.txt'
    optlayout = open(layoutfile)
    optimizedlayout = np.loadtxt(optlayout)
    layout = optimizedlayout[:,1]

    layoutfile = 'src/florisse3D/Plots/zref50/70m/Cshear_layout_1.txt'
    optlayout = open(layoutfile)
    optimizedlayout = np.loadtxt(optlayout)
    layout_1group = optimizedlayout[:,1]

    layoutfile = 'src/florisse3D/Plots/zref50/70m/Cshear_layout_2.txt'
    optlayout = open(layoutfile)
    optimizedlayout = np.loadtxt(optlayout)
    layout_2group = optimizedlayout[:,1]


    shearExp = np.array([0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,\
                        0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.3])
    #
    fig = plt.figure(1)

    font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 15}

    matplotlib.rc('font', **font)
    ax = fig.add_subplot(111)
    ax.plot(shearExp, layout, 'ob', label='layout')
    ax.plot(shearExp, layout_1group, 'or', label='1 group, layout')
    ax.plot(shearExp, layout_2group, 'ok', label='2 groups, layout')
    plt.legend(loc=1)
    plt.axis([0.075,0.305,40,75])
    # # handles, labels = ax.get_legend_handles_labels()
    # # lgd = ax.legend(handles, labels, loc='1', bbox_to_anchor=(1.0,-0.1))
    plt.xlabel('Wind Shear Exponent')
    plt.ylabel('COE ($/MWh)')
    plt.title('Small Rotor, Layout, 5')
    plt.tight_layout()
    plt.show()
