import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
    layoutfile = 'src/florisse3D/Plots/multiStartResults/smallRotor/Cshear_layout.txt'
    optlayout = open(layoutfile)
    optimizedlayout = np.loadtxt(optlayout)
    layout = optimizedlayout[:,1]

    # layoutfile_yaw = 'src/florisse3D/Plots/multiStartResults/smallRotor/Cshear_layoutYAW.txt'
    # optlayout_yaw = open(layoutfile_yaw)
    # optimizedlayout_yaw = np.loadtxt(optlayout_yaw)
    # layout_yaw = optimizedlayout_yaw[:,1]
    #
    #
    layoutfile = 'src/florisse3D/Plots/multiStartResults/smallRotor/Cshear_layout_1group.txt'
    optlayout = open(layoutfile)
    optimizedlayout = np.loadtxt(optlayout)
    layout_1group = optimizedlayout[:,1]
    #
    layoutfile_yaw = 'src/florisse3D/Plots/multiStartResults/smallRotor/Cshear_layout_1groupYAW.txt'
    optlayout_yaw = open(layoutfile_yaw)
    optimizedlayout_yaw = np.loadtxt(optlayout_yaw)
    layout_1group_yaw = optimizedlayout_yaw[:,1]
    #
    layoutfile = 'src/florisse3D/Plots/multiStartResults/smallRotor/Cshear_layout_2group.txt'
    optlayout = open(layoutfile)
    optimizedlayout = np.loadtxt(optlayout)
    layout_2group = optimizedlayout[:,1]
    #
    # layoutfile_yaw = 'src/florisse3D/Plots/multiStartResults/smallRotor/Cshear_layout_2groupYAW.txt'
    # optlayout_yaw = open(layoutfile_yaw)
    # optimizedlayout_yaw = np.loadtxt(optlayout_yaw)
    # layout_2group_yaw = optimizedlayout_yaw[:,1]

    shearExp = np.array([0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,\
                        0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.3])

    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.plot(shearExp, layout, 'ob', label='layout')
    # ax.plot(shearExp, layout_yaw, 'oc', label='layout_yaw')
    ax.plot(shearExp, layout_1group, 'or', label='1 group')
    ax.plot(shearExp, layout_1group_yaw, 'om', label='1 group, yaw')
    ax.plot(shearExp, layout_2group, 'ok', label='2 group')
    # ax.plot(shearExp, layout_2group_yaw, 'ow', label='2 group, yaw')
    plt.legend(loc=4)
    plt.axis([0.075,0.305,75,115])
    # handles, labels = ax.get_legend_handles_labels()
    # lgd = ax.legend(handles, labels, loc='1', bbox_to_anchor=(1.0,-0.1))
    plt.xlabel('Wind Shear Exponent')
    plt.ylabel('COE')
    plt.title('Optimized COE vs Wind Shear: Layout')
    plt.show()
