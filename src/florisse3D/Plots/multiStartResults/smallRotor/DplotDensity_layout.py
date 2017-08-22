import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
    layoutfile = 'src/florisse3D/Plots/multiStartResults/smallRotor/Ddensity_layout.txt'
    optlayout = open(layoutfile)
    optimizedlayout = np.loadtxt(optlayout)
    layout = optimizedlayout[:,1]

    # layoutfile_yaw = 'src/florisse3D/Plots/multiStartResults/smallRotor/Ddensity_layoutYAW.txt'
    # optlayout_yaw = open(layoutfile_yaw)
    # optimizedlayout_yaw = np.loadtxt(optlayout_yaw)
    # layout_yaw = optimizedlayout_yaw[:,1]
    #
    #
    # layoutfile = 'src/florisse3D/Plots/multiStartResults/smallRotor/Ddensity_layout_1group.txt'
    # optlayout = open(layoutfile)
    # optimizedlayout = np.loadtxt(optlayout)
    # layout_1group = optimizedlayout[:,1]
    #
    # layoutfile_yaw = 'src/florisse3D/Plots/multiStartResults/smallRotor/Ddensity_layout_1groupYAW.txt'
    # optlayout_yaw = open(layoutfile_yaw)
    # optimizedlayout_yaw = np.loadtxt(optlayout_yaw)
    # layout_1group_yaw = optimizedlayout_yaw[:,1]
    # #
    # layoutfile = 'src/florisse3D/Plots/multiStartResults/smallRotor/Ddensity_layout_2group.txt'
    # optlayout = open(layoutfile)
    # optimizedlayout = np.loadtxt(optlayout)
    # layout_2group = optimizedlayout[:,1]
    #
    # layoutfile_yaw = 'src/florisse3D/Plots/multiStartResults/smallRotor/Ddensity_layout_2groupYAW.txt'
    # optlayout_yaw = open(layoutfile_yaw)
    # optimizedlayout_yaw = np.loadtxt(optlayout_yaw)
    # layout_2group_yaw = optimizedlayout_yaw[:,1]


    density = np.array([0.024842,0.049684,0.0745261,0.0993681,0.12421,0.149052,0.173894,0.198736,0.223578,0.24842])

    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.plot(density, layout, 'ob', label='layout')
    # ax.plot(density, layout_yaw, 'oc', label='layout_yaw')
    # ax.plot(density, layout_1group, 'or', label='1 group')
    # ax.plot(density, layout_1group_yaw, 'om', label='1 group, yaw')
    # ax.plot(density, layout_2group, 'ok', label='2 group')
    # ax.plot(density, layout_2group_yaw, 'ow', label='2 group, yaw')
    plt.legend(loc=2)
    plt.axis([0.07,0.255,60,160])
    # handles, labels = ax.get_legend_handles_labels()
    # lgd = ax.legend(handles, labels, loc='1', bbox_to_anchor=(1.0,-0.1))
    plt.xlabel('Turbine Density')
    plt.ylabel('COE')
    plt.title('Optimized COE vs Turbine Density: layout')
    plt.show()
