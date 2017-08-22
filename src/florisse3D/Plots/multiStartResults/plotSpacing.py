import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
    gridfile = 'src/florisse3D/Plots/multiStartResults/spacing_grid.txt'
    optgrid = open(gridfile)
    optimizedgrid = np.loadtxt(optgrid)
    grid = optimizedgrid[:,0]


    layoutfile = 'src/florisse3D/Plots/multiStartResults/spacing_layout.txt'
    optlayout = open(layoutfile)
    optimizedlayout = np.loadtxt(optlayout)
    layout = optimizedlayout[:,0]


    group1file = 'src/florisse3D/Plots/multiStartResults/spacing_1group.txt'
    opt1group = open(group1file)
    optimized1group= np.loadtxt(opt1group)
    group1 = optimized1group[:,0]


    group2file = 'src/florisse3D/Plots/multiStartResults/spacing_2group.txt'
    opt2group = open(group2file)
    optimized2group= np.loadtxt(opt2group)
    group2 = optimized2group[:,0]


    group5file = 'src/florisse3D/Plots/multiStartResults/spacing_5group.txt'
    opt5group = open(group5file)
    optimized5group= np.loadtxt(opt5group)
    group5 = optimized5group[:,0]


    spacing = np.array([2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0,8.5,9.0,9.5,10.0])

    plt.plot(spacing, grid, 'ob', label='grid')
    plt.plot(spacing, layout, 'or', label='layout')
    plt.plot(spacing, group1, 'og', label='1 group')
    plt.plot(spacing, group2, 'ok', label='2 groups')
    # plt.plot(spacing, group5, 'oy', label='5 groups')

    plt.legend()
    plt.show()
