import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
    shear_file = 'src/florisse3D/Plots/multiStartResults/shearHEIGHTS.txt'
    shear = open(shear_file)
    shear_ = np.loadtxt(shear)
    shear1 = shear_[:,1]
    shear2 = shear_[:,2]

    density_file = 'src/florisse3D/Plots/multiStartResults/densityHEIGHTS.txt'
    density = open(density_file)
    density_ = np.loadtxt(density)
    density1 = density_[:,1]
    density2 = density_[:,2]


    shear = np.array([0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,\
            0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.3])

    density = np.array([0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25,0.275,0.3])

    plt.figure(1)
    plt.plot(shear,shear1, 'ob')
    plt.plot(shear,shear2, 'or')
    plt.xlabel('Shear Exponent')
    plt.ylabel('Turbine Height (m)')
    plt.title('Optimized Turbine Heights vs. Shear Exponent')
    # plt.show()

    plt.figure(2)
    plt.plot(density,density1, 'ob')
    plt.plot(density,density2, 'or')
    plt.xlabel('Turbine Density')
    plt.ylabel('Turbine Height (m)')
    plt.title('Optimized Turbine Heights vs. Turbine Density')
    plt.show()
