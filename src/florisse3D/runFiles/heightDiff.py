import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":


    shear_ex = np.array([0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.3])
    # shear_ex = np.array([2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0,8.5,9.0,9.5,10.0])
    n = len(shear_ex)
    diff = np.zeros(n)

    for i in range(n):
        shearExp = shear_ex[i]
        opt_filenameXYZ = 'XYZ_XYZdt_%s.txt'%shear_ex[i]

        optXYZ = open(opt_filenameXYZ)
        optimizedXYZ = np.loadtxt(optXYZ)
        turbineX = optimizedXYZ[:,0]
        turbineY = optimizedXYZ[:,1]
        turbineZ = optimizedXYZ[:,2]
        turbineH1 = turbineZ[0]
        turbineH2 = turbineZ[1]

        diff[i] = abs(turbineH1-turbineH2)

    print diff
    plt.plot(shear_ex, diff)
    plt.plot(shear_ex, diff, 'ob')
    plt.xlabel('Shear Exponent')
    # plt.xlabel('Distance Between Turbines in the Starting Grid (Rotor Diameters)')
    # plt.axis([2.0, 10.0, -0.25, 70])
    plt.ylabel('Difference in Height Between Height Groups')
    plt.title('Difference Between Optimized Height Groups vs. Shear Exponent')
    plt.show()
