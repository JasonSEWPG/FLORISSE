import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
    groupsfile = 'src/florisse3D/Plots/multiStartResults/nGroups.txt'
    groups2file = 'src/florisse3D/Plots/multiStartResults/nGroups2.txt'
    opt = open(groupsfile)
    optimized = np.loadtxt(opt)
    groups = optimized[:]
    opt2 = open(groups2file)
    optimized2 = np.loadtxt(opt2)
    groups2 = optimized2[:]

    nGroups = np.array([1,2,3,4,5,6,7,8,9,10])

    plt.figure(1)
    plt.plot(nGroups,groups,'ob')
    plt.axis([0, 11, 62, 64])
    plt.xlabel('Number of Height Groups')
    plt.ylabel('Optimized COE')
    plt.title('Low Shear (0.08), High Density')


    plt.figure(2)
    plt.plot(nGroups,groups2,'ob')
    plt.axis([0, 11, 58., 62.])
    plt.xlabel('Number of Height Groups')
    plt.ylabel('Optimized COE')
    plt.title('Low Shear (0.1), Medium Density')
    plt.show()
