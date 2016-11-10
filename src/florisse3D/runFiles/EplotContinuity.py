import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
    opt_filenameXYZ = 'contY.txt'
    optXYZ = open(opt_filenameXYZ)
    optimizedXYZ = np.loadtxt(optXYZ)

    COE = optimizedXYZ[:,0]
    AEP = optimizedXYZ[:,1]

    print len(COE)
    print len(AEP)

    num = np.linspace(-1,1,200)
    print len(num)

    plt.figure(2)
    plt.scatter(num,COE)
    plt.title('COE')
    # plt.axis([-5, 5, 64, 65])

    # plt.figure(3)
    # plt.scatter(num,AEP)
    # plt.title('AEP')
    plt.show()
