import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
    num = np.linspace(-5,5,50)
    #opt_filenameXYZ = 'XYZ3_warmSAME_6.0.txt'
    #changing X
    COE = np.array([ 68.47443532 , 68.47443532  ,68.47443532,  68.47443532 , 68.47443532,
  68.46111286,  68.46111286,  68.46111286,  68.46111286,  68.46111286,
  68.44792326,  68.44792326,  68.44792326,  68.44792326,  68.44792326,
  68.43473839,  68.43473839,  68.43473839,  68.43473839,  68.43473839,
  68.42267599,  68.42267599,  68.42267599,  68.42267599,  68.42267599,
  68.4120276 ,  68.4120276 ,  68.4120276 ,  68.4120276 ,  68.4120276,
  68.40264736,  68.40264736,  68.40264736,  68.40264736,  68.40264736,
  68.39354963,  68.39354963,  68.39354963,  68.39354963,  68.39354963,
  68.3849695 ,  68.3849695 ,  68.3849695 ,  68.3849695 ,  68.3849695,
  68.37694892,  68.37694892,  68.37694892,  68.37694892,  68.37694892])

    plt.plot(num,COE)
    plt.title('Sweep Over Half of the X values')
    plt.ylabel('COE')
    plt.xlabel('Amount Added to original X')
    plt.show()

    opt_filenameXYZ = 'continuityX.txt'
    optXYZ = open(opt_filenameXYZ)
    optimizedXYZ = np.loadtxt(optXYZ)
    print optimizedXYZ
    print np.shape(optimizedXYZ)
    COE = optimizedXYZ[:,0]
    AEP = optimizedXYZ[:,1]

    print len(COE)
    print len(AEP)

    num = np.linspace(-5,5,200)
    print len(num)

    plt.figure(1)
    plt.plot(num,COE)

    plt.figure(2)
    plt.plot(num,AEP)
    plt.show()
