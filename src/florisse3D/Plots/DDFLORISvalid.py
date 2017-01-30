import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
    t1S = np.array([1.955666, 1.953151, 1.955127])
    t1F = np.array([1.9549664602300143, 1.9549664602300143, 1.9549664602300143])
    error1 = 100*(t1S-t1F)/t1S

    t2S = np.array([0.918373, 1.051117, 1.235663])
    t2F = np.array([0.91297001813109668, 1.061089170817713, 1.229565320799221])
    error2 = 100*(t2S-t2F)/t2S

    totalS = t1S+t2S
    totalF = t1F+t2F
    errortotal = 100*(totalS-totalF)/totalS

    x = np.array([-25,0,25])

    f, ax = plt.subplots(1,1)

    ax.scatter(x, t2S, s=30, c='red')
    ax.scatter(x, t2F, s=30, c='blue')
    ax.axes.get_yaxis().set_ticks([0.9,1.0,1.1,1.2,1.3])
    ax.axes.get_xaxis().set_visible(False)
    ax.set_ylabel('Turbine 2 (MW)', fontsize=15)
    ax.legend(loc=2,prop={'size':10})





    plt.show()
    f.savefig('sofwa.pdf', transparent=True)

    # # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
    # plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
    # plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
