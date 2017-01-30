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

    f, ax = plt.subplots(3,2)

    ax[0, 0].scatter(x, t1S, s=30, facecolors='none', edgecolors='r', label='SOWFA')
    ax[0, 0].set_ylim([1.85, 2.05])
    ax[0, 0].set_ylabel('Turbine 1 (MW)', fontsize=15)
    ax[0, 0].axes.get_xaxis().set_visible(False)
    ax[0, 0].scatter(x, t1F, s=30, facecolors='none', edgecolors='b', label='FLORISSE')
    ax[0, 0].legend(loc=2,prop={'size':10})
    ax[1, 0].scatter(x, t2S, s=30, facecolors='none', edgecolors='r')
    ax[1, 0].scatter(x, t2F, s=30, facecolors='none', edgecolors='b')

    ax[1, 0].axes.get_yaxis().set_ticks([0.9,1.0,1.1,1.2,1.3])
    ax[1, 0].axes.get_xaxis().set_visible(False)
    ax[1, 0].set_ylabel('Turbine 2 (MW)', fontsize=15)
    ax[2, 0].scatter(x, totalS, s=30, facecolors='none', edgecolors='r')
    ax[2, 0].scatter(x, totalF, s=30, facecolors='none', edgecolors='b')
    ax[2, 0].set_xlabel('Height of Turbine 2\n Relative to Turbine 1 (m)', fontsize=15)
    ax[2, 0].set_ylabel('Total (MW)', fontsize=15)

    ax[0, 1].scatter(x, error1, s=30, facecolors='none', edgecolors='g')
    ax[0, 1].axes.get_xaxis().set_visible(False)
    ax[0, 1].set_ylim([-2, 2])
    ax[0, 1].set_ylabel('% Error', fontsize=15)
    ax[0, 1].yaxis.set_label_position("right")
    ax[1, 1].scatter(x, error2, s=30, facecolors='none', edgecolors='g')
    ax[1, 1].axes.get_xaxis().set_visible(False)
    ax[1, 1].set_ylim([-2, 2])
    ax[1, 1].set_ylabel('% Error', fontsize=15)
    ax[1, 1].yaxis.set_label_position("right")
    ax[2, 1].scatter(x, errortotal, s=30, facecolors='none', edgecolors='g')
    ax[2, 1].set_ylim([-2, 2])
    ax[2, 1].set_xlabel('Height of Turbine 2\n Relative to Turbine 1 (m)', fontsize=15)
    ax[2, 1].set_ylabel('% Error', fontsize=15)
    ax[2, 1].yaxis.set_label_position("right")

    plt.show()
    f.savefig('sofwa.pdf', transparent=True)

    # # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
    # plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
    # plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
