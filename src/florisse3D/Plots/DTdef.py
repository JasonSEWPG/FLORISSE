import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
    h = 75.
    mid = h/2.

    db = 6.3
    dm = 5.6
    dt = 3.6

    bl = 0.
    br = db
    ml = (db-dm)/2.
    mr = 6.3-((db-dm)/2.)
    tl = (db-dt)/2.
    tr = 6.3-((db-dt)/2.)


    towerX = np.array([bl,ml,tl,tr,mr,br,bl,ml,mr,br])
    towerY = np.array([0,mid,h,h,mid,0,0,mid,mid,0])

    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.tight_layout()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    plt.plot(towerX,towerY,linewidth=3)
    plt.axis([-70,20,-10,h+10])
    ax.fill_between(towerX, 0, towerY, facecolor='blue', alpha=0.25)

    ax.text(-5, h-5, r'$H$', fontsize=15)
    ax.text(-9, mid-5, r'$0.5H$', fontsize=15)
    ax.text(-4, -5, r'$0$', fontsize=15)

    ax.text(12, h-1, r'$d_3, t_3$', fontsize=15)
    ax.text(12, mid-1, r'$d_2, t_2$', fontsize=15)
    ax.text(12, -1, r'$d_1, t_1$', fontsize=15)

    ax.arrow(-1, -7, 0, 5, head_width=1, head_length=1, fc='k', ec='k')
    ax.arrow(-1, mid-7, 0, 5, head_width=1, head_length=1, fc='k', ec='k')
    ax.arrow(-1, h-7, 0, 5, head_width=1, head_length=1, fc='k', ec='k')

    ax.arrow(11, 0, -2.8, 0, head_width=1, head_length=1, fc='k', ec='k')
    ax.arrow(11, mid, -3, 0, head_width=1, head_length=1, fc='k', ec='k')
    ax.arrow(11, h, -4, 0, head_width=1, head_length=1, fc='k', ec='k')

    plt.plot(np.array([-7,-1]),np.array([0,0]),'--k')
    plt.plot(np.array([-7,-1]),np.array([mid,mid]),'--k')
    plt.plot(np.array([-7,-1]),np.array([h,h]),'--k')

    plt.show()
