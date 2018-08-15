import numpy as np
import matplotlib.pyplot as plt
import matplotlib

if __name__=="__main__":
    f, ax = plt.subplots(2, 2, figsize=(12,9.25),sharex=True)#,sharex=True,sharey=True)
    shearExp = np.array([0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,\
                        0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.3])
    font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 18}

    matplotlib.rc('font', **font)

    gridfile = 'src/florisse3D/Plots/zref50/3/Ashear_grid.txt'
    gridfile = 'Ashear_grid.txt'
    optgrid = open(gridfile)
    optimizedgrid = np.loadtxt(optgrid)
    grid = optimizedgrid[:,1]

    #
    gridfile = 'src/florisse3D/Plots/zref50/3/Ashear_grid_1.txt'
    gridfile = 'Ashear_grid_1.txt'
    optgrid = open(gridfile)
    optimizedgrid = np.loadtxt(optgrid)
    grid_1group = optimizedgrid[:,1]
    #

    gridfile = 'src/florisse3D/Plots/zref50/3/Ashear_grid_2.txt'
    gridfile = 'Ashear_grid_2.txt'
    optgrid = open(gridfile)
    optimizedgrid = np.loadtxt(optgrid)
    grid_2group = optimizedgrid[:,1]

    # ax.plot(shearExp, grid, 'ob', label='grid')
    ax[0][0].plot(shearExp, grid, 'sb', label='baseline',markersize=8)
    ax[0][0].plot(shearExp, grid_1group, 'or', label='1 group')
    ax[0][0].plot(shearExp, grid_2group, 'ok', label='2 groups',markersize=4)
    # ax.plot(shearExp, grid_1group, '*r', label='1 optimized height')
    # ax.plot(shearExp, grid_2group, '^k', label='2 height groups')

    ax[0][0].legend(loc=4)
    # plt.axis([0.075,0.305,55,90])
    # handles, labels = ax.get_legend_handles_labels()
    # lgd = ax.legend(handles, labels, loc='1', bbox_to_anchor=(1.0,-0.1))
    ax[0][0].set_ylabel('COE ($/MWh)')
    ax[0][0].set_ylim(0.,100.)
    # ax[0][0].title('Small Farm: Small Rotor')
    # plt.title('Varied Wind Shear Exponent')
    plt.tight_layout()

    file = 'src/florisse3D/Plots/zref50/3/A2heights.txt'
    file = 'A2heights.txt'
    opt = open(file)
    optimized = np.loadtxt(opt)
    group20_2 = optimized[:,0]
    group20_1 = optimized[:,1]

    ax[0][1].plot(shearExp, group20_1, 'b',linewidth=3,label='hub height, group 1')
    ax[0][1].plot(shearExp, group20_2, 'r',linewidth=3,label='hub height, group 2')
    ax[0][1].plot([0,1],[90.,90.],'--k')#, label='90 m')
    ax[0][1].text(0.18,82,'90 m')
    # ax.plot(shearExp, group20_1-35., 'b',label='blade tip, group 1')
    # ax.plot(shearExp, group20_2+35., 'r',label='blade tip, group 1')
    ax[0][1].set_ylabel('Optimized Hub Height (m)')
    ax[0][1].set_ylim(0.,120.)
    # ax[0][1].title('Small Farm: Small Rotor: Hub Heights')
    ax[0][1].legend(loc=4)
    # plt.tight_layout()
    # ax[0][1].set_axis([0.075,0.305,0.,120.])

    """SMALL"""
    """baseline"""
    ideal_AEPb = np.array([  4.10465416e+08,   4.17767591e+08,   4.25199673e+08,
             4.32763970e+08,   4.40462837e+08,   4.48298666e+08,
             4.56273895e+08,   4.64391003e+08,   4.72652514e+08,
             4.80948195e+08,   4.88928743e+08,   4.97051265e+08,
             5.05318286e+08,   5.13678226e+08,   5.21896202e+08,
             5.30260376e+08,   5.38570749e+08,   5.46787472e+08,
             5.55060792e+08,   5.63261342e+08,   5.71475307e+08,
             5.79223386e+08,   5.86466197e+08])
    AEPb = np.array([  3.32025425e+08,   3.37932154e+08,   3.43943963e+08,
             3.50062723e+08,   3.56290335e+08,   3.62628736e+08,
             3.69079898e+08,   3.75645825e+08,   3.82328561e+08,
             3.89106507e+08,   3.95907921e+08,   4.02830332e+08,
             4.09875892e+08,   4.17030079e+08,   4.24221821e+08,
             4.31541504e+08,   4.38948883e+08,   4.46369475e+08,
             4.53841695e+08,   4.61374767e+08,   4.68974588e+08,
             4.76554957e+08,   4.84106713e+08])
    tower_costb = np.array([ 33623302.70392097,  33646203.34831154,  33668589.78934412,
            33692474.2789682 ,  33716183.53672943,  33738601.59064662,
            33762560.133247  ,  33787465.9042061 ,  33812224.16363604,
            33836008.52188704,  33860961.72435628,  33886299.78171607,
            33912976.08816385,  33938710.40879235,  33963953.91642834,
            33991477.81607558,  34018332.21902361,  34043470.33565913,
            34073004.96871205,  34100808.47193441,  34128578.17181974,
            34150037.28091785,  34184633.58843114])
    wake_lossb = np.array([ 19.11001226,  19.11001226,  19.11001226,  19.11001226,
            19.11001226,  19.11001226,  19.11001226,  19.11001226,
            19.11001226,  19.09596267,  19.0254354 ,  18.95597898,
            18.88757977,  18.81491992,  18.71528872,  18.61705614,
            18.49745208,  18.36508741,  18.23567765,  18.08868585,
            17.93615896,  17.72518725,  17.45360349])

    """1 group"""
    ideal_AEP1 = np.array([  3.80582227e+08,   3.86008237e+08,   3.92349241e+08,
             4.03964668e+08,   4.16503562e+08,   4.28884306e+08,
             4.35040476e+08,   4.41288621e+08,   4.47621670e+08,
             4.54316444e+08,   4.65922871e+08,   4.79303087e+08,
             4.90765720e+08,   5.05196607e+08,   5.18085846e+08,
             5.33937720e+08,   5.44072087e+08,   5.58221879e+08,
             5.70052903e+08,   5.80131653e+08,   5.92579178e+08,
             6.03311781e+08,   6.12218000e+08])
    AEP1 = np.array([  3.07852916e+08,   3.12242016e+08,   3.17371253e+08,
             3.26766971e+08,   3.36909680e+08,   3.46924462e+08,
             3.51904187e+08,   3.56958312e+08,   3.62081114e+08,
             3.67496516e+08,   3.76884954e+08,   3.87708208e+08,
             3.97473483e+08,   4.09772191e+08,   4.20887290e+08,
             4.34759633e+08,   4.43934311e+08,   4.56747104e+08,
             4.67629787e+08,   4.77465787e+08,   4.90537456e+08,
             5.01774691e+08,   5.10827959e+08])
    tower_cost1 = np.array([ 16148460.62586858,  16846891.25405893,  17750706.02986099,
            20132317.94840223,  22530719.40113551,  24616717.76340432,
            24631934.5155254 ,  24647202.16436171,  24661351.84414344,
            24754650.94066271,  26211618.95978993,  28011547.29311379,
            29326188.51107006,  31356199.10710458,  32818499.75960995,
            35058453.66280904,  35616249.42262504,  37287338.74469142,
            38228504.06060486,  38794644.41251355,  40330388.36312236,
            41187801.6434521 ,  41380022.57770525])
    wake_loss1 = np.array([ 19.11001226,  19.11001226,  19.11001226,  19.11001226,
            19.11001226,  19.11001226,  19.11001226,  19.11001226,
            19.11001226,  19.11001226,  19.11001226,  19.11001226,
            19.00952606,  18.88857028,  18.76109083,  18.57484178,
            18.40524052,  18.17821533,  17.96730009,  17.69699423,
            17.2199303 ,  16.82995317,  16.56110088])


    """2 groups"""
    ideal_AEP2 = np.array([  3.75302501e+08,   3.79214876e+08,   3.83570146e+08,
             3.91944070e+08,   4.02189471e+08,   4.13096114e+08,
             4.23407435e+08,   4.33030228e+08,   4.43003473e+08,
             4.53103559e+08,   4.59878960e+08,   4.68169114e+08,
             4.76680767e+08,   4.89609566e+08,   5.16585214e+08,
             5.24604202e+08,   5.32634669e+08,   5.40696885e+08,
             5.48435919e+08,   5.57871213e+08,   5.75112439e+08,
             5.93125481e+08,   6.12218301e+08])
    AEP2 = np.array([  3.19923444e+08,   3.24576206e+08,   3.29676031e+08,
             3.36980184e+08,   3.45567472e+08,   3.54649875e+08,
             3.62471569e+08,   3.68941857e+08,   3.75447333e+08,
             3.82209579e+08,   3.87231368e+08,   3.93671949e+08,
             4.01035340e+08,   4.09684404e+08,   4.25305760e+08,
             4.32481195e+08,   4.39742917e+08,   4.47090973e+08,
             4.54453261e+08,   4.63011223e+08,   4.76807936e+08,
             4.92220508e+08,   5.10828264e+08])
    tower_cost2 = np.array([ 18034083.76835743,  18966233.51424982,  19990797.55853888,
            21740751.65476598,  23753763.21608059,  25729595.02630559,
            27037235.94703647,  27682023.60292311,  28244566.7699696 ,
            28775963.49347209,  28698549.56700625,  28999723.57504175,
            29450297.76843689,  30359500.33693677,  33442196.51148397,
            33492932.99028545,  33545139.60379894,  33595482.69446328,
            33649105.43458862,  34073121.97303188,  36000741.02755602,
            38301138.44859688,  41380058.71817537])
    wake_loss2 = np.array([ 14.75584551,  14.40836683,  14.05065417,  14.02340041,
            14.07843906,  14.14833907,  14.39177983,  14.79997613,
            15.24957343,  15.64630828,  15.7971114 ,  15.91244762,
            15.86920065,  16.32426485,  17.66977676,  17.56047833,
            17.4400498 ,  17.31208631,  17.13648841,  17.00392278,
            17.09309272,  17.01241568,  16.56109219])


    ax[1][0].plot(shearExp,wake_lossb,'sb',label='baseline',markersize=8)
    ax[1][0].plot(shearExp,wake_loss1,'or', label='1 group')
    ax[1][0].plot(shearExp,wake_loss2,'ok', label='2 groups',markersize=4)
    # ax[1][0].title('Small Farm: Small Rotor')
    # ax[1][0].set_xlabel('Wind Shear Exponent')
    ax[1][0].set_ylabel('% Wake Loss')
    ax[1][0].set_ylim(0.,60.)
    ax[1][0].set_xlim(0.075,0.305)
    ax[1][0].set_xlabel('Wind Shear Exponent')
    # ax[1][0].legend(loc=4)
    # plt.savefig('small3_wl.pdf', transparent=True)


    ax[1][1].plot(shearExp,ideal_AEPb/1000000.,'s',markerfacecolor='none',markeredgecolor='blue',markersize=8)
    ax[1][1].plot(shearExp,ideal_AEP1/1000000.,'o',markerfacecolor='none',markeredgecolor='red')
    ax[1][1].plot(shearExp,ideal_AEP2/1000000.,'o',markerfacecolor='none',markeredgecolor='black',label='ideal AEP',markersize=4)
    ax[1][1].plot(shearExp,AEPb/1000000.,'sb',markersize=8)
    ax[1][1].plot(shearExp,AEP1/1000000.,'or')
    ax[1][1].plot(shearExp,AEP2/1000000.,'ok',label='true AEP',markersize=4)
    # ax[1][1].title('Small Farm: Small Rotor')
    # ax[1][1].set_xlabel('Wind Shear Exponent')
    ax[1][1].set_xlim(0.075,0.305)
    ax[1][1].set_ylim(0.,700.)
    ax[1][1].set_ylabel('AEP (GWh)')
    ax[1][1].set_xlabel('Wind Shear Exponent')
    ax[1][1].legend(loc=4)
    # plt.savefig('small3_AEP.pdf', transparent=True)
    #
    # plt.figure(3)
    # plt.plot(shearExp,tower_costb,'ob',label='baseline')
    # plt.plot(shearExp,tower_cost1,'or', label='1 group')
    # plt.plot(shearExp,tower_cost2,'ok', label='2 groups')
    # plt.title('Small Farm: Small Rotor')
    # plt.xlabel('Wind Shear Exponent')
    # plt.ylabel('Tower Cost')
    # plt.xlim(0.075,0.305)
    # plt.legend(loc=4)
    # plt.savefig('small3_tc.pdf', transparent=True)

    plt.suptitle('Big Wind Farm: Small Rotor',fontsize=18,y=0.98)
    f.tight_layout(rect=[0, 0.0, 1, 0.95])
    plt.savefig('big_farm_small_rotor.pdf', transparent=True)

    plt.show()
