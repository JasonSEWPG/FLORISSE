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

    ax[0][0].legend(loc=1)
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
    ideal_AEPb = np.array([  4.13075682e+08,   4.20424294e+08,   4.27903638e+08,
             4.35516040e+08,   4.43263866e+08,   4.51149525e+08,
             4.59175471e+08,   4.67344198e+08,   4.75658246e+08,
             4.83800955e+08,   4.91832254e+08,   5.00006429e+08,
             5.08326023e+08,   5.16615858e+08,   5.24886095e+08,
             5.33303459e+08,   5.41507934e+08,   5.49776909e+08,
             5.57992195e+08,   5.66244895e+08,   5.74368896e+08,
             5.81891933e+08,   5.89056286e+08])
    AEPb = np.array([  3.63481089e+08,   3.69947414e+08,   3.76528775e+08,
             3.83227218e+08,   3.90044826e+08,   3.96983720e+08,
             4.04046056e+08,   4.11234031e+08,   4.18549880e+08,
             4.25880950e+08,   4.33250182e+08,   4.40750513e+08,
             4.48384274e+08,   4.56061403e+08,   4.63788229e+08,
             4.71652515e+08,   4.79461656e+08,   4.87267263e+08,
             4.95104956e+08,   5.02978377e+08,   5.10848598e+08,
             5.18519462e+08,   5.26129856e+08])
    tower_costb = np.array([ 16488618.45124384,  16486922.76713727,  16490808.84591367,
            16488353.12394327,  16495875.93464068,  16498502.98100495,
            16499129.76014973,  16500725.54187624,  16499147.08831158,
            16503444.01107254,  16504339.40309645,  16507294.63172741,
            16508227.0336556 ,  16509793.9630091 ,  16507991.44958882,
            16512954.73405659,  16514580.53155707,  16516981.94506074,
            16518839.8749893 ,  16518952.67934937,  16521845.23930573,
            16523600.92069299,  16524263.60238877])
    wake_lossb = np.array([ 12.00617588,  12.00617588,  12.00617588,  12.00617588,
            12.00617588,  12.00617588,  12.00617588,  12.00617588,
            12.00617588,  11.97186669,  11.91098618,  11.85103088,
            11.79198896,  11.72136976,  11.64021416,  11.56019943,
            11.45805506,  11.37000197,  11.27027206,  11.1729957 ,
            11.0591465 ,  10.89076294,  10.68258363])

    """1 group"""
    ideal_AEP1 = np.array([  3.93091892e+08,   3.97612765e+08,   4.02185631e+08,
             4.13284020e+08,   4.29448191e+08,   4.47513536e+08,
             4.62741966e+08,   4.79339340e+08,   4.93600797e+08,
             5.08069143e+08,   5.22225605e+08,   5.32134525e+08,
             5.42433759e+08,   5.52699770e+08,   5.64617184e+08,
             5.74387698e+08,   5.83337947e+08,   5.92122974e+08,
             6.02227329e+08,   6.10699270e+08,   6.25454804e+08,
             6.38712044e+08,   6.52572663e+08])
    AEP1 = np.array([  3.45896588e+08,   3.49874677e+08,   3.53898517e+08,
             3.63664414e+08,   3.77887886e+08,   3.93784274e+08,
             4.07184351e+08,   4.21789016e+08,   4.34872934e+08,
             4.48148570e+08,   4.61302552e+08,   4.70560389e+08,
             4.80335601e+08,   4.90026337e+08,   5.01425473e+08,
             5.10866939e+08,   5.20055502e+08,   5.29372878e+08,
             5.39908947e+08,   5.48532898e+08,   5.62853475e+08,
             5.75649621e+08,   5.88529024e+08])
    tower_cost1 = np.array([ 11933940.55214095,  11934882.61624805,  11933944.57434152,
            12859531.54305269,  14364943.04281902,  15960191.0113309 ,
            16997855.23023696,  18098368.89996162,  18860758.99676038,
            19548889.18372348,  20153788.35863838,  20131834.46427092,
            20185474.18660865,  20230094.04421138,  20490313.92758453,
            20479571.59012201,  20470163.1894206 ,  20485691.77983882,
            20644446.20259076,  20561722.854106  ,  21205944.15791781,
            21603425.811908  ,  22003070.86381607])
    wake_loss1 = np.array([ 12.00617588,  12.00617588,  12.00617588,  12.00617588,
            12.00617588,  12.00617588,  12.00617588,  12.00617588,
            11.89784607,  11.79378304,  11.66604102,  11.57115985,
            11.44806285,  11.33950761,  11.19195675,  11.0588647 ,
            10.84833328,  10.59747697,  10.34798301,  10.17953928,
            10.00892932,   9.87337296,   9.81402426])



    """2 groups"""
    ideal_AEP2 = np.array([  3.93091892e+08,   3.97612765e+08,   4.02185631e+08,
             4.13410122e+08,   4.28162508e+08,   4.46516631e+08,
             4.61889936e+08,   4.80676837e+08,   4.91314637e+08,
             5.09146682e+08,   5.20151573e+08,   5.32160656e+08,
             5.42248358e+08,   5.54063418e+08,   5.64644213e+08,
             5.74431188e+08,   5.83330075e+08,   5.92247733e+08,
             6.02181934e+08,   6.10932907e+08,   6.26392712e+08,
             6.36973288e+08,   6.51798159e+08])
    AEP2 = np.array([  3.45896588e+08,   3.49874677e+08,   3.53898517e+08,
             3.63966934e+08,   3.77565376e+08,   3.92942408e+08,
             4.06533556e+08,   4.23022240e+08,   4.32859769e+08,
             4.49140296e+08,   4.59373911e+08,   4.70584833e+08,
             4.80160703e+08,   4.91335627e+08,   5.01451249e+08,
             5.10909364e+08,   5.20047149e+08,   5.29505517e+08,
             5.39863442e+08,   5.48759948e+08,   5.63764583e+08,
             5.74031347e+08,   5.87812425e+08])
    tower_cost2 = np.array([ 11933796.08654021,  11934134.09199596,  11933868.75196715,
            12906445.66867079,  14294465.314464  ,  15822862.44751854,
            16891127.17332344,  18312281.6113744 ,  18549629.11785712,
            19697610.59564706,  19869847.54252298,  20135202.0415589 ,
            20166064.26553036,  20411875.4434912 ,  20491286.24470657,
            20484924.0034958 ,  20468771.3397968 ,  20502458.78338046,
            20639747.19526129,  20588043.95390093,  21320066.97724361,
            21403802.48118504,  21922285.55474718])
    wake_loss2 = np.array([ 12.00617588,  12.00617588,  12.00617588,  11.95983978,
            11.81727279,  11.99825926,  11.98475577,  11.99446142,
            11.89764448,  11.78567748,  11.68460604,  11.5709087 ,
            11.45004029,  11.32140998,  11.19164276,  11.05821302,
            10.84856216,  10.59391399,  10.34878135,  10.17672457,
             9.99822125,   9.88140978,   9.81680178])



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

    plt.suptitle('Big Wind Farm: Big Rotor',fontsize=18,y=0.98)
    f.tight_layout(rect=[0, 0.0, 1, 0.95])
    plt.savefig('big_farm_big_rotor.pdf', transparent=True)

    plt.show()
