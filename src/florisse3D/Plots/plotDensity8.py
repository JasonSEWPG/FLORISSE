import numpy as np
import matplotlib.pyplot as plt
import matplotlib

density = np.array([0.024842,0.049684,0.074526,0.099368,0.12421 ,0.149052,0.173894,0.198736,0.223578,0.24842])
font = {'family' : 'normal',
'weight' : 'normal',
'size'   : 18}

matplotlib.rc('font', **font)

"""SMALL"""
"""baseline"""
ideal_AEPb = np.array([  4.10465416e+08,   4.10465416e+08,   4.10465416e+08,
         4.10465416e+08,   4.10465416e+08,   4.10465416e+08,
         4.10465416e+08,   4.10465416e+08,   4.10465416e+08,
         4.10465416e+08])
AEPb = np.array([  3.58207866e+08,   3.16672039e+08,   2.84192609e+08,
         2.57606823e+08,   2.35698687e+08,   2.17356706e+08,
         2.01695721e+08,   1.88228210e+08,   1.76518509e+08,
         1.66256916e+08])
COEb = np.array([  60.6080485 ,   67.75501158,   74.80211702,   81.89174619,
         88.93338603,   95.92439646,  102.89859359,  109.82176629,
        116.70173945,  123.52708511])
costb = np.array([  2.17102797e+10,   2.14561177e+10,   2.12582088e+10,
         2.10958726e+10,   2.09614823e+10,   2.08498108e+10,
         2.07542060e+10,   2.06715545e+10,   2.06000170e+10,
         2.05372322e+10])
tower_costb = np.array([ 33621778.16560055,  33619017.96889926,  33622963.37465718,
        33623714.70958167,  33619834.25433736,  33622739.76675946,
        33623351.42002705,  33620687.01778759,  33620750.78567463,
        33620119.94539715])
wake_lossb = np.array([ 12.73129192,  22.8504943 ,  30.76332417,  37.24030983,
        42.57769887,  47.0462803 ,  50.86170166,  54.14273576,
        56.99552224,  59.49551172])
"""1 group"""
ideal_AEP1 = np.array([  3.80579113e+08,   3.80579113e+08,   3.80578227e+08,
         3.80582640e+08,   3.80576001e+08,   3.80580637e+08,
         3.80580841e+08,   3.80581951e+08,   3.80583046e+08,
         3.80583046e+08])
AEP1 = np.array([  3.32126475e+08,   2.93614905e+08,   2.63499713e+08,
         2.38852486e+08,   2.18535497e+08,   2.01531604e+08,
         1.87010949e+08,   1.74524471e+08,   1.63667751e+08,
         1.54153215e+08])
COE1 = np.array([  56.94419361,   63.61178647,   70.18379606,   76.79684594,
         83.366903  ,   89.88654074,   96.39156774,  102.85105223,
        109.26873656,  115.63582741])
cost1 = np.array([  1.89126743e+10,   1.86773686e+10,   1.84934101e+10,
         1.83431175e+10,   1.82186276e+10,   1.81149787e+10,
         1.80262785e+10,   1.79500255e+10,   1.78837684e+10,
         1.78256346e+10])
tower_cost1 = np.array([ 16146275.48298173,  16146275.48298173,  16146651.31035441,
        16148632.19578885,  16146390.66366807,  16147943.52396904,
        16148086.72960529,  16148317.3898887 ,  16148827.51330551,
        16148827.51330551])
wake_loss1 = np.array([ 12.73129192,  22.8504943 ,  30.76332417,  37.24030983,
        42.57769887,  47.0462803 ,  50.86170166,  54.14273576,
        56.99552224,  59.49551172])
"""2 groups"""
ideal_AEP2 = np.array([  3.74639363e+08,   3.77571735e+08,   3.80371187e+08,
         3.81621223e+08,   3.82138289e+08,   3.82139180e+08,
         3.82139180e+08,   3.82136519e+08,   3.83002841e+08,
         3.83000340e+08])
AEP2 = np.array([  3.35936841e+08,   3.14176753e+08,   3.00147650e+08,
         2.85772314e+08,   2.72251264e+08,   2.59293945e+08,
         2.47728578e+08,   2.37329108e+08,   2.28717285e+08,
         2.20180756e+08])
COE2 = np.array([ 56.67246093,  61.49103679,  65.55478815,  69.21994805,
        72.66343152,  75.99022289,  79.25226982,  82.45392609,
        85.59336265,  88.67140349])
cost2 = np.array([  1.90383675e+10,   1.93190543e+10,   1.96761156e+10,
         1.97811447e+10,   1.97827111e+10,   1.97038047e+10,
         1.96330521e+10,   1.95687167e+10,   1.95766815e+10,
         1.95237367e+10])
tower_cost2 = np.array([ 17059478.13083974,  19882028.76527821,  22923403.52468141,
        24247834.90495812,  24827352.58421196,  24829210.65989877,
        24828566.7220996 ,  24822946.54247962,  25220119.01638192,
        25214546.44079829])
wake_loss2 = np.array([ 10.33060742,  16.79018205,  21.09085544,  25.11624175,
        28.75582678,  32.1467259 ,  35.1732063 ,  37.89415679,
        40.28313625,  42.51160291])


fig = plt.figure(1)
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()
ax1.plot(density,wake_lossb,'ob',label='baseline')
ax1.plot(density,wake_loss1,'or', label='1 group')
ax1.plot(density,wake_loss2,'ok', label='2 groups')
ax1.set_title('0.08 Shear Exponent: Small Rotor', y=1.15)
ax1.set_xlabel('Turbine Density')
ax1.set_ylabel('Wake Loss')
ax1.set_ylim(0.,70.)
ax1.set_xlim(0.02,0.255)
ax1.legend(loc=4)

ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks([0.0099401955,0.039760782,0.0621262219,0.1104466167,0.1590431281])
ax2.set_xticklabels(['10','5','4','3','2.5'])
ax2.set_xlabel('Grid Spacing (D)')
plt.tight_layout()
plt.savefig('small8_wl.pdf', transparent=True)
#

fig = plt.figure(2)
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()
ax1.plot(density,AEPb,'ob',label='baseline')
ax1.plot(density,AEP1,'or', label='1 group')
ax1.plot(density,AEP2,'ok', label='2 groups')
ax1.plot(density,ideal_AEPb,'o',markeredgecolor='blue',markerfacecolor='none')
ax1.plot(density,ideal_AEP1,'o',markeredgecolor='red',markerfacecolor='none')
ax1.plot(density,ideal_AEP2,'o',markeredgecolor='black',markerfacecolor='none')
ax1.set_title('0.08 Shear Exponent: Small Rotor', y=1.15)
ax1.set_xlabel('Turbine Density')
ax1.set_ylabel('AEP')
# ax1.set_ylim(0.,70.)
ax1.set_xlim(0.02,0.255)
ax1.legend(loc=3)

ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks([0.0099401955,0.039760782,0.0621262219,0.1104466167,0.1590431281])
ax2.set_xticklabels(['10','5','4','3','2.5'])
ax2.set_xlabel('Grid Spacing (D)')
plt.tight_layout()
plt.savefig('small8_AEP.pdf', transparent=True)
#
fig = plt.figure(3)
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()
ax1.plot(density,tower_costb,'ob',label='baseline')
ax1.plot(density,tower_cost1,'or', label='1 group')
ax1.plot(density,tower_cost2,'ok', label='2 groups')
ax1.set_title('0.08 Shear Exponent: Small Rotor', y=1.15)
ax1.set_xlabel('Turbine Density')
ax1.set_ylabel('Tower Cost')
# ax1.set_ylim(0.,70.)
ax1.set_xlim(0.02,0.255)
ax1.legend(loc=4)

ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks([0.0099401955,0.039760782,0.0621262219,0.1104466167,0.1590431281])
ax2.set_xticklabels(['10','5','4','3','2.5'])
ax2.set_xlabel('Grid Spacing (D)')
plt.tight_layout()
plt.savefig('small8_tc.pdf', transparent=True)

fig = plt.figure(4)
ax1 = fig.add_subplot(111)
# ax2 = ax1.twiny()
ax1.plot(tower_costb,wake_lossb,'ob',label='baseline')
ax1.plot(tower_cost1,wake_loss1,'or', label='1 group')
ax1.plot(tower_cost2,wake_loss2,'ok', label='2 groups')
ax1.set_title('0.08 Shear Exponent: Small Rotor', y=1.15)
ax1.set_xlabel('Tower Cost')
ax1.set_ylabel('Wake Loss')
ax1.set_ylim(0.,70.)
# ax1.set_xlim(0.02,0.255)
ax1.legend(loc=4)

# ax2.set_xlim(ax1.get_xlim())
# ax2.set_xticks([0.0099401955,0.039760782,0.0621262219,0.1104466167,0.1590431281])
# ax2.set_xticklabels(['10','5','4','3','2.5'])
# ax2.set_xlabel('Grid Spacing (D)')
plt.tight_layout()
plt.savefig('small8_wltc.pdf', transparent=True)


plt.show()
