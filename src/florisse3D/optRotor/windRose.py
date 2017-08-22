from windrose import WindroseAxes
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np
from FLORISSE3D.setupOptimization import amaliaWind
from math import radians

ws, wf, wd, nDirections = amaliaWind()
wd += 270.
for i in range(len(wd)):
    wd[i] = radians(wd[i])*-1.
# ax = plt.subplot(111, polar=True)
# # ax.contour(wd,wf, bins=72)
# plt.show()

# N = len(wf)
# bottom = 0
# max_height = max(wf)
#
# width = (2*np.pi) / N
#
# ax = plt.subplot(111, polar=True)
# bars = ax.bar(wd, wf, width=width, bottom=bottom, alpha = 0.25)
# ax.set_xticklabels(['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE'])
#
# ax.set_rgrids([0.01,0.02,0.03], angle=-45.)
# ax.set_yticklabels(['1%','2%','3%'])
# plt.title('Princess Amalia Wind Rose', y=1.07)
#
# # Use custom colors and opacity
# # for r, bar in zip(radii, bars):
# #     bar.set_facecolor(plt.cm.jet(r / 10.))
# #     bar.set_alpha(0.8)
#
# plt.show()

N = 72
bottom = 0
max_height = max(ws)

width = (2*np.pi) / N

ax = plt.subplot(111, polar=True)
bars = ax.bar(wd, ws, width=width, bottom=bottom, alpha = 0.25, color='red')
ax.set_xticklabels(['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE'])

ax.set_rgrids([6.,8.,10.], angle=-32.)
ax.set_yticklabels(['6 m/s','8 m/s','10 m/s'])
plt.title('Princess Amalia Wind Speeds', y=1.07)

# Use custom colors and opacity
# for r, bar in zip(radii, bars):
#     bar.set_facecolor(plt.cm.jet(r / 10.))
#     bar.set_alpha(0.8)

plt.show()
