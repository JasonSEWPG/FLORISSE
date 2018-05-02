from windrose import WindroseAxes
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np
from FLORISSE3D.setupOptimization import amaliaWind
from math import radians

def gaus_1D(wind_direction):
    sigma = 60.
    mu = 270.
    p1 = 1./(2.*np.pi*sigma**2)
    exponent = -(wind_direction-mu)**2/(2.*sigma**2)
    return p1*np.exp(exponent)

# ws, wf, wd, nDirections = amaliaWind(1)
wf = np.array([1.17812570e-02, 1.09958570e-02, 9.60626600e-03, 1.21236860e-02,
                           1.04722450e-02, 1.00695140e-02, 9.68687400e-03, 1.00090550e-02,
                           1.03715390e-02, 1.12172280e-02, 1.52249700e-02, 1.56279300e-02,
                           1.57488780e-02, 1.70577560e-02, 1.93535770e-02, 1.41980570e-02,
                           1.20632100e-02, 1.20229000e-02, 1.32111160e-02, 1.74605400e-02,
                           1.72994400e-02, 1.43993790e-02, 7.87436000e-03, 0.00000000e+00,
                           2.01390000e-05, 0.00000000e+00, 3.42360000e-04, 3.56458900e-03,
                           7.18957000e-03, 8.80068000e-03, 1.13583200e-02, 1.41576700e-02,
                           1.66951900e-02, 1.63125500e-02, 1.31709000e-02, 1.09153300e-02,
                           9.48553000e-03, 1.01097900e-02, 1.18819700e-02, 1.26069900e-02,
                           1.58895900e-02, 1.77021600e-02, 2.04208100e-02, 2.27972500e-02,
                           2.95438600e-02, 3.02891700e-02, 2.69861000e-02, 2.21527500e-02,
                           2.12465500e-02, 1.82861400e-02, 1.66147400e-02, 1.90111800e-02,
                           1.90514500e-02, 1.63932050e-02, 1.76215200e-02, 1.65341460e-02,
                           1.44597600e-02, 1.40370300e-02, 1.65745000e-02, 1.56278200e-02,
                           1.53459200e-02, 1.75210100e-02, 1.59702700e-02, 1.51041500e-02,
                           1.45201100e-02, 1.34527800e-02, 1.47819600e-02, 1.33923300e-02,
                           1.10562900e-02, 1.04521380e-02, 1.16201970e-02, 1.10562700e-02])

ws = np.array([6.53163342, 6.11908394, 6.13415514, 6.0614625,  6.21344602,
                            5.87000793, 5.62161519, 5.96779107, 6.33589422, 6.4668016,
                            7.9854581,  7.6894432,  7.5089221,  7.48638098, 7.65764618,
                            6.82414044, 6.36728201, 5.95982999, 6.05942132, 6.1176321,
                            5.50987893, 4.18461796, 4.82863115, 0.,         0.,         0.,
                            5.94115843, 5.94914252, 5.59386528, 6.42332524, 7.67904937,
                            7.89618066, 8.84560463, 8.51601497, 8.40826823, 7.89479475,
                            7.86194762, 7.9242645,  8.56269962, 8.94563889, 9.82636368,
                           10.11153102, 9.71402212, 9.95233636,  10.35446959, 9.67156182,
                            9.62462527, 8.83545158, 8.18011771, 7.9372492,  7.68726143,
                            7.88134508, 7.31394723, 7.01839896, 6.82858346, 7.06213432,
                            7.01949894, 7.00575122, 7.78735165, 7.52836352, 7.21392201,
                            7.4356621,  7.54099962, 7.61335262, 7.90293531, 7.16021596,
                            7.19617087, 7.5593657,  7.03278586, 6.76105501, 6.48004694,
                            6.94716392])

nDirections = len(wf)
wd = np.linspace(0,360-360/nDirections, nDirections)


wd += 270.
for i in range(len(wd)):
    wd[i] = radians(wd[i])*-1.
# ax = plt.subplot(111, polar=True)
# # ax.contour(wd,wf, bins=72)
# plt.show()

N = len(wf)
bottom = 0
max_height = max(wf)

width = (2*np.pi) / N

ax = plt.subplot(111, polar=True)
bars = ax.bar(wd, ws, width=width, bottom=bottom, color='red',alpha = 0.25)
ax.set_xticklabels(['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE'],fontsize=20,family='serif')

# ax.set_rgrids([0.01,0.02,0.03], angle=-45.)
ax.set_rgrids([6.,8.,10.], angle=-35.)
ax.set_yticklabels([r'6$\frac{m}{s}$',r'8$\frac{m}{s}$',r'10$\frac{m}{s}$'],fontsize=18,family='serif')
plt.title('Princess Amalia Average Wind Speeds', y=1.09,fontsize=20,family='serif')
plt.tight_layout()
plt.savefig('amalia_speeds.pdf',transparent=True)

# Use custom colors and opacity
# for r, bar in zip(radii, bars):
#     bar.set_facecolor(plt.cm.jet(r / 10.))
#     bar.set_alpha(0.8)

plt.show()

# N = 72
# bottom = 0
# max_height = max(ws)
#
# width = (2*np.pi) / N
#
# ax = plt.subplot(111, polar=True)
# bars = ax.bar(wd, ws, width=width, bottom=bottom, alpha = 0.25, color='red')
# ax.set_xticklabels(['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE'])
#
# ax.set_rgrids([6.,8.,10.], angle=-32.)
# ax.set_yticklabels(['6 m/s','8 m/s','10 m/s'])
# plt.title('Princess Amalia Wind Speeds', y=1.07)
#
# # Use custom colors and opacity
# # for r, bar in zip(radii, bars):
# #     bar.set_facecolor(plt.cm.jet(r / 10.))
# #     bar.set_alpha(0.8)
#
# plt.show()
