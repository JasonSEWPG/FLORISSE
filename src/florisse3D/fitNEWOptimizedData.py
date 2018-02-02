import numpy as np
from scipy.interpolate import LinearNDInterpolator,SmoothBivariateSpline
from scipy import ndimage
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import math
np.set_printoptions(threshold='nan')

#loading data
filename = 'src/florisse3D/optRotor/BEST_DATA.txt'
opened = open(filename)
data = np.loadtxt(opened)
"ratedPower, rotorDiameter, ratedQ, blade_mass, Vrated, I1, I2, I3, ratedT, extremeT"
ratedPower = data[:,0]
rotorDiameter = data[:,1]
ratedQ = data[:,2]
blade_mass = data[:,3]
print max(blade_mass)
print min(blade_mass)
Vrated = data[:,4]
I1 = data[:,5]
I2 = data[:,6]
I3 = data[:,7]
ratedT = data[:,8]
extremeT = data[:,9]

results_rated_power = np.zeros((20,20))
results_rotor_diameter = np.zeros((20,20))
results_ratedQ = np.zeros((20,20))
results_blade_mass = np.zeros((20,20))
results_Vrated = np.zeros((20,20))
results_I1 = np.zeros((20,20))
results_I2 = np.zeros((20,20))
results_I3 = np.zeros((20,20))
results_ratedT = np.zeros((20,20))
results_extremeT = np.zeros((20,20))

n = 0
# print ratedPower
for i in range(20):
    for j in range(20):
        for k in range(len(ratedPower)):
            results_rated_power[i][j] = float(i*500.+500.)
            results_rotor_diameter[i][j] = float(j*6.+46.)
            if ratedPower[k]==i*500.+500. and rotorDiameter[k]==float(j*6.+46.):
                results_ratedQ[i][j] = ratedQ[n]
                results_blade_mass[i][j] = blade_mass[n]
                results_Vrated[i][j] = Vrated[n]
                results_I1[i][j] = I1[n]
                results_I2[i][j] = I2[n]
                results_I3[i][j] = I3[n]
                results_ratedT[i][j] = ratedT[n]
                results_extremeT[i][j] = extremeT[n]
                n += 1


# for i in range(20):
#     for j in range(20):
#         if results_blade_mass[i][j] != 0:
#             plt.plot(i*500.+500.,j*6.+46.,'ob')
#         else:
#             plt.plot(i*500.+500.,j*6.+46.,'or')
# plt.show()


# ratedQ_SMOOTH = np.zeros(len(rotorDiameter))
# blade_mass_SMOOTH = np.zeros(len(rotorDiameter))
# Vrated_SMOOTH = np.zeros(len(rotorDiameter))
# I1_SMOOTH = np.zeros(len(rotorDiameter))
# I2_SMOOTH = np.zeros(len(rotorDiameter))
# I3_SMOOTH = np.zeros(len(rotorDiameter))
# ratedT_SMOOTH = np.zeros(len(rotorDiameter))
# extremeT_SMOOTH = np.zeros(len(rotorDiameter))
#
# hor = np.array([-1,0,1])
# ver = np.array([-1,0,1])
# n = 0
# r = 0
#
# for i in range(20):
#     for j in range(20):
#         smooth_ratedQ = results_ratedQ[i][j]
#         smooth_blade_mass = results_blade_mass[i][j]
#         smooth_Vrated = results_Vrated[i][j]
#         smooth_I1 = results_I1[i][j]
#         smooth_I2 = results_I2[i][j]
#         smooth_I3 = results_I3[i][j]
#         smooth_ratedT = results_ratedT[i][j]
#         smooth_extremeT = results_extremeT[i][j]
#         if results_ratedQ[i][j]==0.:
#             print 'no'
#             # plt.plot(i*190.+500.,j*2.+40.,'or')
#         else:
#             # plt.plot(i*190.+500.,j*2.+40.,'ob')
#             r = 0
#             for k in range(3):
#                 for l in range(3):
#                     print i
#                     print j
#                     if i > 0 and i < 19 and j > 0 and j < 19:
#                         if results_ratedQ[i+hor[k]][j+ver[l]] != 0.:
#                             r += 1
#                             smooth_ratedQ += results_ratedQ[i+hor[k]][j+ver[l]]
#                             smooth_blade_mass += results_blade_mass[i+hor[k]][j+ver[l]]
#                             smooth_Vrated += results_Vrated[i+hor[k]][j+ver[l]]
#                             smooth_I1 += results_I1[i+hor[k]][j+ver[l]]
#                             smooth_I2 += results_I2[i+hor[k]][j+ver[l]]
#                             smooth_I3 += results_I3[i+hor[k]][j+ver[l]]
#                             smooth_ratedT += results_ratedT[i+hor[k]][j+ver[l]]
#                             smooth_extremeT += results_extremeT[i+hor[k]][j+ver[l]]
#             if r != 0.:
#                 ratedQ_SMOOTH[n] = smooth_ratedQ/float(r)
#                 blade_mass_SMOOTH[n] = smooth_blade_mass/float(r)
#                 Vrated_SMOOTH[n] = smooth_Vrated/float(r)
#                 I1_SMOOTH[n] = smooth_I1/float(r)
#                 I2_SMOOTH[n] = smooth_I2/float(r)
#                 I3_SMOOTH[n] = smooth_I3/float(r)
#                 ratedT_SMOOTH[n] = smooth_ratedT/float(r)
#                 extremeT_SMOOTH[n] = smooth_extremeT/float(r)
#                 n += 1
#



cartcoord = list(zip(ratedPower,rotorDiameter))
w = np.ones(len(ratedPower))*2.
#
# interp_spline_ratedQ = SmoothBivariateSpline(ratedPower,rotorDiameter,ratedQ,w)
# interp_spline_smooth_ratedQ = SmoothBivariateSpline(ratedPower,rotorDiameter,ratedQ_SMOOTH,w)
#
order = 2

# print blade_mass_SMOOTH
interp_spline_blade_mass = SmoothBivariateSpline(ratedPower,rotorDiameter,blade_mass,w,kx=order,ky=order)
# interp_spline_blade_mass = LinearNDInterpolator(cartcoord,blade_mass)
# interp_spline_smooth_blade_mass = SmoothBivariateSpline(ratedPower,rotorDiameter,blade_mass_SMOOTH,w,kx=order,ky=order)
#
# interp_spline_Vrated = SmoothBivariateSpline(ratedPower,rotorDiameter,Vrated,w)
# interp_spline_smooth_Vrated = SmoothBivariateSpline(ratedPower,rotorDiameter,Vrated_SMOOTH,w)
#
# interp_spline_I1 = SmoothBivariateSpline(ratedPower,rotorDiameter,I1,w)
# interp_spline_smooth_I1 = SmoothBivariateSpline(ratedPower,rotorDiameter,I1_SMOOTH,w)
#
# interp_spline_I2 = SmoothBivariateSpline(ratedPower,rotorDiameter,I2,w)
# interp_spline_smooth_I2 = SmoothBivariateSpline(ratedPower,rotorDiameter,I2_SMOOTH,w)
#
# interp_spline_I3 = SmoothBivariateSpline(ratedPower,rotorDiameter,I3,w)
# interp_spline_smooth_I3 = SmoothBivariateSpline(ratedPower,rotorDiameter,I3_SMOOTH,w)
#
# interp_spline_ratedT = SmoothBivariateSpline(ratedPower,rotorDiameter,ratedT,w)
# interp_spline_smooth_ratedT = SmoothBivariateSpline(ratedPower,rotorDiameter,ratedT_SMOOTH,w)
#
# interp_spline_extremeT = SmoothBivariateSpline(ratedPower,rotorDiameter,extremeT,w)
# interp_spline_smooth_extremeT = SmoothBivariateSpline(ratedPower,rotorDiameter,extremeT_SMOOTH,w)
#
#
#
num = 100
x = np.linspace(500.,10000.,num)
y = np.linspace(50.,160.,num)
X,Y = np.meshgrid(x,y)
#
# Z_ratedQ = np.zeros((num,num))
# Z_smooth_ratedQ = np.zeros((num,num))
#
Z_blade_mass = np.zeros((num,num))
Z_smooth_blade_mass = np.zeros((num,num))
#
# Z_Vrated = np.zeros((num,num))
# Z_smooth_Vrated = np.zeros((num,num))
#
# Z_I1 = np.zeros((num,num))
# Z_smooth_I1 = np.zeros((num,num))
#
# Z_I2 = np.zeros((num,num))
# Z_smooth_I2 = np.zeros((num,num))
#
# Z_I3 = np.zeros((num,num))
# Z_smooth_I3 = np.zeros((num,num))
#
# Z_ratedT = np.zeros((num,num))
# Z_smooth_ratedT = np.zeros((num,num))
#
# Z_extremeT = np.zeros((num,num))
# Z_smooth_extremeT = np.zeros((num,num))
#
for i in range(num):
    for j in range(num):
#         Z_ratedQ[i][j] = interp_spline_ratedQ(x[i],y[j])
#         Z_smooth_ratedQ[i][j] = interp_spline_smooth_ratedQ(x[i],y[j])
#
        Z_blade_mass[j][i] = interp_spline_blade_mass(x[i],y[j])
        # Z_smooth_blade_mass[j][i] = interp_spline_smooth_blade_mass(x[i],y[j])
#
#         Z_Vrated[i][j] = interp_spline_Vrated(x[i],y[j])
#         Z_smooth_Vrated[i][j] = interp_spline_smooth_Vrated(x[i],y[j])
#
#         Z_I1[i][j] = interp_spline_I1(x[i],y[j])
#         Z_smooth_I1[i][j] = interp_spline_smooth_I1(x[i],y[j])
#
#         Z_I2[i][j] = interp_spline_I2(x[i],y[j])
#         Z_smooth_I2[i][j] = interp_spline_smooth_I2(x[i],y[j])
#
#         Z_I3[i][j] = interp_spline_I3(x[i],y[j])
#         Z_smooth_I3[i][j] = interp_spline_smooth_I3(x[i],y[j])
#
#         Z_ratedT[i][j] = interp_spline_ratedT(x[i],y[j])
#         Z_smooth_ratedT[i][j] = interp_spline_smooth_ratedT(x[i],y[j])
#
#         Z_extremeT[i][j] = interp_spline_extremeT(x[i],y[j])
#         Z_smooth_extremeT[i][j] = interp_spline_smooth_extremeT(x[i],y[j])
#
#
for i in range(num):
    for j in range(num):
#         if math.isnan(Z_Vrated[i][j])==True:
#             Z_Vrated[i][j] = 10.7937998964
#
#         if Z_smooth_Vrated[i][j] < 5.:
#             Z_smooth_Vrated[i][j] = 5.
#
#         if Z_ratedQ[i][j] < 100000.:
#             Z_ratedQ[i][j] = 100000.
#
#         if Z_ratedQ[i][j] > 2000000.:
#             Z_ratedQ[i][j] = 2000000.
#
#         if Z_smooth_ratedQ[i][j] < 100000.:
#             Z_smooth_ratedQ[i][j] = 100000.
#
#         if Z_smooth_ratedQ[i][j] > 2000000.:
#             Z_smooth_ratedQ[i][j] = 2000000.
#
        if Z_blade_mass[i][j] > 26426.7546363:
            Z_blade_mass[i][j] = 26426.7546363
#
#         if Z_blade_mass[i][j] < 100.:
#             Z_blade_mass[i][j] = 100.
#
#         if Z_smooth_blade_mass[i][j] > 26426.7546363:
#             Z_smooth_blade_mass[i][j] = 26426.7546363
# #
#         if Z_smooth_blade_mass[i][j] < 1790.68473726:
#             Z_smooth_blade_mass[i][j] = 1790.68473726
#
#         if Z_I1[i][j] < 100.:
#             Z_I1[i][j] = 100.
#
#         if Z_smooth_I1[i][j] < 100.:
#             Z_smooth_I1[i][j] = 100.
#
#         if Z_I2[i][j] < 100.:
#             Z_I2[i][j] = 100.
#
#         if Z_smooth_I2[i][j] < 100.:
#             Z_smooth_I2[i][j] = 100.
#
#         if Z_I3[i][j] < 100.:
#             Z_I3[i][j] = 100.
#
#         if Z_smooth_I3[i][j] < 100.:
#             Z_smooth_I3[i][j] = 100.
#
#         if Z_ratedT[i][j] < 100.:
#             Z_ratedT[i][j] = 100.
#
#         if Z_smooth_ratedT[i][j] < 100.:
#             Z_smooth_ratedT[i][j] = 100.
#
#         if Z_extremeT[i][j] < 100.:
#             Z_extremeT[i][j] = 100.
#
#         if Z_smooth_extremeT[i][j] < 100.:
#             Z_smooth_extremeT[i][j] = 100.
#
#         if Z_extremeT[i][j] > 200000.:
#             Z_extremeT[i][j] = 200000.
#
#         if Z_smooth_extremeT[i][j] > 200000.:
#             Z_smooth_extremeT[i][j] = 200000.
#
# for i in range(num):
#     for j in range(num):
#         print Z_smooth_blade_mass[i][j]
#
# fig, ax = plt.subplots(nrows=1, ncols=1, subplot_kw={'projection': '3d'})
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, Z_smooth_blade_mass)
ax.plot_surface(X, Y, Z_blade_mass)
ax.plot(ratedPower, rotorDiameter, blade_mass, 'or')
# ax.plot(ratedPower, rotorDiameter, blade_mass_SMOOTH, 'oy')

k = 5
m = 7

print 'Rated Power: ', results_rated_power[k][m]
print 'Rotor Diameter: ', results_rotor_diameter[k][m]
print 'DATA: ', results_blade_mass[k][m]
print 'FIT: ', interp_spline_blade_mass(results_rated_power[k][m],results_rotor_diameter[k][m])
# print 'SMOOTH FIT: ', interp_spline_smooth_blade_mass(results_rated_power[k][m],results_rotor_diameter[k][m])

# print 'max, max: ', interp_spline_smooth_blade_mass(10000.,160.), results_blade_mass[19][19]

#
# ax[1].plot_wireframe(X, Y, Z_smooth_blade_mass, color='k')
#
# # for axes in ax:
# #     # axes.set_zlim(-0.2,1)
# #     axes.set_axis_off()
#
fig.tight_layout()
plt.xlabel('Turbine Rating')
plt.ylabel('Rotor Diameter')

#
# #
# # fig, ax = plt.subplots(nrows=1, ncols=2, subplot_kw={'projection': '3d'})
# # ax[0].plot_wireframe(X, Y, Z_Vrated, color='k')
# #
# # ax[1].plot_wireframe(X, Y, Z_smooth_Vrated, color='k')
# #
# # # for axes in ax:
# # #     # axes.set_zlim(-0.2,1)
# # #     axes.set_axis_off()
# #
# # fig.tight_layout()
# # plt.xlabel('Turbine Rating')
# # plt.ylabel('Rotor Diameter')
#
plt.show()
