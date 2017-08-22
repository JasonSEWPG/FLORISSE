import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 0
# 0.08/_COE_1_17278878_146.txt
# 0.15/_COE_1_17278878_37.txt
# 0.25/_COE_1_17278878_164.txt

# 1
# 0.08/_COE_1_17278879_101.txt
# 0.15/_COE_1_17278879_202.txt
# 0.25/_COE_1_17278879_546.txt


# 2
# 0.08/_COE_1_17278880_75.txt
# 0.15/_COE_1_17278880_185.txt
# 0.25/_COE_1_17278880_513.txt

font = {'family' : 'normal',
    'weight' : 'normal',
    'size'   : 15}

matplotlib.rc('font', **font)


XY = np.array([6.631423410992911727e+01, 5.917452295747700930e+01, 5.083648233886807333e+01])
XY1 = np.array([6.266570661526196773e+01, 5.815023344376653824e+01, 5.062860581435698037e+01])
XY2 = np.array([6.076994626308957947e+01, 5.695370643708696434e+01, 5.086779906146537655e+01])

shear = np.array([0.08,0.15,0.25])

fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.plot(shear, XY, 'ob', label='90 m')
ax.plot(shear, XY1, 'or', label='1 group')
ax.plot(shear, XY2, 'ok', label='2 group')
ax.axis([0.075,0.305,40,75])

plt.legend()
plt.title('Small Rotor, Layout & Control, 4')
plt.xlabel('Wind Shear Exponent')
plt.ylabel('COE ($/MWh)')
plt.tight_layout()
plt.show()
