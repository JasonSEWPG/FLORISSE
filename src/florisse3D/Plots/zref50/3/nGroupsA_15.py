import numpy as np
import matplotlib.pyplot as plt
import matplotlib


COE = np.array([7.451685624567737420e+01,
6.447784251391024668e+01,
6.397843563068124695e+01,
6.394431116868332055e+01,
6.395581617244454264e+01,])
# ,
# ])







font = {'family' : 'normal',
    'weight' : 'normal',
    'size'   : 18}

matplotlib.rc('font', **font)




# nGroups = np.arange(25)+1
nGroups = np.array([1,2,9,18,27])#,36,81])

fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.plot(nGroups, COE,'ob')
ax.set_xticks([2,9,18,27,36,81])
# ax.set_ylim([66,83])
# ax.set_xlim([-5,85])
# ax.text(2,80,'1 height group')
plt.xlabel('Number of Height Groups')
plt.ylabel('COE ($/MWh)')
plt.title('Small Rotor Wind Farm')
plt.tight_layout()
plt.show()
