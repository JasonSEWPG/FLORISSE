import numpy as np
import os
import matplotlib.pyplot as plt

rotorDiameter = np.arange(40.,180.,2.)
ratedPower = np.arange(500.,10000.,190.)
nRotor = len(rotorDiameter)
nRated = len(ratedPower)
final_array = np.zeros((0,10))

num_feasible = 0
num_possible = 0
num_finished = 0

for i in range(nRated):
    for j in range(nRotor):
        num_possible += 1
        if os.path.exists('results_%s_%s.txt'%(ratedPower[i],rotorDiameter[j])):
            num_finished += 1
            with open('SNOPT_%s_%s.out'%(ratedPower[i],rotorDiameter[j]), 'r') as inF:
                for line in inF:
                    # if 'the problem appears to be infeasible' in line:
                    #     num_infeasible += 1
                    if 'SNOPTC INFO   1 -- optimality conditions satisfied' in line:
                        num_feasible += 1

final_array = np.zeros((num_feasible,10))
feasible = 0

for i in range(nRated):
    for j in range(nRotor):
        if os.path.exists('results_%s_%s.txt'%(ratedPower[i],rotorDiameter[j])):
            with open('SNOPT_%s_%s.out'%(ratedPower[i],rotorDiameter[j]), 'r') as inF:
                for line in inF:
                    # if 'the problem appears to be infeasible' in line:
                        # plt.plot(ratedPower[i],rotorDiameter[j],'or')
                    if 'SNOPTC INFO   1 -- optimality conditions satisfied' in line:
                        filename = 'results_%s_%s.txt'%(ratedPower[i],rotorDiameter[j])
                        openedFile = open(filename)
                        loadedData = np.loadtxt(openedFile)
                        # ratedQ, blade_mass, Vrated, I1, I2, I3, ratedT, extremeT
                        ratedQ = loadedData[0]
                        blade_mass = loadedData[1]
                        Vrated = loadedData[2]
                        I1 = loadedData[3]
                        I2 = loadedData[4]
                        I3 = loadedData[5]
                        ratedT = loadedData[6]
                        extremeT = loadedData[7]
                        results_array = np.array([ratedPower[i],rotorDiameter[j],ratedQ,blade_mass,Vrated,I1,I2,I3,ratedT,extremeT])
                        final_array[feasible] = results_array
                        feasible += 1
                    # plt.plot(ratedPower[i],rotorDiameter[j],'ob')
    print i

plt.show()

print np.shape(final_array)
print final_array[0]
print 'feasible: ', num_feasible
print 'possible: ', num_possible
print 'finished: ', num_finished

np.savetxt('OPTIMIZED.txt', np.c_[final_array], header="ratedPower, rotorDiameter, ratedQ, blade_mass, Vrated, I1, I2, I3, ratedT, extremeT")
