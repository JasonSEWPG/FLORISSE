import numpy as np
from openmdao.api import pyOptSparseDriver, Problem, Group
from FLORISSE3D.GeneralWindFarmComponents import organizeWindSpeeds
from openmdao.api import Problem, Group, IndepVarComp, pyOptSparseDriver, ExecComp, ScipyOptimizer

if __name__=="__main__":

    nTurbines = 3
    speeds = np.array([10.,7.,5.,3.])
    nDirections = len(speeds)

    windSpeeds = np.zeros((nTurbines,nDirections))
    for i in range(nTurbines):
        windSpeeds[i] = speeds

    prob = Problem()
    root = prob.root = Group()

    root.add('organizeWindSpeeds', organizeWindSpeeds(nTurbines, nDirections), promotes=['*'])

    prob.setup()

    prob['windSpeeds'] = windSpeeds

    prob.run()

    print prob['output0']
    print windSpeeds

    middle0 = prob['output0']
    middle1 = prob['output1']

    doutput0_windspeeds = np.zeros((nTurbines,nDirections*nTurbines))
    doutput1_windspeeds = np.zeros((nTurbines,nDirections*nTurbines))

    print doutput0_windspeeds

    step = 1.0E-6

    for i in range(nTurbines):
        for j in range(nDirections):
            nwindSpeeds = np.zeros((nTurbines,nDirections))
            # for l in range(nTurbines):
            nwindSpeeds[i] = speeds
            nwindSpeeds[i][j] += step

            prob = Problem()
            root = prob.root = Group()

            root.add('organizeWindSpeeds', organizeWindSpeeds(nTurbines, nDirections), promotes=['*'])

            prob.setup()

            prob['windSpeeds'] = nwindSpeeds

            prob.run()
            print (prob['output0']-middle0)/step

            doutput0_windspeeds[i][j] = (prob['output0']-middle0)/step
            doutput1_windspeeds[i][j] = (prob['output1']-middle1)/step

    print '0: ', doutput0_windspeeds
    print '1: ', doutput1_windspeeds
