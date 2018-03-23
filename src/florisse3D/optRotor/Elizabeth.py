import numpy as np
from openmdao.api import Group, Problem, IndepVarComp, pyOptSparseDriver, Component
from FLORISSE3D.setupOptimization import *
from FLORISSE3D.GeneralWindFarmComponents import\
            get_z, getTurbineZ, AEPobj, DeMUX, hGroups, randomStart,\
            getRotorDiameter, getRatedPower, DeMUX
from FLORISSE3D.floris import AEPGroup
import matplotlib.pyplot as plt
import math


"""make your component"""
class grid_func(Component):
    def __init__(self, nTurbs, nRows):
        super(grid_func, self).__init__()
        self.nTurbs = nTurbs
        self.nRows = nRows

        self.add_param('d', 0.0, desc='diameter of turbine blades')
        self.add_param('s', 0.0, desc='spacing coefficient between turbines')

        self.add_output('turbineX', np.zeros(nTurbs), desc='x location of each turbine')
        self.add_output('turbineY', np.zeros(nTurbs), desc='y location of each turbine')

    def solve_nonlinear(self, params, unknowns, resids):
        d = params['d']
        s = params['s']

        nRows = self.nRows
        nTurbs = self.nTurbs
        turbX = np.zeros(nTurbs)
        turbY = np.zeros(nTurbs)

        n_coord = 0
        for i in range(0, nRows):
            for j in range(0, nRows):
                turbY[n_coord] = i * d * s
                turbX[n_coord] = j * d * s
                n_coord += 1

        unknowns['turbineX'] = turbX
        unknowns['turbineY'] = turbY

class changing_spacing(Component):
    def __init__(self, nTurbs, nRows):
        super(changing_spacing, self).__init__()

        self.nTurbs = nTurbs
        self.nRows = nRows

        self.add_param('d', 0.0, desc='diameter of turbine blades')
        self.add_param('s', np.arange(nRows), desc='changing spacing coefficient in the y direction')
        self.add_param('v', np.arange(nRows), desc='changing spacing coefficient in the x direction')

        self.add_output('turbineX', np.zeros(nTurbs), desc='x location of turbine')
        self.add_output('turbineY', np.zeros(nTurbs), desc='y location of turbine')

    def solve_nonlinear(self, params, unknowns, resids):
        d = params['d']
        s = params['s']
        v = params['v']

        nRows = self.nRows
        nTurbs = self.nTurbs
        turbX = np.zeros(nTurbs)
        turbY = np.zeros(nTurbs)

        n_coord = 0
        row_total = 0
        for i in range(0, nRows):
            column_total = 0
            for j in range(0, nRows):
                turbY[n_coord] = d * s[i] + row_total
                turbX[n_coord] = d * v[j] + column_total
                column_total += d*v[j]
                n_coord += 1
            row_total = d * s[i]

        unknowns['turbineX'] = turbX
        unknowns['turbineY'] = turbY

class spline_basic(Component):
    def __init__(self, nTurbs, nRows):
        super(spline_basic, self).__init__()

        self.nTurbs = nTurbs
        self.nRows = nRows

        self.add_param('d', 0.0, desc='diameter of turbine blades')
        self.add_param('s', 0.0, desc='spacing coefficient between turbines')
        self.add_param('A', 0.0, desc='coefficient')
        self.add_param('B', 0.0, desc='coefficient')

        self.add_output('turbineX', np.zeros(nTurbs), desc='x location of turbine')
        self.add_output('turbineY', np.zeros(nTurbs), desc='y location of turbine')

    def solve_nonlinear(self, params, unknowns, resids):

        d = params['d']
        s = params['s']
        A = params['A']
        B = params['B']

        nRows = self.nRows
        nTurbs = self.nTurbs
        turbX = np.zeros(nTurbs)
        turbY = np.zeros(nTurbs)

        n_coord = 0
        for i in range(0, nRows):
            for j in range(0, nRows):
                turbX[n_coord] = d*s*j
                turbY[n_coord] = A * math.sin((B * d * s * j)) + d*s*i
                n_coord += 1

        unknowns['turbineX'] =turbX
        unknowns['turbineY'] = turbY

class spline_changing_spacing(Component):
    def __init__(self, nTurbs, nRows):
        super(spline_changing_spacing, self).__init__()

        self.nTurbs = nTurbs
        self.nRows = nRows

        self.add_param('d', 0.0, desc='diameter of wind turbine')
        self.add_param('s', np.arange(nRows), desc='spacing coefficient in y direction')
        self.add_param('v', np.arange(nRows), desc='spacing coefficient in x direction')
        self.add_param('A', 0.0, desc='coefficient')
        self.add_param('B', 0.0, desc='coefficient')

        self.add_output('turbineX', np.zeros(nTurbs), desc='x location of turbine')
        self.add_output('turbineY', np.zeros(nTurbs), desc='y location of turbine')

    def solve_nonlinear(self, params, unknowns, resids):
        d = params['d']
        s = params['s']
        A = params['A']
        B = params['B']
        v = params['v']

        nRows = self.nRows
        nTurbs = self.nTurbs
        turbX = np.zeros(nTurbs)
        turbY = np.zeros(nTurbs)

        n_coord = 0
        rows = 0
        for i in range(0, nRows):
            cols = 0
            for j in range(0, nRows):
                turbX[n_coord] = d*v[j] + cols
                turbY[n_coord] = A * math.sin(B * d*v[j]) + d*s[i] + rows
                cols += d * v[j]
                n_coord += 1
            rows += d * s[i]


        unknowns['turbineX'] = turbX
        unknowns['turbineY'] = turbY


if __name__ == '__main__':

    nDirections = amaliaWind_23({})
    print nDirections

    """setup the turbine locations"""
    nRows = 5
    nTurbs = nRows**2
    nGroups = 1
    spacing = 4.

    rotor_diameter = 126.4
    turbineX, turbineY = setupGrid(nRows, rotor_diameter, spacing)
    # turbineX, turbineY = amaliaLayout()

    nTurbs = len(turbineX)
    """initial yaw values"""
    yaw = np.zeros((nDirections, nTurbs))


    shearExp = 0.08
    rotorDiameter = np.array([126.4])
    turbineZ = np.array([90.])
    ratedPower = np.array([5000.])

    """OpenMDAO"""
    prob = Problem()
    root = prob.root = Group()

    #Design Variables
    for i in range(nGroups):
        root.add('ratedPower%s'%i, IndepVarComp('ratedPower%s'%i, float(ratedPower[i]), units='kW'), promotes=['*'])
        root.add('turbineH%s'%i, IndepVarComp('turbineH%s'%i, float(turbineZ[i])), promotes=['*'])
        root.add('rotorDiameter%s'%i, IndepVarComp('rotorDiameter%s'%i, float(rotorDiameter[i])), promotes=['*'])

    root.add('Zs', DeMUX(nTurbs)) #have derivatives
    root.add('hGroups', hGroups(nTurbs, nGroups), promotes=['*']) #have derivatives
    root.add('getRotorDiameter', getRotorDiameter(nTurbs, nGroups), promotes=['*']) #have derivatives
    root.add('getRatedPower', getRatedPower(nTurbs, nGroups), promotes=['*'])    #have derivatives

    root.add('AEPGroup', AEPGroup(nTurbs, nDirections), promotes=['*']) #TODO check derivatives?

    root.connect('turbineZ', 'Zs.Array')

    """"add your component"""
    root.add('spline_changing_spacing', spline_changing_spacing(nTurbs, nRows),promotes=['*'])

    prob.setup(check=True)

    amaliaWind_23(prob)
    simpleSetup(nTurbs, prob)

    for i in range(nDirections):
        prob['yaw%s'%i] = yaw[i]
    # prob['turbineX'] = turbineX
    # prob['turbineY'] = turbineY
    prob['shearExp'] = shearExp

    """add your variables"""
    # prob['A'] = 7.
    prob['d'] = 100
    prob['A'] = 1000
    prob['B'] = 20

    prob.run()

    print 'AEP: ', prob['AEP']

    print 'rotor diameter: ', prob['rotorDiameter0']
    print 'turbineX: ', prob['turbineX']
    print 'turbineY: ', prob['turbineY']

    # plt.plot(turbineX,turbineY,'ob',label='start')
    plt.plot(prob['turbineX'],prob['turbineY'],'or',label='opt')
    # for i in range(nTurbs):
        # plt.plot(np.array([turbineX[i],prob['turbineX'][i]]),np.array([turbineY[i],prob['turbineY'][i]]),'--k')
    plt.axis('equal')
    plt.legend(loc=5)
    plt.show()
