import numpy as np
from math import pi, log
import time
from datetime import datetime
from openmdao.api import Group, Component, Problem, ScipyGMRES, IndepVarComp, pyOptSparseDriver
from florisse.floris import AEPGroup
from commonse.environment import PowerWind, LogWind

class farmCost(Component):
    """
    Component to calculate the cost of the wind farm
    """

    def __init__(self, nTurbines):

        super(farmCost, self).__init__()

        self.nTurbines = nTurbines

        self.add_param('mass1', 0.0, units='kg',
                        desc='mass of tower 1')
        self.add_param('mass2', 0.0, units='kg',
                        desc='mass of tower 2')
        self.add_param('H1_H2', np.zeros(nTurbines),
                        desc='array defining which turbines are of H1 and which are H2')

        self.add_output('cost', 0.0, desc='Cost of the wind farm')

    def solve_nonlinear(self, params, unknowns, resids):

        nTurbines = self.nTurbines
        mass1 = params['mass1']
        mass2 = params['mass2']
        H1_H2 = params['H1_H2']

        mass = np.zeros(nTurbines)
        for i in range(nTurbines):
            if H1_H2[i] == 0:
                mass[i] = mass1
            elif H1_H2[i] == 1:
                mass[i] = mass2

        # Local Variables
        fixed_charge_rate = 0.095
        tax_rate = 0.4
        ppi_mat   = 1.0465528035
        slope   = 13.0
        intercept     = 5813.9
        bos = 559. * 5e3
        array_losses = 0.059
        other_losses = 0.0
        availability = 0.94
        losses = availability * (1-array_losses) * (1-other_losses)
        assemblyCostMultiplier = 0.30
        profitMultiplier = 0.20
        overheadCostMultiplier = 0.0
        transportMultiplier = 0.0

        rotor_cost = 1505102.53
        nacelle_cost = 3000270.

        #windpactMassSlope = 0.397251147546925
        #windpactMassInt   = -1414.381881

        tower_mass_coeff = 19.828
        tower_mass_exp = 2.0282


        #twrCostEscalator  = 1.5944
        #twrCostCoeff      = 1.5 # $/kg

        tower_mass_cost_coefficient = 3.08 #$/kg

        tower_cost = np.zeros(nTurbines)
        for i in range(nTurbines):
            #mass = windpactMassSlope * pi * (RotorDiam[i]/2.)**2 * turbineZ[i] + windpactMassInt
            #mass = tower_mass_coeff*turbineZ[i]**tower_mass_exp #new mass from Katherine
            #tower_cost[i] = mass*twrCostEscalator*twrCostCoeff
            # tower_cost = 1390588.80 # to change
            tower_cost[i] = tower_mass_cost_coefficient*mass[i] #new cost from Katherine


        parts_cost_farm = nTurbines*(rotor_cost + nacelle_cost) + np.sum(tower_cost) #parts cost for the entire wind farm
        turbine_multiplier = (1 + transportMultiplier + profitMultiplier) * (1+overheadCostMultiplier+assemblyCostMultiplier)
        turbine_cost = turbine_multiplier * parts_cost_farm

        unknowns['cost'] = turbine_cost

        nMass1 = nTurbines-np.sum(H1_H2)
        nMass2 = np.sum(H1_H2)

        dCost_dMass1 = turbine_multiplier*tower_mass_cost_coefficient*nMass1
        dCost_dMass2 = turbine_multiplier*tower_mass_cost_coefficient*nMass2
        self.dCost_dMass1 = dCost_dMass1
        self.dCost_dMass2 = dCost_dMass2
        # for i in range(nTurbines):
        #     dCost_dTurbineZ[i] = tower_mass_exp*turbine_multiplier*tower_mass_cost_coefficient*tower_mass_coeff*turbineZ[i]**(tower_mass_exp-1)
        #
        # self.dCost_dTurbineZ = dCost_dTurbineZ


    def linearize(self, params, unknowns, resids):

        J = {}
        J['cost', 'mass1'] = self.dCost_dMass1
        J['cost', 'mass2'] = self.dCost_dMass2

        return J


class COEComponent(Component):
    """
    # Componenet to calculate the cost of energy (COE)
    """

    def __init__(self):

        super(COEComponent, self).__init__()

        self.add_param('cost', 0.0, desc='Cost of the wind farm')
        self.add_param('AEP', 0.0, desc='AEP of the wind farm')

        self.add_output('COE', 0.0, desc='Cost of Energy for the wind farm')


    def solve_nonlinear(self, params, unknowns, resids):

        cost = params['cost']
        AEP = params['AEP']

        bos = 559. * 5e3
        fixed_charge_rate = 0.095
        tax_rate = 0.4

        unknowns['COE'] = (fixed_charge_rate*(cost+bos)+ 0.0122*(1-tax_rate))/AEP

    def linearize(self, params, unknowns, resids):

        cost = params['cost']
        AEP = params['AEP']

        bos = 559. * 5e3
        fixed_charge_rate = 0.095

        J = {}
        J['COE', 'cost'] = fixed_charge_rate/AEP
        J['COE', 'AEP'] = -1*fixed_charge_rate*(cost+bos)/AEP**2

        return J


class COEGroup(Group):
    """
    Group containing components ot calculate COEGroup
    """
    def __init__(self, nTurbines):

        super(COEGroup, self).__init__()

        self.add('farmCost', farmCost(nTurbines), promotes=['*'])
        self.add('COEComponent', COEComponent(), promotes=['*'])

if __name__=="__main__":
    """
    This is just to test during development
    """

    H1 = 150.
    H2 = 75.

    H1_H2 = np.array([0,0,1,0,1,0,1,1,1])

    nTurbines = len(H1_H2)

    prob = Problem()
    root = prob.root = Group()

    root.add('H1', IndepVarComp('turbineH1', H1), promotes=['*'])
    root.add('H2', IndepVarComp('turbineH2', H2), promotes=['*'])
    root.add('getTurbineZ', getTurbineZ(nTurbines), promotes=['*'])
    root.add('farmCost', farmCost(nTurbines), promotes=['*'])

    prob.driver = pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'SNOPT'
    prob.driver.opt_settings['Major iterations limit'] = 1000

    prob.driver.add_objective('cost')

    # --- Design Variables ---
    prob.driver.add_desvar('turbineH1', lower=60., upper=None)
    prob.driver.add_desvar('turbineH2', lower=60., upper=None)

    prob.setup()

    #prob['turbineH1'] = H1
    #prob['turbineH2'] = H2
    prob['H1_H2'] = H1_H2

    prob.run()

    print "Cost: ", prob['cost']
    print 'H1: ', prob['turbineH1']
    print 'H2: ', prob['turbineH2']
