import numpy as np
from math import pi, log
import time
from datetime import datetime
from openmdao.api import Group, Component, Problem, ScipyGMRES, IndepVarComp
#, pyOptSparseDriver
from FLORISSE3D.floris import AEPGroup



class transportationCost(Component):
    def __init__(self, nTurbines):

        super(transportationCost, self).__init__()

        self.nTurbines = nTurbines
        self.add_param('transportDist', 0.0, desc='transportation distance')
        self.add_param('cost', 0.0, desc='TCC (whole farm)')

        self.add_output('transportation_cost', 0.0, desc='transportation cost')

    def solve_nonlinear(self, params, unknowns, resids):
        nTurbs = float(self.nTurbines)
        turbineCost = params['cost']/(1543.209877*nTurbs)
        #turbineCost = params['cost']/(5000.*nTurbs)

        transportation_cost = turbineCost*1543.209877*nTurbs
        #transportation_cost = turbineCost*5000.*nTurbs
        transportation_cost += 1867.*params['transportDist']**(0.726)*nTurbs

        unknowns['transportation_cost'] = transportation_cost

    def linearize(self, params, unknowns, resids):
        nTurbs = float(self.nTurbines)

        J = {}
	J['transportation_cost', 'cost'] = 1.
        return J


class powerPerformanceCost(Component):
    def __init__(self, nTurbines):

        super(powerPerformanceCost, self).__init__()

        self.nTurbines = nTurbines
        self.add_param('turbineZ', np.zeros(nTurbines), units='m', desc='the hub heights of each turbine')

        self.add_output('power_performance_cost', 0.0, desc='power performance cost')

    def solve_nonlinear(self, params, unknowns, resids):

        nTurbs = float(self.nTurbines)
        avgHeight = np.sum(params['turbineZ'])/nTurbs
        # avgHeight = 110.

        cost = np.zeros(int(nTurbs))

        for i in range(int(nTurbs)):

            hL = 85.0
            hU = 95.0

            c3 = -114.8
            c2 = 30996.0
            c1 = -2781030.0
            c0 = 83175600.0

            mL1 = 232600.0
            mU1 = 290000.0

            if avgHeight <= hL:
                multiplier1 = mL1
            elif avgHeight >= hU:
                multiplier1 = mU1
            else:
                multiplier1 = c3*avgHeight**3 + c2*avgHeight**2 + c1*avgHeight + c0

            c3 = -48.4
            c2 = 13068.0
            c1 = -1172490.0
            c0 = 35061600.0

            mL2 = 92600.
            mU2 = 116800.

            if avgHeight <= hL:
                multiplier2 = mL2
            elif avgHeight >= hU:
                multiplier2 = mU2
            else:
                multiplier2 = c3*avgHeight**3 + c2*avgHeight**2 + c1*avgHeight + c0

            permanent = 2. #number of permanent met towers
            temporary = 2.

            cost[i] = 200000. + permanent*multiplier1 + temporary*multiplier2

        power_perf_cost = np.sum(cost)/nTurbs

        unknowns['power_performance_cost'] = power_perf_cost

    def linearize(self, params, unknowns, resids):
        nTurbs = self.nTurbines

        J = {}
	J['power_performance_cost', 'turbineZ'] = np.zeros([1,nTurbs])
        return J


class accessRoadCost(Component):
    def __init__(self, nTurbines):

        super(accessRoadCost, self).__init__()

        self.nTurbines = nTurbines
        self.add_param('rotor_diameter', 126.4, desc='rotor diameter')

        self.add_output('access_road_cost', 0.0, desc='access road cost')

    def solve_nonlinear(self, params, unknowns, resids):
        nTurbs = float(self.nTurbines)

        factor1 = 62653.6 #complex layout, flat terrain
        factor2 = 30.9
        diameter = params['rotor_diameter']
        constructionTime = 5.
        accessRoadEntrances = 1.

        #TODO (area)
        roads_cost = (nTurbs*factor1 + nTurbs*diameter*factor2
                   + constructionTime*55500.
                   + accessRoadEntrances*3800.)*1.05

        unknowns['access_road_cost'] = roads_cost


class foundationCost(Component):
    def __init__(self, nTurbines):

        super(foundationCost, self).__init__()

        self.nTurbines = nTurbines
        self.add_param('rotor_diameter', 126.4, desc='rotor diameter')
        self.add_param('turbineZ', np.zeros(nTurbines), units='m', desc='the hub heights of each turbine')

        self.add_output('foundation_cost', 0.0, desc='foundation cost')

    def solve_nonlinear(self, params, unknowns, resids):
        nTurbs = float(self.nTurbines)
        # avgHeight = np.sum(params['turbineZ'])/nTurbs
        turbineZ = params['turbineZ']

        topMass = 24.05066 #??
        #topMass = 88. #??

        #cost = (5000.*params['rotor_diameter']*topMass/1000.0 \
        #        + 163421.5*nTurbs**(-0.1458) + (turbineZ-80.)*500.)
        cost = (1543.209877*params['rotor_diameter']*topMass/1000.0 \
                + 163421.5*nTurbs**(-0.1458) + (turbineZ-80.)*500.)

        foundation_cost = np.sum(cost)

        unknowns['foundation_cost'] = foundation_cost

    def linearize(self, params, unknowns, resids):
        nTurbs = self.nTurbines

        J = {}
	J['foundation_cost', 'turbineZ'] = np.ones([1,nTurbs])*500.
        return J


class erectionCost(Component):
    def __init__(self, nTurbines):

        super(erectionCost, self).__init__()

        self.nTurbines = nTurbines
        self.add_param('turbineZ', np.zeros(nTurbines), units='m', desc='the hub heights of each turbine')

        self.add_output('erection_cost', 0.0, desc='erection cost')

    def solve_nonlinear(self, params, unknowns, resids):
        nTurbs = float(self.nTurbines)
        avgHeight = np.sum(params['turbineZ'])/nTurbs
        turbineZ = params['turbineZ']

        weatherDelayDays = 5.
        craneBreakdowns = 1.

        #cost = (37.*5000. + 27000.*nTurbs**(-0.42145) + (turbineZ-80.)*500.)
        cost = (37.*1543.209877 + 27000.*nTurbs**(-0.42145) + (turbineZ-80.)*500.)

        erection_cost = np.sum(cost)+ 20000.*weatherDelayDays + 35000.*craneBreakdowns + 181.*nTurbs + 1834.

        unknowns['erection_cost'] = erection_cost

    def linearize(self, params, unknowns, resids):
        nTurbs = self.nTurbines

        J = {}
	J['erection_cost', 'turbineZ'] = np.ones([1, nTurbs])*500.
        return J


class electircalMaterialsCost(Component):
    def __init__(self, nTurbines):

        super(electircalMaterialsCost, self).__init__()

        self.nTurbines = nTurbines
        self.add_param('rotor_diameter', 126.4, desc='rotor diameter')

        self.add_output('electrical_materials_cost', 0.0, desc='electrical materials cost')

    def solve_nonlinear(self, params, unknowns, resids):
        nTurbs = float(self.nTurbines)

        factor1 = 67519.4 #complex layout, flat terrain
        factor2 = 27874.4
        factor3 = 681.7
        thermalBackfill = 2. #TODO (what is this)

        elec_mat_cost = nTurbs*factor2

        #TODO (area)
        elec_mat_cost += 5.*35375. + 1.*50000. + params['rotor_diameter']*nTurbs*factor3 + \
                    thermalBackfill*5. + 41945.

        unknowns['electrical_materials_cost'] = elec_mat_cost

class electircalInstallationCost(Component):
    def __init__(self, nTurbines):

        super(electircalInstallationCost, self).__init__()

        self.nTurbines = nTurbines
        self.add_param('rotor_diameter', 126.4, desc='rotor diameter')

        self.add_output('electrical_installation_cost', 0.0, desc='electrical installation cost')

    def solve_nonlinear(self, params, unknowns, resids):
        nTurbs = float(self.nTurbines)

        factor1 = 7683.5
        factor2 = 564.9
        factor3 = 446.0
        rockTrenchingLength = 10.

        elec_instal_cost = 5.*14985.+155000.

        #TODO area
        elec_instal_cost += nTurbs*(factor1 + params['rotor_diameter']*(factor2 + \
                factor3*rockTrenchingLength/100.0)) + 10000.

        unknowns['electrical_installation_cost'] = elec_instal_cost


class insuranceCost(Component):
    def __init__(self, nTurbines):

        super(insuranceCost, self).__init__()

        self.nTurbines = nTurbines
        self.add_param('cost', 0.0, desc='TCC (whole farm)')
        self.add_param('foundation_cost', 0.0, desc='foundation costs')

        self.add_output('insurance_cost', 0.0, desc='insurance cost')
        self.add_output('alpha_insurance', 0.0, desc='alpha insurance')

    def solve_nonlinear(self, params, unknowns, resids):
        nTurbs = float(self.nTurbines)
        turbineCost = params['cost']/(1543.209877*nTurbs)
        #turbineCost = params['cost']/(5000.*nTurbs)

        alpha_insurance = 3.5 + 0.7 + 0.4 + 1.0
        insurance_cost = (0.7 + 0.4 + 1.0) * turbineCost * 37.5

        alpha_insurance /= 1000.0
        insurance_cost += 0.02*params['foundation_cost'] + 20000.

        unknowns['insurance_cost'] = insurance_cost
        unknowns['alpha_insurance'] = alpha_insurance

    def linearize(self, params, unknowns, resids):
        nTurbs = float(self.nTurbines)

        J = {}
	J['insurance_cost', 'cost'] = (0.7+0.4+1.0)*37.5/(1543.209877*nTurbs)
        #J['insurance_cost', 'cost'] = (0.7+0.4+1.0)*37.5/(5000.*nTurbs)
        J['insurance_cost', 'foundation_cost'] = 0.02

        return J


class markupCost(Component):
    def __init__(self, nTurbines):

        super(markupCost, self).__init__()

        self.nTurbines = nTurbines
        self.add_param('transportation_cost', 0.0, desc='transportation costs')

        self.add_output('markup_cost', 0.0, desc='markup cost')
        self.add_output('alpha_markup', 0.0, desc='alpha markup')

    def solve_nonlinear(self, params, unknowns, resids):
        nTurbs = float(self.nTurbines)

        contingency = 3.0
        warranty = 0.02
        useTax = 0.0
        overhead = 5.0
        profitMargin = 5.0

        alpha_markup = (contingency + warranty + useTax + overhead + profitMargin)/100.0
        markup_cost = -alpha_markup * params['transportation_cost']

        unknowns['markup_cost'] = markup_cost
        unknowns['alpha_markup'] = alpha_markup

    def linearize(self, params, unknowns, resids):
        alpha_markup = unknowns['alpha_markup']

        J = {}
	J['markup_cost', 'transportation_cost'] = -alpha_markup

        return J


class BOScalc(Component):
    def __init__(self, nTurbines):

        super(BOScalc, self).__init__()

        self.nTurbines = nTurbines
        self.add_param('transportation_cost', 0.0, desc='transportation cost')
        self.add_param('power_performance_cost', 0.0, desc='power performance cost')
        self.add_param('access_road_cost', 0.0, desc='access road cost')
        self.add_param('foundation_cost', 0.0, desc='foundations cost')
        self.add_param('erection_cost', 0.0, desc='erection cost')
        self.add_param('electrical_materials_cost', 0.0, desc='electrical materials cost')
        self.add_param('electrical_installation_cost', 0.0, desc='electrical installation cost')
        self.add_param('insurance_cost', 0.0, desc='insurance cost')
        self.add_param('markup_cost', 0.0, desc='markup cost')
        self.add_param('alpha_insurance', 0.0, desc='alpha insurance')
        self.add_param('alpha_markup', 0.0, desc='alpha markup')
        self.add_param('cost', 0.0, desc='TCC (whole farm)')

        self.add_output('BOS', 0.0, desc='BOS costs')

    def solve_nonlinear(self, params, unknowns, resids):
        nTurbs = float(self.nTurbines)
        #constant = 16867866.8985 #BOS costs that remain constant
        constant = 18507985.8237 #small rotor 81 turbine farm
        total_cost = constant+params['transportation_cost']+params['power_performance_cost']+\
                    params['access_road_cost']+params['foundation_cost']+params['erection_cost']+\
                    params['electrical_materials_cost']+params['electrical_installation_cost']+\
                    params['insurance_cost']+params['markup_cost']
        self.total_cost = total_cost

        # print 'total: ', total_cost
        # print 'access_road_cost: ', params['access_road_cost']
        # print 'electrical_materials_cost: ', params['electrical_materials_cost']
        # print 'electrical_installation_cost: ', params['electrical_installation_cost']
        # print 'foundation: ', params['foundation_cost']

        alpha = params['alpha_markup'] + params['alpha_insurance']
        self.alpha = alpha

        #multiplier
        total_cost /= (1.0-alpha)

        #remove TCC
        total_cost -= params['cost']

        unknowns['BOS'] = total_cost

    def linearize(self, params, unknowns, resids):
        nTurbs = float(self.nTurbines)
        alpha = self.alpha

        J = {}
	J['BOS', 'transportation_cost'] = 1./(1.-alpha)
        J['BOS', 'power_performance_cost'] = 1./(1.-alpha)
        J['BOS', 'access_road_cost'] = 1./(1.-alpha)
        J['BOS', 'foundation_cost'] = 1./(1.-alpha)
        J['BOS', 'erection_cost'] = 1./(1.-alpha)
        J['BOS', 'electrical_materials_cost'] = 1./(1.-alpha)
        J['BOS', 'electrical_installation_cost'] = 1./(1.-alpha)
        J['BOS', 'insurance_cost'] = 1./(1.-alpha)
        J['BOS', 'markup_cost'] = 1./(1.-alpha)
        # J['BOS', 'alpha_insurance'] =
        # J['BOS', 'alpha_markup'] =
        J['BOS', 'cost'] = -1.
        return J


class BOSgroup(Group):
    """
    Group containing components of BOS
    """
    def __init__(self, nTurbines):

        super(BOSgroup, self).__init__()

        self.add('transportationCost', transportationCost(nTurbines), promotes=['*'])
        self.add('powerPerformanceCost', powerPerformanceCost(nTurbines), promotes=['*'])
        self.add('accessRoadCost', accessRoadCost(nTurbines), promotes=['*'])
        self.add('foundationCost', foundationCost(nTurbines), promotes=['*'])
        self.add('erectionCost', erectionCost(nTurbines), promotes=['*'])
        self.add('electircalMaterialsCost', electircalMaterialsCost(nTurbines), promotes=['*'])
        self.add('electircalInstallationCost', electircalInstallationCost(nTurbines), promotes=['*'])
        self.add('insuranceCost', insuranceCost(nTurbines), promotes=['*'])
        self.add('markupCost', markupCost(nTurbines), promotes=['*'])
        self.add('BOScalc', BOScalc(nTurbines), promotes=['*'])

class farmCost(Component):
    """
    Component to calculate the cost of the wind farm
    """

    def __init__(self, nTurbines, nGroups):

        super(farmCost, self).__init__()

        self.nTurbines = nTurbines
        self.nGroups = nGroups

        # self.add_param('mass1', 0.0, units='kg',
        #                 desc='mass of tower 1')
        # self.add_param('mass2', 0.0, units='kg',
        #                 desc='mass of tower 2')
        # self.add_param('H1_H2', np.zeros(nTurbines),
        #                 desc='array defining which turbines are of H1 and which are H2')
        self.add_param('hGroup', np.zeros(nTurbines), desc='array to define height groups')
        for i in range(nGroups):
            self.add_param('mass%s'%i, 0.0, units='kg',
                        desc='mass of each tower')
        self.add_output('cost', 0.0, desc='Cost of the wind farm')
        self.add_output('tower_cost', 0.0, desc='Cost of the wind farm towers')

    def solve_nonlinear(self, params, unknowns, resids):

        nTurbines = self.nTurbines
        nGroups = self.nGroups
        # mass1 = params['mass1']
        # mass2 = params['mass2']
        # H1_H2 = params['H1_H2']

        mass = np.zeros(nTurbines)
        # for i in range(nTurbines):
        #     if H1_H2[i] == 0:
        #         mass[i] = mass1
        #     elif H1_H2[i] == 1:
        #         mass[i] = mass2
        for i in range(nTurbines):
            for j in range(nGroups):
                if j == params['hGroup'][i]:
                    mass[i] = params['mass%s'%j]
        # print "mass: ", mass

        # Local Variables
        # fixed_charge_rate = 0.095
        # tax_rate = 0.4
        # ppi_mat   = 1.0465528035
        # slope   = 13.0
        # intercept     = 5813.9
        # bos = 559. * 5e3
        # array_losses = 0.059
        # other_losses = 0.0
        # availability = 0.94
        # losses = availability * (1-array_losses) * (1-other_losses)
        # assemblyCostMultiplier = 0.30
        # profitMultiplier = 0.20
        # overheadCostMultiplier = 0.0
        # transportMultiplier = 0.0

        # rotor_cost = 1505102.53 #Number from Ryan
        # nacelle_cost = 3000270. #Number from Ryan
        """these are for the NREL 5 MW reference turbine"""
        #nacelle_cost = 2446465.19*3/4.#turbine_costsse_2015.py run
        #rotor_cost = 1658752.71*3/4. #turbine_costsse_2015.py run
        #nacelle_cost = 1715919.90#*0.2260637636 #nrel_csm_tcc_2015.py run
        #rotor_cost = 1206984.20#*0.2260637636 #nrel_csm_tcc_2015.py run
        #nacelle_cost = 629380.93 #scaled some of the inputs for 1.5MW turbine
        #rotor_cost = 359401.65  #scaled some of the inputs for 1.5MW turbine
        nacelle_cost = 562773.00 #http://www.nrel.gov/docs/fy06osti/32495.pdf page 27 (40 on pdf)
        rotor_cost = 257730.00 #http://www.nrel.gov/docs/fy06osti/32495.pdf
        """"""

        #windpactMassSlope = 0.397251147546925
        #windpactMassInt   = -1414.381881

        # tower_mass_coeff = 19.828
        # tower_mass_exp = 2.0282


        #twrCostEscalator  = 1.5944
        #twrCostCoeff	   = 1.5 # $/kg

        tower_mass_cost_coefficient = 3.08 #$/kg

        tower_cost = np.zeros(nTurbines)
        for i in range(nTurbines):
            #mass = windpactMassSlope * pi * (RotorDiam[i]/2.)**2 * turbineZ[i] + windpactMassInt
            #mass = tower_mass_coeff*turbineZ[i]**tower_mass_exp #new mass from Katherine
            #tower_cost[i] = mass*twrCostEscalator*twrCostCoeff
            # tower_cost = 1390588.80 # to change
            tower_cost[i] = tower_mass_cost_coefficient*mass[i] #new cost from Katherine

        # print 'TOWER COST: ', tower_cost
        parts_cost_farm = nTurbines*(rotor_cost + nacelle_cost) + np.sum(tower_cost) #parts cost for the entire wind farm
        # turbine_multiplier = (1 + transportMultiplier + profitMultiplier) * (1+overheadCostMultiplier+assemblyCostMultiplier)
        turbine_multiplier = 4./3.
        turbine_cost = turbine_multiplier * parts_cost_farm

        # print turbine_cost/nTurbines
        unknowns['tower_cost'] = np.sum(tower_cost)
        unknowns['cost'] = turbine_cost

        # nMass1 = nTurbines-np.sum(H1_H2)
        # nMass2 = np.sum(H1_H2)
        # print "H2_H2: ", H1_H2
        # print "nMass1: ", nMass1
        # print "nMass2: ", nMass2

        # dCost_dMass1 = turbine_multiplier*tower_mass_cost_coefficient*nMass1
        # dCost_dMass2 = turbine_multiplier*tower_mass_cost_coefficient*nMass2
        # self.dCost_dMass1 = dCost_dMass1
        # self.dCost_dMass2 = dCost_dMass2
        dCost_dMass = turbine_multiplier*tower_mass_cost_coefficient
        self.dCost_dMass = dCost_dMass

        # for i in range(nTurbines):
        #     dCost_dTurbineZ[i] = tower_mass_exp*turbine_multiplier*tower_mass_cost_coefficient*tower_mass_coeff*turbineZ[i]**(tower_mass_exp-1)
        #
	# self.dCost_dTurbineZ = dCost_dTurbineZ


    def linearize(self, params, unknowns, resids):
        hGroup = params['hGroup']
        dCost = self.dCost_dMass
        nGroups = self.nGroups
        nTurbines = self.nTurbines

        J = {}
	nEach_Group = np.zeros(nGroups)
        for i in range(nGroups):
            for j in range(nTurbines):
                if hGroup[j] == i:
                    nEach_Group[i] += 1

        for i in range(self.nGroups):
            J['cost', 'mass%s'%i] = dCost*nEach_Group[i]
        # J['cost', 'mass1'] = self.dCost_dMass1
        # J['cost', 'mass2'] = self.dCost_dMass2

        return J


class COEComponent(Component):
    """
    # Componenet to calculate the cost of energy (COE)
    """

    def __init__(self, nTurbines):

        super(COEComponent, self).__init__()

        # self.deriv_options['form'] = 'forward'
        # self.deriv_options['step_size'] = 1.E-6
        # self.deriv_options['step_calc'] = 'relative'

        self.nTurbines = nTurbines
        self.add_param('cost', 0.0, desc='Cost of the wind farm')
        self.add_param('AEP', 0.0, desc='AEP of the wind farm')
        self.add_param('BOS', 0.0, desc='balance of station costs')

        self.add_output('COE', 0.0, desc='Cost of Energy for the wind farm')
        self.add_output('farm_cost', 0.0, desc='Cost of the farm')


    def solve_nonlinear(self, params, unknowns, resids):

        cost = params['cost']
        AEP = params['AEP']
        BOS = params['BOS']

        # bos = 450. * 5.e3 * self.nTurbines #450 $/kW*kW http://www.nrel.gov/docs/fy14osti/61546.pdf (estimation from plots)
        fixed_charge_rate = 0.102 #http://www.nrel.gov/docs/fy15osti/63267.pdf pg 58: 2013 COE
        tax_rate = 0.389 #http://www.nrel.gov/docs/fy15osti/63267.pdf pg 54 tax credit calculation
        O_M_coeff = 0.01 #operating and maintainence cost per kWh

        # print 'top part: ', (fixed_charge_rate*(cost+BOS)+ O_M_coeff*AEP*(1.-tax_rate))
        # unknowns['COE'] = 1000.*(fixed_charge_rate*(cost+bos)+ 0.0122*19566000*(1-tax_rate))/AEP # $/MWh
        unknowns['farm_cost'] = 1000.*(fixed_charge_rate*(cost+BOS)+ O_M_coeff*AEP*(1.-tax_rate))
        unknowns['COE'] = 1000.*(fixed_charge_rate*(cost+BOS)+ O_M_coeff*AEP*(1.-tax_rate))/AEP # $/MWh

    def linearize(self, params, unknowns, resids):

        cost = params['cost']
        AEP = params['AEP']
        BOS = params['BOS']

        fixed_charge_rate = 0.102
        tax_rate = 0.389
        O_M_coeff = 0.01

        J = {}
	J['COE', 'cost'] = 1000.*fixed_charge_rate/AEP
        J['COE', 'AEP'] = -1000.*((BOS+cost)*fixed_charge_rate+AEP*O_M_coeff*(1.-tax_rate))/(AEP**2) + \
                            1000.*O_M_coeff*(1.-tax_rate)/AEP
        J['COS', 'BOS'] = 1000.*fixed_charge_rate/AEP

        return J


class COEGroup(Group):
    """
    Group containing components ot calculate COEGroup
    """
    def __init__(self, nTurbines, nGroups):

        super(COEGroup, self).__init__()

        self.add('farmCost', farmCost(nTurbines, nGroups), promotes=['*'])
        self.add('COEComponent', COEComponent(nTurbines), promotes=['*'])
        self.add('BOSgroup', BOSgroup(nTurbines), promotes=['*'])
