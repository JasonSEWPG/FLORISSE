import numpy as np
from math import pi, log
from openmdao.api import Group, Component, Problem, ScipyGMRES, IndepVarComp


class getRating(Component):
    """
    find turbine rating from rotor diameter
    """

    def __init__(self, nTurbines):

        super(getRating, self).__init__()

        # self.deriv_options['form'] = 'forward'
        # self.deriv_options['step_size'] = 1.E-6
        # self.deriv_options['step_calc'] = 'relative'

        self.nTurbines = nTurbines
        self.add_param('rotorDiameter', np.zeros(nTurbines), desc='array of rotor diameters')

        self.add_output('ratedPower', np.zeros(nTurbines), units='kW',  desc='rated power array')
        for i in range(nTurbines):
            self.add_output('rated_powers%s'%i, 0.0, units='kW', desc='rated power of each turbine')


    def solve_nonlinear(self, params, unknowns, resids):

        rotorDiameter = params['rotorDiameter']
        ratedPower = np.zeros(self.nTurbines)

        for i in range(self.nTurbines):
            ratedPower[i] = 5000.*rotorDiameter[i]**2/126.4**2
            unknowns['rated_powers%s'%i] = 5000.*rotorDiameter[i]**2/126.4**2

        unknowns['ratedPower'] = ratedPower

    def linearize(self, params, unknowns, resids):

        J = {}
        J['ratedPower', 'rotorDiameter'] = np.zeros((self.nTurbines, self.nTurbines))
        for i in range(nTurbines):
            J['ratedPower', 'rotorDiameter'][i][i] = 2.*5000.*params['rotorDiameter'][i]/126.4**2

        return J


class getMinFreq(Component):
    """
    linear relation for the frequency constraint/rotation rate of the turbine wrt rotor diameter
    """

    def __init__(self):

        super(getMinFreq, self).__init__()
        self.add_param('diameter', 0.0, desc='rotor diameter')
        self.add_output('minFreq', 0.0, desc='frequency constraint')


    def solve_nonlinear(self, params, unknowns, resids):
        unknowns['minFreq'] = -0.002305*params['diameter']+0.4873

    def linearize(self, params, unknowns, resids):
        J = {}
        J['minFreq', 'diameter'] = -0.002305
        return J


class freqConstraint(Component):
    """
    linear relation for the frequency constraint/rotation rate of the turbine wrt rotor diameter
    """

    def __init__(self):

        super(freqConstraint, self).__init__()
        self.add_param('freq', 0.0, desc='frequency')
        self.add_param('minFreq', 0.0, desc='upper bound for freq')
        self.add_output('freqConstraint', 0.0, desc='frequency constraint')


    def solve_nonlinear(self, params, unknowns, resids):
        unknowns['freqConstraint'] = params['freq']-1.1*params['minFreq']

    def linearize(self, params, unknowns, resids):

        J = {}
        J['freqConstraint', 'freq'] = 1.
        J['freqConstraint', 'minFreq'] = -1.1
        return J

class freqConstraintGroup(Group):
    """
    linear relation for the frequency constraint/rotation rate of the turbine wrt rotor diameter
    """

    def __init__(self):

        super(freqConstraintGroup, self).__init__()

        self.add('getMinFreq', getMinFreq(), promotes=['*'])
        self.add('freqConstraint', freqConstraint(), promotes=['*'])


if __name__=="__main__":
    """
    This is just to test during development
    """
