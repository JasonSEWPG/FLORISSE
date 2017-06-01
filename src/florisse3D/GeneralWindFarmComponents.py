from openmdao.api import Component, Group, Problem, IndepVarComp
from akima import Akima, akima_interp
from utilities import smooth_min, hermite_spline
import config

import numpy as np
from scipy import interp
from scipy.io import loadmat
from scipy.spatial import ConvexHull
import random

def add_gen_params_IdepVarComps(openmdao_group, datasize):
    openmdao_group.add('gp0', IndepVarComp('gen_params:pP', 1.88, pass_by_obj=True), promotes=['*'])
    openmdao_group.add('gp1', IndepVarComp('gen_params:windSpeedToCPCT_wind_speed', np.zeros(datasize), units='m/s',
                                  desc='range of wind speeds', pass_by_obj=True), promotes=['*'])
    openmdao_group.add('gp2', IndepVarComp('gen_params:windSpeedToCPCT_CP', np.zeros(datasize),
                                  desc='power coefficients', pass_by_obj=True), promotes=['*'])
    openmdao_group.add('gp3', IndepVarComp('gen_params:windSpeedToCPCT_CT', np.zeros(datasize),
                                  desc='thrust coefficients', pass_by_obj=True), promotes=['*'])
    openmdao_group.add('gp4', IndepVarComp('gen_params:CPcorrected', False,
                                  pass_by_obj=True), promotes=['*'])
    openmdao_group.add('gp5', IndepVarComp('gen_params:CTcorrected', False,
                                  pass_by_obj=True), promotes=['*'])


class WindFrame(Component):
    """ Calculates the locations of each turbine in the wind direction reference frame """

    def __init__(self, nTurbines, resolution=0, differentiable=True, nSamples=0):

        # print 'entering windframe __init__ - analytic'

        super(WindFrame, self).__init__()

        # set finite difference options (fd used for testing only)
        # self.deriv_options['form'] = 'central'
        # self.deriv_options['step_size'] = 500.
        # self.deriv_options['step_calc'] = 'relative'

        # if not differentiable:
        #     self.deriv_options['type'] = 'fd'
        #     self.deriv_options['form'] = 'forward'

        self.nTurbines = nTurbines
        self.nSamples = nSamples

        # flow property variables
        # self.add_param('wind_speed', np.ones(nTurbines)*8.0, units='m/s', desc='free stream wind velocity') #TODO change to an array
        self.add_param('wind_direction', val=270.0, units='deg',
                       desc='wind direction using direction from, in deg. cw from north as in meteorological data')

        # Explicitly size input arrays
        self.add_param('turbineX', val=np.zeros(nTurbines), units='m', desc='x positions of turbines in original ref. frame')
        self.add_param('turbineY', val=np.zeros(nTurbines), units='m', desc='y positions of turbines in original ref. frame')

        # add output
        self.add_output('turbineXw', val=np.zeros(nTurbines), units='m', desc='downwind coordinates of turbines')
        self.add_output('turbineYw', val=np.zeros(nTurbines), units='m', desc='crosswind coordinates of turbines')

        # ############################ visualization arrays ##################################
        if nSamples > 0:
            # visualization input
            self.add_param('wsPositionX', np.zeros(nSamples), units='m', pass_by_object=True,
                           desc='X position of desired measurements in original ref. frame')
            self.add_param('wsPositionY', np.zeros(nSamples), units='m', pass_by_object=True,
                           desc='Y position of desired measurements in original ref. frame')
            self.add_param('wPositionZ', np.zeros(nSamples), units='m', pass_by_object=True,
                           desc='Z position of desired measurements in original ref. frame')

            # visualization output
            self.add_output('wsPositionXw', np.zeros(nSamples), units='m', pass_by_object=True,
                            desc='position of desired measurements in wind ref. frame')
            self.add_output('wsPositionYw', np.zeros(nSamples), units='m', pass_by_object=True,
                            desc='position of desired measurements in wind ref. frame')

    def solve_nonlinear(self, params, unknowns, resids):

        windDirectionDeg = params['wind_direction']

        # get turbine positions and velocity sampling positions
        turbineX = params['turbineX']
        turbineY = params['turbineY']

        if self.nSamples > 0:
            velX = params['wsPositionX']
            velY = params['wsPositionY']

        # adjust directions
        windDirectionDeg = 270. - windDirectionDeg
        if windDirectionDeg < 0.:
            windDirectionDeg += 360.
        windDirectionRad = np.pi*windDirectionDeg/180.0    # inflow wind direction in radians

        # convert to downwind(x)-crosswind(y) coordinates
        unknowns['turbineXw'] = turbineX*np.cos(-windDirectionRad)-turbineY*np.sin(-windDirectionRad)
        unknowns['turbineYw'] = turbineX*np.sin(-windDirectionRad)+turbineY*np.cos(-windDirectionRad)

        if self.nSamples > 0:
            unknowns['wsPositionXw'] = velX*np.cos(-windDirectionRad)-velY*np.sin(-windDirectionRad)
            unknowns['wsPositionYw'] = velX*np.sin(-windDirectionRad)+velY*np.cos(-windDirectionRad)

    def linearize(self, params, unknowns, resids):

        # obtain necessary inputs
        nTurbines = self.nTurbines
        windDirectionDeg = params['wind_direction']

        # convert from meteorological polar system (CW, 0 deg.=N) to standard polar system (CCW, 0 deg.=E)
        windDirectionDeg = 270. - windDirectionDeg
        if windDirectionDeg < 0.:
            windDirectionDeg += 360.

        # convert inflow wind direction to radians
        windDirectionRad = np.pi*windDirectionDeg/180.0

        # calculate gradients of conversion to wind direction reference frame
        dturbineXw_dturbineX = np.eye(nTurbines, nTurbines)*np.cos(-windDirectionRad)
        dturbineXw_dturbineY = np.eye(nTurbines, nTurbines)*(-np.sin(-windDirectionRad))
        dturbineYw_dturbineX = np.eye(nTurbines, nTurbines)*np.sin(-windDirectionRad)
        dturbineYw_dturbineY = np.eye(nTurbines, nTurbines)*np.cos(-windDirectionRad)

        # initialize Jacobian dict
        J = {}

        # populate Jacobian dict
        J[('turbineXw', 'turbineX')] = dturbineXw_dturbineX
        J[('turbineXw', 'turbineY')] = dturbineXw_dturbineY
        J[('turbineYw', 'turbineX')] = dturbineYw_dturbineX
        J[('turbineYw', 'turbineY')] = dturbineYw_dturbineY

        return J


class AdjustCtCpYaw(Component):
    """ Adjust Cp and Ct to yaw if they are not already adjusted """

    def __init__(self, nTurbines, direction_id=0, differentiable=True):

        # print 'entering adjustCtCp __init__ - analytic'
        super(AdjustCtCpYaw, self).__init__()

        self. direction_id = direction_id

        # set finite difference options (fd used for testing only)
        # self.deriv_options['form'] = 'central'
        # self.deriv_options['step_size'] = 500.
        # self.deriv_options['step_calc'] = 'relative'
        #
        # if not differentiable:
        #     self.deriv_options['type'] = 'fd'
        #     self.deriv_options['form'] = 'forward'

        # Explicitly size input arrays
        self.add_param('Ct_in', val=np.zeros(nTurbines), desc='Thrust coefficient for all turbines')
        self.add_param('Cp_in', val=np.zeros(nTurbines)+(0.7737/0.944) * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2),
                       desc='power coefficient for all turbines')
        self.add_param('yaw%i' % direction_id, val=np.zeros(nTurbines), units='deg', desc='yaw of each turbine')

        # Explicitly size output arrays
        self.add_output('Ct_out', val=np.zeros(nTurbines), desc='Thrust coefficient for all turbines')
        self.add_output('Cp_out', val=np.zeros(nTurbines), desc='power coefficient for all turbines')

        # parameters since var trees are not supports
        self.add_param('gen_params:pP', 1.88, pass_by_obj=True)
        self.add_param('gen_params:CTcorrected', False,
                       desc='CT factor already corrected by CCBlade calculation (approximately factor cos(yaw)^2)', pass_by_obj=True)
        self.add_param('gen_params:CPcorrected', False,
                       desc='CP factor already corrected by CCBlade calculation (assumed with approximately factor cos(yaw)^3)', pass_by_obj=True)
        # self.add_param('floris_params:FLORISoriginal', True,
        #                desc='override all parameters and use FLORIS as original in first Wind Energy paper', pass_by_obj=True)

    def solve_nonlinear(self, params, unknowns, resids):

        direction_id = self.direction_id

        # print 'entering adjustCtCP - analytic'

        # collect inputs
        Ct = params['Ct_in']
        Cp = params['Cp_in']
        yaw = params['yaw%i' % direction_id] * np.pi / 180.
        # print 'in Ct correction, Ct_in: ', Ct

        pP = params['gen_params:pP']

        CTcorrected = params['gen_params:CTcorrected']
        CPcorrected = params['gen_params:CPcorrected']

        # calculate new CT values, if desired
        if not CTcorrected:
            # print "ct not corrected"
            unknowns['Ct_out'] = np.cos(yaw)*np.cos(yaw)*Ct
            # print 'in ct correction Ct_out: ', unknowns['Ct_out']
        else:
            unknowns['Ct_out'] = Ct

        # calculate new CP values, if desired
        if not CPcorrected:
            unknowns['Cp_out'] = Cp * np.cos(yaw) ** pP
        else:
            unknowns['Cp_out'] = Cp

    def linearize(self, params, unknowns, resids):

        direction_id = self.direction_id

        # collect inputs
        Ct = params['Ct_in']
        Cp = params['Cp_in']
        nTurbines = np.size(Ct)
        yaw = params['yaw%i' % direction_id] * np.pi / 180.

        pP = params['gen_params:pP']

        CTcorrected = params['gen_params:CTcorrected']
        CPcorrected = params['gen_params:CPcorrected']

        # initialize Jacobian dict
        J = {}

        # calculate gradients and populate Jacobian dict
        if not CTcorrected:
            J[('Ct_out', 'Ct_in')] = np.eye(nTurbines) * np.cos(yaw) * np.cos(yaw)
            J[('Ct_out', 'Cp_in')] = np.zeros((nTurbines, nTurbines))
            J[('Ct_out', 'yaw%i' % direction_id)] = np.eye(nTurbines) * Ct * (
                -2. * np.sin(yaw) * np.cos(yaw)) * np.pi / 180.
        else:
            J[('Ct_out', 'Ct_in')] = np.eye(nTurbines, nTurbines)
            J[('Ct_out', 'Cp_in')] = np.zeros((nTurbines, nTurbines))
            J[('Ct_out', 'yaw%i' % direction_id)] = np.zeros((nTurbines, nTurbines))

        if not CPcorrected:
            J[('Cp_out', 'Cp_in')] = np.eye(nTurbines, nTurbines) * np.cos(yaw) ** pP
            J[('Cp_out', 'Ct_in')] = np.zeros((nTurbines, nTurbines))
            J[('Cp_out', 'yaw%i' % direction_id)] = np.eye(nTurbines, nTurbines) * (
                -Cp * pP * np.sin(yaw) * np.cos(yaw) ** (pP - 1.0)) * np.pi / 180.
        else:
            J[('Cp_out', 'Cp_in')] = np.eye(nTurbines, nTurbines)
            J[('Cp_out', 'Ct_in')] = np.zeros((nTurbines, nTurbines))
            J[('Cp_out', 'yaw%i' % direction_id)] = np.zeros((nTurbines, nTurbines))

        return J


class WindFarmAEP(Component):
    """ Estimate the AEP based on power production for each direction and weighted by wind direction frequency  """

    def __init__(self, nDirections, rec_func_calls=True):

        super(WindFarmAEP, self).__init__()

        # # set finite difference options (fd used for testing only)
        # self.deriv_options['form'] = 'central'
        # self.deriv_options['step_size'] = 500.
        # self.deriv_options['step_calc'] = 'relative'

        # define inputs
        self.add_param('dirPowers', np.zeros(nDirections), units='kW',
                       desc='vector containing the power production at each wind direction ccw from north')
        self.add_param('windFrequencies', np.zeros(nDirections),
                       desc='vector containing the weighted frequency of wind at each direction ccw from east using '
                            'direction too')

        # define output
        self.add_output('AEP', val=0.0, units='kWh', desc='total annual energy output of wind farm')

        # pass bool for function call recording
        self.rec_func_calls = rec_func_calls

    def solve_nonlinear(self, params, unknowns, resids):

        # locally name input values
        dirPowers = params['dirPowers']
        windFrequencies = params['windFrequencies']

        # number of hours in a year
        hours = 8760.0

        # calculate approximate AEP
        AEP = sum(dirPowers*windFrequencies)*hours

        # promote AEP result to class attribute
        unknowns['AEP'] = AEP

        # print AEP

        # increase objective function call count
        if self.rec_func_calls:
            config.obj_func_calls += 1

    def linearize(self, params, unknowns, resids):

        # # print 'entering AEP - provideJ'

        # assign params to local variables
        windFrequencies = params['windFrequencies']
        # dirPowers = params['dirPowers']
        nDirs = np.size(windFrequencies)

        # number of hours in a year
        hours = 8760.0

        # calculate the derivative of outputs w.r.t. the power in each wind direction
        dAEP_dpower = np.ones(nDirs)*windFrequencies*hours

        # initialize Jacobian dict
        J = {}

        # populate Jacobian dict
        J['AEP', 'dirPowers'] = np.array([dAEP_dpower])

        # increase gradient function call count
        if self.rec_func_calls:
            config.sens_func_calls += 1

        return J


class SpacingComp(Component):
    """
    Calculates inter-turbine spacing for all turbine pairs
    """

    def __init__(self, nTurbines):

        super(SpacingComp, self).__init__()

        # set finite difference options (fd used for testing only)
        # self.deriv_options['form'] = 'central'
        # self.deriv_options['step_size'] = 500.
        # self.deriv_options['step_calc'] = 'relative'

        # Explicitly size input arrays
        self.add_param('turbineX', val=np.zeros(nTurbines),
                       desc='x coordinates of turbines in wind dir. ref. frame')
        self.add_param('turbineY', val=np.zeros(nTurbines),
                       desc='y coordinates of turbines in wind dir. ref. frame')

        # Explicitly size output array
        self.add_output('wtSeparationSquared', val=np.zeros((nTurbines-1.)*nTurbines/2.),
                        desc='spacing of all turbines in the wind farm')

    def solve_nonlinear(self, params, unknowns, resids):
        # print 'in dist const'

        turbineX = params['turbineX']
        turbineY = params['turbineY']
        nTurbines = turbineX.size
        separation_squared = np.zeros((nTurbines-1.)*nTurbines/2.)

        k = 0
        for i in range(0, nTurbines):
            for j in range(i+1, nTurbines):
                separation_squared[k] = (turbineX[j]-turbineX[i])**2+(turbineY[j]-turbineY[i])**2
                k += 1
        unknowns['wtSeparationSquared'] = separation_squared

    def linearize(self, params, unknowns, resids):

        # obtain necessary inputs
        turbineX = params['turbineX']
        turbineY = params['turbineY']

        # get number of turbines
        nTurbines = turbineX.size

        # initialize gradient calculation array
        dS = np.zeros(((nTurbines-1.)*nTurbines/2., 2*nTurbines))

        # set turbine pair counter to zero
        k = 0

        # calculate the gradient of the distance between each pair of turbines w.r.t. turbineX and turbineY
        for i in range(0, nTurbines):
            for j in range(i+1, nTurbines):
                # separation wrt Xj
                dS[k, j] = 2*(turbineX[j]-turbineX[i])
                # separation wrt Xi
                dS[k, i] = -2*(turbineX[j]-turbineX[i])
                # separation wrt Yj
                dS[k, j+nTurbines] = 2*(turbineY[j]-turbineY[i])
                # separation wrt Yi
                dS[k, i+nTurbines] = -2*(turbineY[j]-turbineY[i])
                # increment turbine pair counter
                k += 1

        # initialize Jacobian dict
        J = {}

        # populate Jacobian dict
        J['wtSeparationSquared', 'turbineX'] = dS[:, :nTurbines]
        J['wtSeparationSquared', 'turbineY'] = dS[:, nTurbines:]

        return J


class BoundaryComp(Component):

    def __init__(self, nTurbines, nVertices):

        super(BoundaryComp, self).__init__()

        self.nTurbines = nTurbines
        self.nVertices = nVertices

        # Explicitly size input arrays
        self.add_param('boundaryVertices', np.zeros([nVertices, 2]), units='m', pass_by_obj=True,
                       desc="vertices of the convex hull CCW in order s.t. boundaryVertices[i] -> first point of face"
                            "for unit_normals[i]")
        self.add_param('boundaryNormals', np.zeros([nVertices, 2]), pass_by_obj=True,
                       desc="unit normal vector for each boundary face CCW where boundaryVertices[i] is "
                            "the first point of the corresponding face")
        self.add_param('turbineX', np.zeros(nTurbines), units='m',
                       desc='x coordinates of turbines in global ref. frame')
        self.add_param('turbineY', np.zeros(nTurbines), units='m',
                       desc='y coordinates of turbines in global ref. frame')

        # Explicitly size output array
        # (vector with positive elements if turbines outside of hull)
        self.add_output('boundaryDistances', np.zeros([nTurbines, nVertices]),
                        desc="signed perpendicular distance from each turbine to each face CCW; + is inside")

    def solve_nonlinear(self, params, unknowns, resids):

        turbineX = params['turbineX']
        turbineY = params['turbineY']

        # put locations in correct arrangement for calculations
        locations = np.zeros([self.nTurbines, 2])
        for i in range(0, self.nTurbines):
            locations[i] = np.array([turbineX[i], turbineY[i]])

        # print "in comp, locs are: ", locations

        # calculate distance from each point to each face
        unknowns['boundaryDistances'] = calculate_distance(locations,
                                                           params['boundaryVertices'], params['boundaryNormals'])

    def linearize(self, params, unknowns, resids):

        unit_normals = params['boundaryNormals']

        # initialize array to hold distances from each point to each face
        dfaceDistance_dx = np.zeros([self.nTurbines*self.nVertices, self.nTurbines])
        dfaceDistance_dy = np.zeros([self.nTurbines*self.nVertices, self.nTurbines])

        for i in range(0, self.nTurbines):
            # determine if point is inside or outside of each face, and distance from each face
            for j in range(0, self.nVertices):

                # define the derivative vectors from the point of interest to the first point of the face
                dpa_dx = np.array([-1.0, 0.0])
                dpa_dy = np.array([0.0, -1.0])

                # find perpendicular distance derivatives from point to current surface (vector projection)
                ddistanceVec_dx = np.vdot(dpa_dx, unit_normals[j])*unit_normals[j]
                ddistanceVec_dy = np.vdot(dpa_dy, unit_normals[j])*unit_normals[j]

                # calculate derivatives for the sign of perpendicular distance from point to current face
                dfaceDistance_dx[i*self.nVertices+j, i] = np.vdot(ddistanceVec_dx, unit_normals[j])
                dfaceDistance_dy[i*self.nVertices+j, i] = np.vdot(ddistanceVec_dy, unit_normals[j])

        # initialize Jacobian dict
        J = {}

        # return Jacobian dict
        J['boundaryDistances', 'turbineX'] = dfaceDistance_dx
        J['boundaryDistances', 'turbineY'] = dfaceDistance_dy

        return J


class MUX(Component):
    """ Connect input elements into a single array  """

    def __init__(self, nElements, units=None):

        super(MUX, self).__init__()

        # set finite difference options (fd used for testing only)
        # self.deriv_options['form'] = 'central'
        # self.deriv_options['step_size'] = 500.
        # self.deriv_options['step_calc'] = 'relative'

        # define necessary class attributes
        self.nElements = nElements

        # define inputs
        if units is None:
            for i in range(0, nElements):
                self.add_param('input%i' % i, val=0.0, desc='scalar input')
        else:
            for i in range(0, nElements):
                self.add_param('input%i' % i, val=0.0, units=units, desc='scalar input')

        # define output array
        if units is None:
            self.add_output('Array', np.zeros(nElements), desc='ndArray of all the scalar inputs')
        else:
            self.add_output('Array', np.zeros(nElements), units=units, desc='ndArray of all the scalar inputs')

    def solve_nonlinear(self, params, unknowns, resids):

        # assign input values to elements of the output array
        for i in range(0, self.nElements):
            exec("unknowns['Array'][%i] = params['input%i']" % (i, i))

    def linearize(self, params, unknowns, resids):

        # initialize gradient calculation array
        dArray_dInput = np.zeros(self.nElements)

        # initialize Jacobian dict
        J = {}

        # calculate gradient and populate Jacobian dict
        for i in range(0, self.nElements):
            dArray_dInput[i] = 1.0
            J['Array', 'input%i' % i] = np.array(dArray_dInput)
            dArray_dInput[i] = 0.0

        return J


class DeMUX(Component):
    """ split a given array into separate elements """

    def __init__(self, nElements, units=None):

        super(DeMUX, self).__init__()

        # set finite difference options (fd used for testing only)
        # self.deriv_options['form'] = 'central'
        # self.deriv_options['step_size'] = 500.
        # self.deriv_options['step_calc'] = 'relative'

        # initialize necessary class attributes
        self.nElements = nElements

        # define input
        if units is None:
            self.add_param('Array', np.zeros(nElements), desc='ndArray of scalars')
        else:
            self.add_param('Array', np.zeros(nElements), units=units, desc='ndArray of scalars')

        # define outputs
        if units is None:
            for i in range(0, nElements):
                self.add_output('output%i' % i, val=0.0, desc='scalar output')
        else:
            for i in range(0, nElements):
                self.add_output('output%i' % i, val=0.0, units=units, desc='scalar output')

    def solve_nonlinear(self, params, unknowns, resids):

        # assign elements of the input array to outputs
        for i in range(0, self.nElements):
            exec("unknowns['output%i'] = params['Array'][%i]" % (i, i))


    def linearize(self, params, unknowns, resids):

        # initialize gradient calculation array
        doutput_dArray = np.eye(self.nElements)

        # intialize Jacobian dict
        J = {}

        # calculate the gradients and populate the Jacobian dict
        for i in range(0, self.nElements):
            J['output%i' % i, 'Array'] = np.reshape(doutput_dArray[i, :], (1, self.nElements))

        return J

"""
Don't use this anywhere, may be able to delete
"""
class DeMUXArrays(Component):
    """ split a given array of arrays into separate arrays """

    def __init__(self, nElements, nArrays, units=None):

        super(DeMUXArrays, self).__init__()

        # set finite difference options (fd used for testing only)
        # self.deriv_options['form'] = 'central'
        # self.deriv_options['step_size'] = 500.
        # self.deriv_options['step_calc'] = 'relative'
        # self.deriv_options['type'] = 'fd'

        # initialize necessary class attributes
        self.nElements = nElements
        self.nArrays = nArrays

        # define input
        if units is None:
            self.add_param('Array', np.zeros((nArrays, nElements)), desc='ndArray of arrays')
        else:
            self.add_param('Array', np.zeros((nArrays, nElements)), units=units, desc='ndArray of arrays')

        # define outputs
        if units is None:
            for i in range(0, nArrays):
                self.add_output('output%i' % i, np.zeros(nElements), desc='scalar output')
        else:
            for i in range(0, nElements):
                self.add_output('output%i' % i, np.zeros(nElements), units=units, desc='scalar output')

    def solve_nonlinear(self, params, unknowns, resids):

        # assign elements of the input array to outputs
        for i in range(0, self.nArrays):
            exec("unknowns['output%i'] = params['Array'][%i][:]" % (i, i))
    """
    #TODO need to do linearize still
    def linearize(self, params, unknowns, resids):

        # initialize gradient calculation array
        doutput_dArray = np.eye(self.nElements)

        # intialize Jacobian dict
        J = {}

        # calculate the gradients and populate the Jacobian dict
        for i in range(0, self.nElements):
            J['output%i' % i, 'Array'] = np.reshape(doutput_dArray[i, :], (1, self.nElements))

        return J
    """


class organizeWindSpeeds(Component):
    """ split wind speeds to connect to direction groups """

    def __init__(self, nTurbines, nDirections, units=None):

        super(organizeWindSpeeds, self).__init__()

        # set finite difference options (fd used for testing only)
        # self.deriv_options['form'] = 'central'
        # self.deriv_options['step_size'] = 500.
        # self.deriv_options['step_calc'] = 'relative'
        # self.deriv_options['type'] = 'fd'

        self.nDirections = nDirections
        self.nTurbines = nTurbines

        # define input
        if units is None:
            self.add_param('windSpeeds', np.zeros((nTurbines, nDirections)), desc='ndArray of scalars')
        else:
            self.add_param('windSpeeds', np.zeros((nTurbines, nDirections)), units=units, desc='ndArray of scalars')

        # define outputs
        if units is None:
            for i in range(0, nDirections):
                self.add_output('output%i' % i, np.zeros(nTurbines), desc='scalar output')
        else:
            for i in range(0, nDirections):
                self.add_output('output%i' % i, np.zeros(nTurbines), units=units, desc='scalar output')

    def solve_nonlinear(self, params, unknowns, resids):

        nDirections = self.nDirections
        nTurbines = self.nTurbines

        # print 'ORGANIZE: ', params['windSpeeds']

        # assign elements of the input array to outputs
        for direction_id in range(nDirections):
            for turbine_id in range(nTurbines):
                unknowns['output%i'%direction_id][turbine_id] = params['windSpeeds'][turbine_id][direction_id]
        # print 'organizeWindSpeeds'
        # print 'windSpeeds: ', params['windSpeeds']
        # print 'output0: ', unknowns['output0']
        # print 'output1: ', unknowns['output1']
    """
    #TODO need to do linearize?????
    &&&&&&&&&&&&&&&&&&&&&&&&&&
    &&&&&&&&&&&&&&&&&&&&&&&&&&
    &&&&&&&&&&&&&&&&&&&&&&&&&&
    &&&&&&&&&&&&&&&&&&&&&&&&&&
    &&&&&&&&&&&&&&&&&&&&&&&&&&
    """
    def linearize(self, params, unknowns, resids):

        nTurbines = self.nTurbines
        nDirections = self.nDirections

        # intialize Jacobian dict
        J = {}

        for direction_id in range(nDirections):
            J['output%i'%direction_id, 'windSpeeds'] = np.zeros((nTurbines, nDirections*nTurbines))
            for turbine_id in range(nTurbines):
                J['output%i'%direction_id, 'windSpeeds'][turbine_id][nDirections*turbine_id+direction_id] = 1.0

        # print 'organizeWindSpeeds GRADIENTS: '
        # print 'output0: ', J['output0', 'windSpeeds']
        # print 'output1: ', J['output1', 'windSpeeds']
            # print "Index for output%i: "%direction_id, np.argwhere(J['output%i'%direction_id, 'windSpeeds'])
            # print "THIS IS J!!!! for output%i: "%direction_id, J['output%i'%direction_id, 'windSpeeds']


        return J


# ---- if you know wind speed to power and thrust, you can use these tools ----------------
class CPCT_Interpolate_Gradients(Component):

    def __init__(self, nTurbines, direction_id=0, datasize=0):

        super(CPCT_Interpolate_Gradients, self).__init__()

        # # set finite difference options (fd used for testing only)
        # self.deriv_options['form'] = 'central'
        # self.deriv_options['step_size'] = 500.
        # self.deriv_options['step_calc'] = 'relative'

        # define class attributes
        self.nTurbines = nTurbines
        self.direction_id = direction_id
        self.datasize = datasize

        # add inputs and outputs
        self.add_param('yaw%i' % direction_id, np.zeros(nTurbines), desc='yaw error', units='deg')
        self.add_param('wtVelocity%i' % direction_id, np.zeros(nTurbines), units='m/s', desc='hub height wind speed') # Uhub
        self.add_output('Cp_out', np.zeros(nTurbines))
        self.add_output('Ct_out', np.zeros(nTurbines))

        # add variable trees
        self.add_param('gen_params:pP', 1.88, pass_by_obj=True)
        self.add_param('gen_params:windSpeedToCPCT_wind_speed', np.zeros(datasize), units='m/s',
                       desc='range of wind speeds', pass_by_obj=True)
        self.add_param('gen_params:windSpeedToCPCT_CP', np.zeros(datasize), iotype='out',
                       desc='power coefficients', pass_by_obj=True)
        self.add_param('gen_params:windSpeedToCPCT_CT', np.zeros(datasize), iotype='out',
                       desc='thrust coefficients', pass_by_obj=True)

    def solve_nonlinear(self, params, unknowns, resids):

        # obtain necessary inputs
        direction_id = self.direction_id
        pP = self.params['gen_params:pP']

        wind_speed_ax = np.cos(self.params['yaw%i' % direction_id]*np.pi/180.0)**(pP/3.0)*self.params['wtVelocity%i' % direction_id]
        # use interpolation on precalculated CP-CT curve
        wind_speed_ax = np.maximum(wind_speed_ax, self.params['gen_params:windSpeedToCPCT_wind_speed'][0])
        wind_speed_ax = np.minimum(wind_speed_ax, self.params['gen_params:windSpeedToCPCT_wind_speed'][-1])
        self.unknowns['Cp_out'] = interp(wind_speed_ax, self.params['gen_params:windSpeedToCPCT_wind_speed'], self.params['gen_params:windSpeedToCPCT_CP'])
        self.unknowns['Ct_out'] = interp(wind_speed_ax, self.params['gen_params:windSpeedToCPCT_wind_speed'], self.params['gen_params:windSpeedToCPCT_CT'])

        # for i in range(0, len(self.unknowns['Ct_out'])):
        #     self.unknowns['Ct_out'] = max(max(self.unknowns['Ct_out']), self.unknowns['Ct_out'][i])
        # normalize on incoming wind speed to correct coefficients for yaw
        self.unknowns['Cp_out'] = self.unknowns['Cp_out'] * np.cos(self.params['yaw%i' % direction_id]*np.pi/180.0)**pP
        self.unknowns['Ct_out'] = self.unknowns['Ct_out'] * np.cos(self.params['yaw%i' % direction_id]*np.pi/180.0)**2
        # print 'in CPCT interp, wind_speed_hub = ', self.params['wtVelocity%i' % direction_id]
        # print 'in CPCT: ', params['velocitiesTurbines0']

    def linearize(self, params, unknowns, resids):  # standard central differencing
        # set step size for finite differencing
        h = 1e-6
        direction_id = self.direction_id

        # calculate upper and lower function values
        wind_speed_ax_high_yaw = np.cos((self.params['yaw%i' % direction_id]+h)*np.pi/180.0)**(self.params['gen_params:pP']/3.0)*self.params['wtVelocity%i' % direction_id]
        wind_speed_ax_low_yaw = np.cos((self.params['yaw%i' % direction_id]-h)*np.pi/180.0)**(self.params['gen_params:pP']/3.0)*self.params['wtVelocity%i' % direction_id]
        wind_speed_ax_high_wind = np.cos(self.params['yaw%i' % direction_id]*np.pi/180.0)**(self.params['gen_params:pP']/3.0)*(self.params['wtVelocity%i' % direction_id]+h)
        wind_speed_ax_low_wind = np.cos(self.params['yaw%i' % direction_id]*np.pi/180.0)**(self.params['gen_params:pP']/3.0)*(self.params['wtVelocity%i' % direction_id]-h)

        # use interpolation on precalculated CP-CT curve
        wind_speed_ax_high_yaw = np.maximum(wind_speed_ax_high_yaw, self.params['gen_params:windSpeedToCPCT_wind_speed'][0])
        wind_speed_ax_low_yaw = np.maximum(wind_speed_ax_low_yaw, self.params['gen_params:windSpeedToCPCT_wind_speed'][0])
        wind_speed_ax_high_wind = np.maximum(wind_speed_ax_high_wind, self.params['gen_params:windSpeedToCPCT_wind_speed'][0])
        wind_speed_ax_low_wind = np.maximum(wind_speed_ax_low_wind, self.params['gen_params:windSpeedToCPCT_wind_speed'][0])

        wind_speed_ax_high_yaw = np.minimum(wind_speed_ax_high_yaw, self.params['gen_params:windSpeedToCPCT_wind_speed'][-1])
        wind_speed_ax_low_yaw = np.minimum(wind_speed_ax_low_yaw, self.params['gen_params:windSpeedToCPCT_wind_speed'][-1])
        wind_speed_ax_high_wind = np.minimum(wind_speed_ax_high_wind, self.params['gen_params:windSpeedToCPCT_wind_speed'][-1])
        wind_speed_ax_low_wind = np.minimum(wind_speed_ax_low_wind, self.params['gen_params:windSpeedToCPCT_wind_speed'][-1])

        CP_high_yaw = interp(wind_speed_ax_high_yaw, self.params['gen_params:windSpeedToCPCT_wind_speed'], self.params['gen_params:windSpeedToCPCT_CP'])
        CP_low_yaw = interp(wind_speed_ax_low_yaw, self.params['gen_params:windSpeedToCPCT_wind_speed'], self.params['gen_params:windSpeedToCPCT_CP'])
        CP_high_wind = interp(wind_speed_ax_high_wind, self.params['gen_params:windSpeedToCPCT_wind_speed'], self.params['gen_params:windSpeedToCPCT_CP'])
        CP_low_wind = interp(wind_speed_ax_low_wind, self.params['gen_params:windSpeedToCPCT_wind_speed'], self.params['gen_params:windSpeedToCPCT_CP'])

        CT_high_yaw = interp(wind_speed_ax_high_yaw, self.params['gen_params:windSpeedToCPCT_wind_speed'], self.params['gen_params:windSpeedToCPCT_CT'])
        CT_low_yaw = interp(wind_speed_ax_low_yaw, self.params['gen_params:windSpeedToCPCT_wind_speed'], self.params['gen_params:windSpeedToCPCT_CT'])
        CT_high_wind = interp(wind_speed_ax_high_wind, self.params['gen_params:windSpeedToCPCT_wind_speed'], self.params['gen_params:windSpeedToCPCT_CT'])
        CT_low_wind = interp(wind_speed_ax_low_wind, self.params['gen_params:windSpeedToCPCT_wind_speed'], self.params['gen_params:windSpeedToCPCT_CT'])

        # normalize on incoming wind speed to correct coefficients for yaw
        CP_high_yaw = CP_high_yaw * np.cos((self.params['yaw%i' % direction_id]+h)*np.pi/180.0)**self.params['gen_params:pP']
        CP_low_yaw = CP_low_yaw * np.cos((self.params['yaw%i' % direction_id]-h)*np.pi/180.0)**self.params['gen_params:pP']
        CP_high_wind = CP_high_wind * np.cos((self.params['yaw%i' % direction_id])*np.pi/180.0)**self.params['gen_params:pP']
        CP_low_wind = CP_low_wind * np.cos((self.params['yaw%i' % direction_id])*np.pi/180.0)**self.params['gen_params:pP']

        CT_high_yaw = CT_high_yaw * np.cos((self.params['yaw%i' % direction_id]+h)*np.pi/180.0)**2
        CT_low_yaw = CT_low_yaw * np.cos((self.params['yaw%i' % direction_id]-h)*np.pi/180.0)**2
        CT_high_wind = CT_high_wind * np.cos((self.params['yaw%i' % direction_id])*np.pi/180.0)**2
        CT_low_wind = CT_low_wind * np.cos((self.params['yaw%i' % direction_id])*np.pi/180.0)**2

        # compute derivative via central differencing and arrange in sub-matrices of the Jacobian
        dCP_dyaw = np.eye(self.nTurbines)*(CP_high_yaw-CP_low_yaw)/(2.0*h)
        dCP_dwind = np.eye(self.nTurbines)*(CP_high_wind-CP_low_wind)/(2.0*h)
        dCT_dyaw = np.eye(self.nTurbines)*(CT_high_yaw-CT_low_yaw)/(2.0*h)
        dCT_dwind = np.eye(self.nTurbines)*(CT_high_wind-CT_low_wind)/(2.0*h)

        # compile Jacobian dict from sub-matrices
        J = {}
        J['Cp_out', 'yaw%i' % direction_id] = dCP_dyaw
        J['Cp_out', 'wtVelocity%i' % direction_id] = dCP_dwind
        J['Ct_out', 'yaw%i' % direction_id] = dCT_dyaw
        J['Ct_out', 'wtVelocity%i' % direction_id] = dCT_dwind

        return J


class CPCT_Interpolate_Gradients_Smooth(Component):

    def __init__(self, nTurbines, direction_id=0, datasize=0):

        super(CPCT_Interpolate_Gradients_Smooth, self).__init__()

        # set finite difference options (fd used for testing only)
        # self.deriv_options['form'] = 'central'
        # self.deriv_options['step_size'] = 500.
        # self.deriv_options['step_calc'] = 'relative'

        # define class attributes
        self.nTurbines = nTurbines
        self.direction_id = direction_id
        self.datasize = datasize

        # add inputs and outputs
        self.add_param('yaw%i' % direction_id, np.zeros(nTurbines), desc='yaw error', units='deg')
        self.add_param('wtVelocity%i' % direction_id, np.zeros(nTurbines), units='m/s', desc='hub height wind speed') # Uhub
        self.add_output('Cp_out', np.zeros(nTurbines))
        self.add_output('Ct_out', np.zeros(nTurbines))

        # add variable trees
        self.add_param('gen_params:pP', 3.0, pass_by_obj=True)
        self.add_param('gen_params:windSpeedToCPCT_wind_speed', np.zeros(datasize), units='m/s',
                       desc='range of wind speeds', pass_by_obj=True)
        self.add_param('gen_params:windSpeedToCPCT_CP', np.zeros(datasize),
                       desc='power coefficients', pass_by_obj=True)
        self.add_param('gen_params:windSpeedToCPCT_CT', np.zeros(datasize),
                       desc='thrust coefficients', pass_by_obj=True)

    def solve_nonlinear(self, params, unknowns, resids):
        direction_id = self.direction_id
        pP = self.params['gen_params:pP']
        yaw = self.params['yaw%i' % direction_id]
        start = 5
        skip = 8
        # Cp = params['gen_params:windSpeedToCPCT_CP'][start::skip]
        Cp = params['gen_params:windSpeedToCPCT_CP']
        # Ct = params['gen_params:windSpeedToCPCT_CT'][start::skip]
        Ct = params['gen_params:windSpeedToCPCT_CT']
        # windspeeds = params['gen_params:windSpeedToCPCT_wind_speed'][start::skip]
        windspeeds = params['gen_params:windSpeedToCPCT_wind_speed']
        #
        # Cp = np.insert(Cp, 0, Cp[0]/2.0)
        # Cp = np.insert(Cp, 0, 0.0)
        # Ct = np.insert(Ct, 0, np.max(params['gen_params:windSpeedToCPCT_CP'])*0.99)
        # Ct = np.insert(Ct, 0, np.max(params['gen_params:windSpeedToCPCT_CT']))
        # windspeeds = np.insert(windspeeds, 0, 2.5)
        # windspeeds = np.insert(windspeeds, 0, 0.0)
        #
        # Cp = np.append(Cp, 0.0)
        # Ct = np.append(Ct, 0.0)
        # windspeeds = np.append(windspeeds, 30.0)

        CPspline = Akima(windspeeds, Cp)
        CTspline = Akima(windspeeds, Ct)

        # n = 500
        # x = np.linspace(0.0, 30., n)
        CP, dCPdvel, _, _ = CPspline.interp(params['wtVelocity%i' % direction_id])
        CT, dCTdvel, _, _ = CTspline.interp(params['wtVelocity%i' % direction_id])

        # print 'in solve_nonlinear', dCPdvel, dCTdvel
        # pP = 3.0
        # print "in rotor, pP = ", pP
        Cp_out = CP*np.cos(yaw*np.pi/180.)**pP
        Ct_out = CT*np.cos(yaw*np.pi/180.)**2.

        # print "in rotor, Cp = [%f. %f], Ct = [%f, %f]" % (Cp_out[0], Cp_out[1], Ct_out[0], Ct_out[1])

        self.dCp_out_dyaw = (-np.sin(yaw*np.pi/180.))*(np.pi/180.)*pP*CP*np.cos(yaw*np.pi/180.)**(pP-1.)
        self.dCp_out_dvel = dCPdvel*np.cos(yaw*np.pi/180.)**pP

        # print 'in solve_nonlinear', self.dCp_out_dyaw, self.dCp_out_dvel

        self.dCt_out_dyaw = (-np.sin(yaw*np.pi/180.))*(np.pi/180.)*2.*CT*np.cos(yaw*np.pi/180.)
        self.dCt_out_dvel = dCTdvel*np.cos(yaw*np.pi/180.)**2.

        # normalize on incoming wind speed to correct coefficients for yaw
        self.unknowns['Cp_out'] = Cp_out
        self.unknowns['Ct_out'] = Ct_out

    def linearize(self, params, unknowns, resids):  # standard central differencing

        # obtain necessary inputs
        direction_id = self.direction_id

        # compile Jacobian dict
        J = {}
        J['Cp_out', 'yaw%i' % direction_id] = np.eye(self.nTurbines)*self.dCp_out_dyaw
        J['Cp_out', 'wtVelocity%i' % direction_id] = np.eye(self.nTurbines)*self.dCp_out_dvel
        J['Ct_out', 'yaw%i' % direction_id] = np.eye(self.nTurbines)*self.dCt_out_dyaw
        J['Ct_out', 'wtVelocity%i' % direction_id] = np.eye(self.nTurbines)*self.dCt_out_dvel

        return J


class WindDirectionPower(Component):

    def __init__(self, nTurbines, direction_id=0, differentiable=True, use_rotor_components=False):

        super(WindDirectionPower, self).__init__()

        # define class attributes
        self.differentiable = differentiable
        self.nTurbines = nTurbines
        self.direction_id = direction_id
        self.use_rotor_components = use_rotor_components

        # # set finite difference options (only used for testing)
        # self.deriv_options['form'] = 'central'
        # self.deriv_options['step_size'] = 500.
        # self.deriv_options['step_calc'] = 'relative'
        #
        # if not differentiable:
        #     self.deriv_options['type'] = 'fd'
        #     self.deriv_options['form'] = 'forward'

        self.add_param('air_density', 1.1716, units='kg/(m*m*m)', desc='air density in free stream')
        self.add_param('rotorDiameter', np.zeros(nTurbines) + 126.4, units='m', desc='rotor diameters of all turbine')
        self.add_param('Cp', np.zeros(nTurbines)+(0.7737/0.944) * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2), desc='power coefficient for all turbines')
        self.add_param('generatorEfficiency', np.zeros(nTurbines)+0.944, desc='generator efficiency of all turbines')
        self.add_param('wtVelocity%i' % direction_id, np.zeros(nTurbines), units='m/s',
                       desc='effective hub velocity for each turbine')

        self.add_param('ratedPower', np.ones(nTurbines)*5000., units='kW',
                       desc='rated power for each turbine', pass_by_obj=True)

        # outputs
        self.add_output('wtPower%i' % direction_id, np.zeros(nTurbines), units='kW', desc='power output of each turbine')
        self.add_output('dir_power%i' % direction_id, 0.0, units='kW', desc='total power output of the wind farm')

    def solve_nonlinear(self, params, unknowns, resids):

        # obtain necessary inputs
        use_rotor_components = self.use_rotor_components
        direction_id = self.direction_id
        nTurbines = self.nTurbines
        wtVelocity = self.params['wtVelocity%i' % direction_id]
        ratedPower = params['ratedPower']
        air_density = params['air_density']
        rotorArea = 0.25*np.pi*np.power(params['rotorDiameter'], 2)
        Cp = params['Cp']
        generatorEfficiency = params['generatorEfficiency']
        # print 'DirPowers: '
        # print 'wtVelocity: ', wtVelocity

        # calculate initial values for wtPower (W)
        wtPower = generatorEfficiency*(0.5*air_density*rotorArea*Cp*np.power(wtVelocity, 3))

        # adjust units from W to kW
        wtPower /= 1000.0

        # rated_velocity = np.power(1000.*ratedPower/(generator_efficiency*(0.5*air_density*rotorArea*Cp)), 1./3.)
        #
        # dwt_power_dvelocitiesTurbines = np.eye(nTurbines)*generator_efficiency*(1.5*air_density*rotorArea*Cp *
        #                                                                         np.power(wtVelocity, 2))
        # dwt_power_dvelocitiesTurbines /= 1000.

        # adjust wt power based on rated power
        if not use_rotor_components and np.any(wtPower) >= np.any(ratedPower):
            for i in range(0, nTurbines):
                if wtPower[i] >= ratedPower[i]:
                    wtPower[i] = ratedPower[i]


        # if np.any(rated_velocity+1.) >= np.any(wtVelocity) >= np.any(rated_velocity-1.) and not \
        #         use_rotor_components:
        #     for i in range(0, nTurbines):
        #         if wtVelocity[i] >= rated_velocity[i]+1.:
        #             spline_start_power = generator_efficiency[i]*(0.5*air_density*rotorArea[i]*Cp[i]*np.power(rated_velocity[i]-1., 3))
        #             deriv_spline_start_power = 3.*generator_efficiency[i]*(0.5*air_density*rotorArea[i]*Cp[i]*np.power(rated_velocity[i]-1., 2))
        #             spline_end_power = generator_efficiency[i]*(0.5*air_density*rotorArea[i]*Cp[i]*np.power(rated_velocity[i]+1., 3))
        #             wtPower[i], deriv = hermite_spline(wtVelocity[i], rated_velocity[i]-1.,
        #                                                                      rated_velocity[i]+1., spline_start_power,
        #                                                                      deriv_spline_start_power, spline_end_power, 0.0)
        #             dwt_power_dvelocitiesTurbines[i][i] = deriv/1000.
        #
        # if np.any(wtVelocity) >= np.any(rated_velocity+1.) and not use_rotor_components:
        #     for i in range(0, nTurbines):
        #         if wtVelocity[i] >= rated_velocity[i]+1.:
        #             wtPower = ratedPower
        #             dwt_power_dvelocitiesTurbines[i][i] = 0.0



        # self.dwt_power_dvelocitiesTurbines = dwt_power_dvelocitiesTurbines

        # calculate total power for this direction
        dir_power = np.sum(wtPower)

        # print 'DirectionPower:'
        # print 'wtVelocity: ', wtVelocity
        # print 'wtPower: ', wtPower
        # print 'dir_power: ', dir_power
        # pass out results
        unknowns['wtPower%i' % direction_id] = wtPower
        unknowns['dir_power%i' % direction_id] = dir_power

        # print wtPower

    def linearize(self, params, unknowns, resids):

        # obtain necessary inputs
        direction_id = self.direction_id
        use_rotor_components = self.use_rotor_components
        nTurbines = self.nTurbines
        wtVelocity = self.params['wtVelocity%i' % direction_id]
        air_density = params['air_density']
        rotorDiameter = params['rotorDiameter']
        rotorArea = 0.25*np.pi*np.power(rotorDiameter, 2)
        Cp = params['Cp']
        generatorEfficiency = params['generatorEfficiency']
        ratedPower = params['ratedPower']
        wtPower = unknowns['wtPower%i' % direction_id]

        # calcuate initial gradient values
        dwtPower_dwtVelocity = np.eye(nTurbines)*generatorEfficiency*(1.5*air_density*rotorArea*Cp *
                                                                                np.power(wtVelocity, 2))
        dwtPower_dCp = np.eye(nTurbines)*generatorEfficiency*(0.5*air_density*rotorArea*np.power(wtVelocity, 3))
        dwtPower_drotorDiameter = np.eye(nTurbines)*generatorEfficiency*(0.5*air_density*(0.5*np.pi*rotorDiameter)*Cp *
                                                                           np.power(wtVelocity, 3))
        # dwt_power_dvelocitiesTurbines = self.dwt_power_dvelocitiesTurbines

        # adjust gradients for unit conversion from W to kW
        dwtPower_dwtVelocity /= 1000.
        dwtPower_dCp /= 1000.
        dwtPower_drotorDiameter /= 1000.

        # rated_velocity = np.power(1000.*ratedPower/(generator_efficiency*(0.5*air_density*rotorArea*Cp)), 1./3.)

        # if np.any(rated_velocity+1.) >= np.any(wtVelocity) >= np.any(rated_velocity-1.) and not \
        #         use_rotor_components:
        #
        #     spline_start_power = generator_efficiency*(0.5*air_density*rotorArea*Cp*np.power(rated_velocity-1., 3))
        #     deriv_spline_start_power = 3.*generator_efficiency*(0.5*air_density*rotorArea*Cp*np.power(rated_velocity-1., 2))
        #     spline_end_power = generator_efficiency*(0.5*air_density*rotorArea*Cp*np.power(rated_velocity+1., 3))
        #     wtPower, dwt_power_dvelocitiesTurbines = hermite_spline(wtVelocity, rated_velocity-1.,
        #                                                              rated_velocity+1., spline_start_power,
        #                                                              deriv_spline_start_power, spline_end_power, 0.0)

        # set gradients for turbines above rated power to zero
        if np.any(wtPower) >= np.any(ratedPower) and not use_rotor_components:
            for i in range(0, nTurbines):
                if wtPower[i] >= ratedPower[i]:
                    dwtPower_dwtVelocity[i][i] = 0.0
                    dwtPower_dCp[i][i] = 0.0
                    dwtPower_drotorDiameter[i][i] = 0.0

        # compile elements of Jacobian
        ddir_power_dwtVelocity = np.array([np.sum(dwtPower_dwtVelocity, 0)])
        ddir_power_dCp = np.array([np.sum(dwtPower_dCp, 0)])
        ddir_power_drotorDiameter = np.array([np.sum(dwtPower_drotorDiameter, 0)])

        # print 'DirectionPower Gradients:'
        # print 'ddir_power_dwtVelocity: ', ddir_power_dwtVelocity
        # print 'ddir_power_dCp: ', ddir_power_dCp
        # print 'ddir_power_drotorDiameter: ', ddir_power_drotorDiameter

        # initialize Jacobian dict
        J = {}

        # populate Jacobian dict
        J['wtPower%i' % direction_id, 'wtVelocity%i' % direction_id] = dwtPower_dwtVelocity
        J['wtPower%i' % direction_id, 'Cp'] = dwtPower_dCp
        J['wtPower%i' % direction_id, 'rotorDiameter'] = dwtPower_drotorDiameter

        J['dir_power%i' % direction_id, 'wtVelocity%i' % direction_id] = ddir_power_dwtVelocity
        J['dir_power%i' % direction_id, 'Cp'] = ddir_power_dCp
        J['dir_power%i' % direction_id, 'rotorDiameter'] = ddir_power_drotorDiameter

        return J

#
# def calculate_boundary(vertices):
#
#     # find the points that actually comprise a convex hull
#     hull = ConvexHull(list(vertices))
#
#     # keep only vertices that actually comprise a convex hull and arrange in CCW order
#     vertices = vertices[hull.vertices]
#
#     # get the real number of vertices
#     nVertices = vertices.shape[0]
#
#     # initialize normals array
#     unit_normals = np.zeros([nVertices, 2])
#
#     # determine if point is inside or outside of each face, and distance from each face
#     for j in range(0, nVertices):
#
#         # calculate the unit normal vector of the current face (taking points CCW)
#         if j < nVertices - 1:  # all but the set of point that close the shape
#             normal = np.array([vertices[j+1, 1]-vertices[j, 1],
#                                -(vertices[j+1, 0]-vertices[j, 0])])
#             unit_normals[j] = normal/np.linalg.norm(normal)
#         else:   # the set of points that close the shape
#             normal = np.array([vertices[0, 1]-vertices[j, 1],
#                                -(vertices[0, 0]-vertices[j, 0])])
#             unit_normals[j] = normal/np.linalg.norm(normal)
#
#     return vertices, unit_normals
#
#
# def calculate_distance(points, vertices, unit_normals, return_bool=False):
#
#     """
#     :param points: points that you want to calculate the distance from to the faces of the convex hull
#     :param vertices: vertices of the convex hull CCW in order s.t. vertices[i] -> first point of face for
#            unit_normals[i]
#     :param unit_normals: unit normal vector for each face CCW where vertices[i] is first point of face
#     :param return_bool: set to True to return an array of bools where True means the corresponding point
#            is inside the hull
#     :return face_distace: signed perpendicular distance from each point to each face; + is inside)
#     :return [inside]: (optional) an array of zeros and ones where 1.0 means the corresponding point is inside the hull
#     """
#     print points.shape, vertices.shape, unit_normals.shape
#     nPoints = len(points[0, :])
#     nVertices = len(unit_normals)
#
#     # initialize array to hold distances from each point to each face
#     face_distance = np.zeros([nPoints, nVertices])
#
#     if not return_bool:
#         # loop through points and find distance to each face
#         for i in range(0, nPoints):
#
#             # determine if point is inside or outside of each face, and distance from each face
#             for j in range(0, nVertices):
#
#                 # define the vector from the point of interest to the first point of the face
#                 pa = np.array([vertices[j, 0]-points[0, i], vertices[j, 1]-points[0, i]])
#
#                 # find perpendicular distance from point to current surface (vector projection)
#                 d_vec = np.vdot(pa, unit_normals[j])*unit_normals[j]
#
#                 # calculate the sign of perpendicular distance from point to current face (+ is inside, - is outside)
#                 face_distance[i, j] = np.vdot(d_vec, unit_normals[j])
#
#         return face_distance
#
#     else:
#         # initialize array to hold boolean indicating whether a point is inside the hull or not
#         inside = np.zeros(nPoints)
#
#         # loop through points and find distance to each face
#         for i in range(0, nPoints):
#
#             # determine if point is inside or outside of each face, and distance from each face
#             for j in range(0, nVertices):
#
#                 # define the vector from the point of interest to the first point of the face
#                 pa = np.array([vertices[j, 0]-points[0, i], vertices[j, 1]-points[1, i]])
#
#                 # find perpendicular distance from point to current surface (vector projection)
#                 d_vec = np.vdot(pa, unit_normals[j])*unit_normals[j]
#
#                 # calculate the sign of perpendicular distance from point to current face (+ is inside, - is outside)
#                 face_distance[i, j] = np.vdot(d_vec, unit_normals[j])
#
#             # check if the point is inside the convex hull by checking the sign of the distance
#             if np.all(face_distance[i] > 0):
#                 inside[i] = 1.0
#
#         return face_distance, inside
#

def calculate_boundary(vertices):

    # find the points that actually comprise a convex hull
    hull = ConvexHull(list(vertices))

    # keep only vertices that actually comprise a convex hull and arrange in CCW order
    vertices = vertices[hull.vertices]

    # get the real number of vertices
    nVertices = vertices.shape[0]

    # initialize normals array
    unit_normals = np.zeros([nVertices, 2])

    # determine if point is inside or outside of each face, and distance from each face
    for j in range(0, nVertices):

        # calculate the unit normal vector of the current face (taking points CCW)
        if j < nVertices - 1:  # all but the set of point that close the shape
            normal = np.array([vertices[j+1, 1]-vertices[j, 1],
                               -(vertices[j+1, 0]-vertices[j, 0])])
            unit_normals[j] = normal/np.linalg.norm(normal)
        else:   # the set of points that close the shape
            normal = np.array([vertices[0, 1]-vertices[j, 1],
                               -(vertices[0, 0]-vertices[j, 0])])
            unit_normals[j] = normal/np.linalg.norm(normal)

    return vertices, unit_normals


def calculate_distance(points, vertices, unit_normals, return_bool=False):

    """
    :param points: points that you want to calculate the distance from to the faces of the convex hull
    :param vertices: vertices of the convex hull CCW in order s.t. vertices[i] -> first point of face for
           unit_normals[i]
    :param unit_normals: unit normal vector for each face CCW where vertices[i] is first point of face
    :param return_bool: set to True to return an array of bools where True means the corresponding point
           is inside the hull
    :return face_distace: signed perpendicular distance from each point to each face; + is inside
    :return [inside]: (optional) an array of zeros and ones where 1.0 means the corresponding point is inside the hull
    """

    # print points.shape, vertices.shape, unit_normals.shape

    nPoints = points.shape[0]
    nVertices = vertices.shape[0]

    # initialize array to hold distances from each point to each face
    face_distance = np.zeros([nPoints, nVertices])

    if not return_bool:
        # loop through points and find distance to each face
        for i in range(0, nPoints):

            # determine if point is inside or outside of each face, and distance from each face
            for j in range(0, nVertices):

                # define the vector from the point of interest to the first point of the face
                pa = np.array([vertices[j, 0]-points[i, 0], vertices[j, 1]-points[i, 1]])

                # find perpendicular distance from point to current surface (vector projection)
                d_vec = np.vdot(pa, unit_normals[j])*unit_normals[j]

                # calculate the sign of perpendicular distance from point to current face (+ is inside, - is outside)
                face_distance[i, j] = np.vdot(d_vec, unit_normals[j])

        return face_distance

    else:
        # initialize array to hold boolean indicating whether a point is inside the hull or not
        inside = np.zeros(nPoints)

        # loop through points and find distance to each face
        for i in range(0, nPoints):

            # determine if point is inside or outside of each face, and distance from each face
            for j in range(0, nVertices):

                # define the vector from the point of interest to the first point of the face
                pa = np.array([vertices[j, 0]-points[i, 0], vertices[j, 1]-points[i, 1]])

                # find perpendicular distance from point to current surface (vector projection)
                d_vec = np.vdot(pa, unit_normals[j])*unit_normals[j]

                # calculate the sign of perpendicular distance from point to current face (+ is inside, - is outside)
                face_distance[i, j] = np.vdot(d_vec, unit_normals[j])

            # check if the point is inside the convex hull by checking the sign of the distance
            if np.all(face_distance[i] >= 0):
                inside[i] = 1.0

        return face_distance, inside


class getUeffintegrate(Component):
    """
    Integrate across the turbine to get effective wind speed
    """
    def __init__(self, nDirections, nTurbines):

        super(getUeffintegrate, self).__init__()

        # self.deriv_options['form'] = 'forward'
        # self.deriv_options['step_size'] = 500.
        # self.deriv_options['step_calc'] = 'relative'
        # self.deriv_options['type'] = 'fd'

        self.nDirections = nDirections
        self.nTurbines = nTurbines

        # inputs
        self.add_param('nIntegrationPoints', 1, desc='number of integration points', pass_by_obj=True)
        self.add_param('rotorDiameter', np.zeros(nTurbines), units='m', desc='rotor diameter of each turbine')
        self.add_param('turbineZ', np.zeros(nTurbines), units='m', desc='the hub height of each turbine')
        self.add_param('wind', 'PowerWind', desc='Wind shear calculation method', pass_by_obj=True)
        self.add_param('Uref', np.zeros(nDirections), units='m/s', desc='refenence wind speed for each direction')
        self.add_param('zref', 90, units='m', desc='height at which Uref was measured', pass_by_obj=True)
        self.add_param('z_roughness', 0.01, units='m', desc='ground roughness height', pass_by_obj=True)
        self.add_param('z0', 0, units='m', desc='height of ground', pass_by_obj=True)
        self.add_param('shearExp', 0.2, desc='PowerWind exponent', pass_by_obj=True)

        # outputs
        self.add_output('windSpeeds', np.zeros((nTurbines, nDirections)), units='m/s', desc='Free stream wind speed on each turbine from each direction')


    def solve_nonlinear(self, params, unknowns, resids):

        nTurbines = self.nTurbines
        nDirections = self.nDirections

        D = params['rotorDiameter']
        r = D/2.

        nPoints = params['nIntegrationPoints']
        wind = params['wind']
        Uref = params['Uref']
        zref = params['zref']
        z_roughness = params['z_roughness']
        z0 = params['z0']
        shearExp = params['shearExp']
        turbineZ = params['turbineZ']

        past = np.array([]) #an array of the heights for which Ueff has already been calculated

        for turbine_id in range(nTurbines):
            turbZ = turbineZ[turbine_id]
            #check if Ueff has already been calculated for this turbine height
            if turbZ in past:
                Ueff = unknowns['windSpeeds'][np.argwhere(past==turbZ)[0]][:]

            else:
                Ueff = np.zeros(nDirections)
                rTurb = r[turbine_id]
                for direction_id in range(nDirections):
                    z = turbZ-rTurb
                    Usum = 0.
                    Asum = 0.
                    for point_id in range(nPoints):
                        dz = D[turbine_id]/nPoints
                        if point_id == 0 or point_id == nPoints-1:
                            dz = dz/2.

                        if z < turbZ:
                            a1 = 2.*np.sqrt(rTurb**2-(turbZ-z)**2)
                        elif z == turbZ:
                            a1 = 2.*rTurb
                        else:
                            a1 = 2.*np.sqrt(rTurb**2-(z-turbZ)**2)

                        if z+dz < turbZ:
                            a2 = 2.*np.sqrt(rTurb**2-(turbZ-(z+dz))**2)
                        elif z+dz == turbZ:
                            a2 = 2.*rTurb
                        else:
                            a2 = 2.*np.sqrt(rTurb**2-(z+dz-turbZ)**2)

                        if wind == 'PowerWind':
                            Ub = PowWind(Uref[direction_id], z, zref, z0, shearExp)
                            Ut = PowWind(Uref[direction_id], z+dz, zref, z0, shearExp)
                        if wind == 'LogWind':
                            Ub = LnWind(Uref[direction_id], z, z0, z_roughness, zref)
                            Ut = LnWind(Uref[direction_id], z, z0, z_roughness, zref)

                        Usum += dz/2.*(a1*Ub+a2*Ut)
                        Asum += dz/2.*(a1+a2)
                        z += dz
                    Ueff[direction_id] = Usum/Asum

            unknowns['windSpeeds'][turbine_id][:] = Ueff
            past = np.append(past, turbZ)

    """
    #TODO need to do linearize
    &&&&&&&&&&&&&&&&&&&&&&&&&&
    &&&&&&&&&&&&&&&&&&&&&&&&&&
    &&&&&&&&&&&&&&&&&&&&&&&&&&
    &&&&&&&&&&&&&&&&&&&&&&&&&&
    &&&&&&&&&&&&&&&&&&&&&&&&&&
    """
    """
    def linearize(self, params, unknowns, resids):

        nTurbines = self.nTurbines
        nDirections = self.nDirections

        # intialize Jacobian dict
        J = {}

        for direction_id in range(nDirections):
            for turbine_id in range(nTurbines):
                J['windSpeeds', 'turbineZ'][turbine_id] = np.zeros((nTurbines, nDirections))
                J['output%i'%direction_id, 'windSpeeds'][turbine_id][direction_id][turbine_id] = 1.0

        return J
    """


class PowWind(Component):
    """
    Integrate across the turbine to get effective wind speed
    """
    def __init__(self, nDirections, nTurbines):

        super(PowWind, self).__init__()

        # self.deriv_options['form'] = 'forward'
        # self.deriv_options['step_size'] = 500.
        # self.deriv_options['step_calc'] = 'relative'

        self.nDirections = nDirections
        self.nTurbines = nTurbines

        self.add_param('Uref', np.zeros(nDirections), units='m/s', desc='Reference speed')
        self.add_param('turbineZ', np.zeros(nTurbines), units='m', desc='Height of interest')
        self.add_param('zref', 90.0, units='m', desc='Reference height')
        self.add_param('z0', 0.0, units='m', desc='Ground height')
        self.add_param('shearExp', 0.15, desc='Shear exponent')

        self.add_output('windSpeeds', np.zeros((nTurbines, nDirections)), units='m/s', desc='Speeds of interest')

    def solve_nonlinear(self, params, unknowns, resids):
        Uref = params['Uref']
        z = params['turbineZ']
        zref = params['zref']
        z0 = params['z0']
        shearExp = params['shearExp']

        nDirections = self.nDirections
        nTurbines = self.nTurbines

        windSpeeds = np.zeros((nTurbines, nDirections))

        for turbine_id in range(nTurbines):
            for direction_id in range(nDirections):
                windSpeeds[turbine_id][direction_id] = Uref[direction_id]*((z[turbine_id]-z0)/(zref-z0))**shearExp

        # print 'PowWind: ', windSpeeds
        unknowns['windSpeeds'] = windSpeeds

    def linearize(self, params, unknowns, resids):
        Uref = params['Uref']
        z = params['turbineZ']
        zref = params['zref']
        z0 = params['z0']
        shearExp = params['shearExp']

        windSpeeds = unknowns['windSpeeds']

        nDirections = self.nDirections
        nTurbines = self.nTurbines

        J = {}

        J['windSpeeds', 'turbineZ'] = np.zeros((nTurbines*nDirections, nTurbines))
        J['windSpeeds', 'Uref'] = np.zeros((nTurbines*nDirections, nDirections))
        # for direction_id in range(nDirections):
        #     for turbine_id in range(nTurbines):
        #         for i in range(nDirections):
        #             J['windSpeeds', 'turbineZ'][turbine_id+i][i] = (Uref[i]/(zref-z0))*shearExp*((z[turbine_id]-z0)/(zref-z0))**(shearExp-1)
        #             # J['windSpeeds', 'Uref'][turbine_id][direction_id] = ((z[turbine_id]-z0)/(zref-z0))**shearExp
        for turbine_id in range(nTurbines):
            d = windSpeeds[turbine_id]*shearExp/(z[turbine_id]-z0)
            for direction_id in range(nDirections):
                J['windSpeeds', 'turbineZ'][nDirections*turbine_id+direction_id][turbine_id] = d[direction_id]

        for direction_id in range(nDirections):
            for turbine_id in range(nTurbines):
                u = windSpeeds[turbine_id][direction_id]/Uref[direction_id]
                J['windSpeeds', 'Uref'][nDirections*turbine_id+direction_id][direction_id] = u


        return J


# def PowWind(uref, z, zref, z0, a):
#     return uref*((z-z0)/(zref-z0))**a

def LnWind(uref, z, z0, z_roughness, zref):
    return uref*log((z-z0)/z_roughness)/log((zref-z0)/z_roughness)


class hGroups(Group):

    def __init__(self, nTurbs):

        super(hGroups, self).__init__()

        self.add('Hgroup_comp', Hgroup_comp(nTurbs), promotes=['*'])
        self.add('getTurbineZ', getTurbineZ(nTurbs), promotes=['*'])


class Hgroup_comp(Component):

    def __init__(self, nTurbs):

        super(Hgroup_comp, self).__init__()

        self.nTurbs = nTurbs
        self.add_param('nGroups', 1, desc='number of height groups')
        self.add_param('randomize', False, desc='randomize order of Hgroup array')

        self.add_output('hGroup', np.array(nTurbs), desc='array assigning height groups')

    def solve_nonlinear(self, params, unknowns, resids):

        nGroups = params['nGroups']
        nTurbs = self.nTurbs
        groups = np.zeros(nGroups)
        hGroup = np.zeros(nTurbs)
        for i in range(nGroups):
            groups[i] = i

        i = 0
        for j in range(nTurbs):
            hGroup[j] = groups[i]
            i += 1
            if i > nGroups-1:
                i = 0

        if params['randomize']:
            random.shuffle(hGroup)

        unknowns['hGroup'] = hGroup


class getTurbineZ(Component):

    def __init__(self, nTurbs):

        super(getTurbineZ, self).__init__()

        # self.deriv_options['form'] = 'forward'
        # self.deriv_options['step_size'] = 500.
        # self.deriv_options['step_calc'] = 'relative'
        self.nTurbs = nTurbs

        for i in range(nTurbs):
            self.add_param('turbineH%s'%i, 0.0, units='m', desc='Turbine height of each group')

        self.add_param('nGroups', 1, desc='number of height groups')
        self.add_param('hGroup', np.zeros(nTurbs), desc='An array indicating which turbines are of each height')
        self.add_output('turbineZ', np.zeros(nTurbs), units='m', desc='The array of turbine heights')


    def solve_nonlinear(self, params, unknowns, resids):
        nTurbs = self.nTurbs
        nGroups = params['nGroups']
        hGroup = params['hGroup']
        turbineZ = np.zeros(nTurbs)

        for j in range(nGroups):
            for k in range(nTurbs):
                if j == hGroup[k]:
                    turbineZ[k] = params['turbineH%s'%j]

        unknowns['turbineZ'] = turbineZ


    def linearize(self, params, unknowns, resids):
        hGroup = params['hGroup']
        nTurbs = self.nTurbs
        nGroups = params['nGroups']

        groups = np.zeros(nGroups)

        for i in range(nGroups):
            groups[i] = i

        J = {}

        for j in range(nGroups):
            J['turbineZ', 'turbineH%s'%j] = np.zeros(nTurbs)

        for k in range(nTurbs):
            for l in range(nGroups):
                if hGroup[k] == l:
                    J['turbineZ', 'turbineH%s'%l][k] = 1.

        # for i in range(nTurbs):
        #     if H1_H2[i] == 0:
        #         J['turbineZ', 'turbineH1'] = np.append(J['turbineZ', 'turbineH1'], 1)
        #         J['turbineZ', 'turbineH2'] = np.append(J['turbineZ', 'turbineH2'], 0)
        #     else:
        #         J['turbineZ', 'turbineH1'] = np.append(J['turbineZ', 'turbineH1'], 0)
        #         J['turbineZ', 'turbineH2'] = np.append(J['turbineZ', 'turbineH2'], 1)

        return J


class getRotorDiameter(Component):

    def __init__(self, nTurbs):

        super(getRotorDiameter, self).__init__()

        self.nTurbs = nTurbs

        for i in range(nTurbs):
            self.add_param('rotorDiameter%s'%i, 0.0, units='m', desc='Rotor diameter of each group')

        self.add_param('nGroups', 1, desc='number of height groups')
        self.add_param('hGroup', np.zeros(nTurbs), desc='An array indicating which turbines are in each group')
        self.add_output('rotorDiameter', np.zeros(nTurbs), units='m', desc='The array of rotor diameters')


    def solve_nonlinear(self, params, unknowns, resids):
        nTurbs = self.nTurbs
        nGroups = params['nGroups']
        hGroup = params['hGroup']
        rotorDiameter = np.zeros(nTurbs)

        for j in range(nGroups):
            for k in range(nTurbs):
                if j == hGroup[k]:
                    rotorDiameter[k] = params['rotorDiameter%s'%j]

        unknowns['rotorDiameter'] = rotorDiameter


    def linearize(self, params, unknowns, resids):
        hGroup = params['hGroup']
        nTurbs = self.nTurbs
        nGroups = params['nGroups']

        groups = np.zeros(nGroups)

        for i in range(nGroups):
            groups[i] = i

        J = {}

        for j in range(nGroups):
            J['rotorDiameter', 'rotorDiameter%s'%j] = np.zeros(nTurbs)

        for k in range(nTurbs):
            for l in range(nGroups):
                if hGroup[k] == l:
                    J['rotorDiameter', 'rotorDiameter%s'%l][k] = 1.


class getRotorCost(Component):

    def __init__(self, nGroups):

        super(getRotorCost, self).__init__()

        self.nGroups = nGroups

        for i in range(nGroups):
            self.add_param('rotorDiameter%s'%i, 0.0, units='m', desc='Rotor diameter of each group')

        self.add_output('rotorCost', np.zeros(nGroups))


    def solve_nonlinear(self, params, unknowns, resids):
        nGroups = self.nGroups
        rotorCost = np.zeros(nGroups)

        for i in range(nGroups):
            rotorCost[i] = 0.4513994169*params['rotorDiameter%s'%i]**3+21.*params['rotorDiameter%s'%i]**2-711.4813844*(params['rotorDiameter%s'%i]-70.)

        unknowns['rotorCost'] = rotorCost


    def linearize(self, params, unknowns, resids):
        nGroups = self.nGroups
        J = {}
        for i in range(nGroups):
            J['rotorCost','rotorDiameter%s'%i] = np.zeros(nGroups)

        for i in range(nGroups):
            J['rotorCost','rotorDiameter%s'%i][i] = 3.*0.4513994169*params['rotorDiameter%s'%i]**2+2.*+21.*params['rotorDiameter%s'%i]-711.4813844

        return J


class get_z(Component):

    def __init__(self, nPoints):

        super(get_z, self).__init__()

        # self.deriv_options['form'] = 'forward'
        # self.deriv_options['step_size'] = 500.
        # self.deriv_options['step_calc'] = 'relative'

        self.nPoints = nPoints
        self.add_param('turbineZ', 0.0, units='m', desc='Turbine height')

        self.add_output('z_param', np.zeros(nPoints), units='m', desc='The array of heights along the turbine')


    def solve_nonlinear(self, params, unknowns, resids):
        unknowns['z_param'] = np.linspace(0, params['turbineZ'], self.nPoints)


    def linearize(self, params, unknowns, resids):
        turbineZ = params['turbineZ']
        nPoints = self.nPoints

        grad = np.zeros(nPoints)
        for i in range(nPoints):
            grad[i] = float(i)/(nPoints-1) *1.
        J = {}
        J['z_param', 'turbineZ'] = grad

        return J


class get_z_DEL(Component):

    def __init__(self):

        super(get_z_DEL, self).__init__()

        # self.deriv_options['form'] = 'forward'
        # self.deriv_options['step_size'] = 500.
        # self.deriv_options['step_calc'] = 'relative'

        self.DEL = np.array([0.000, 1.327, 3.982, 6.636, 9.291, 11.945, 14.600, 17.255, 19.909, 22.564, 25.218, 27.873, 30.527, 33.182, 35.836, 38.491, 41.145, 43.800, 46.455, 49.109, 51.764, 54.418, 57.073, 59.727, 62.382, 65.036, 67.691, 70.345, 73.000, 75.655, 78.309, 80.964, 83.618, 86.273, 87.600])
        self.nDEL = len(self.DEL)
        self.add_param('turbineZ', 0.0, units='m', desc='Turbine height')

        self.add_output('z_DEL', np.zeros(self.nDEL))

    def solve_nonlinear(self, params, unknowns, resids):
        unknowns['z_DEL'] = self.DEL*params['turbineZ']/87.6

    def linearize(self, params, unknowns, resids):
        turbineZ = params['turbineZ']
        DEL = self.DEL
        J = {}
        J['z_DEL', 'turbineZ'] = DEL/87.6

        return J


class AEPobj(Component):
    """
    Objective to maximize AEP
    """

    def __init__(self):

        super(AEPobj, self).__init__()

        # self.deriv_options['form'] = 'forward'
        # self.deriv_options['step_size'] = 500.
        # self.deriv_options['step_calc'] = 'relative'

        self.add_param('AEP', 0.0, desc='AEP of the wind farm')
        self.add_output('AEPobj', 0.0, desc='negative AEP')

    def solve_nonlinear(self, params, unknowns, resids):
        unknowns['AEPobj'] = -1.*params['AEP']

    def linearize(Self, params, unknowns, resids):
        J = {}
        J['AEPobj', 'AEP'] = -1.

        return J


class Loads(Component):
    """
    Objective to maximize AEP
    """

    def __init__(self):

        super(Loads, self).__init__()

        self.add_param('rotor', 0.0, desc='')

        self.add_output('m', np.zeros(1), desc='')
        self.add_output('mIxx', 0., desc='')
        self.add_output('mIyy', 0., desc='')
        self.add_output('mIzz', 0., desc='')
        self.add_output('mIxy', 0., desc='')
        self.add_output('mIxz', 0., desc='')
        self.add_output('mIyz', 0., desc='')

        self.add_output('Fx1', 0., desc='')
        self.add_output('Fy1', 0., desc='')
        self.add_output('Fz1', 0., desc='')
        self.add_output('Mxx1', 0., desc='')
        self.add_output('Myy1', 0., desc='')
        self.add_output('Mzz1', 0., desc='')

        self.add_output('Fx2', 0., desc='')
        self.add_output('Fy2', 0., desc='')
        self.add_output('Fz2', 0., desc='')
        self.add_output('Mxx2', 0., desc='')
        self.add_output('Myy2', 0., desc='')
        self.add_output('Mzz2', 0., desc='')


    def solve_nonlinear(self, params, unknowns, resids):

        diameter = params['rotor']

        m = 0.06*diameter**3+11.72976061*diameter**2-407.3826374*(diameter-70.)
        unknowns['m'] = np.array([m])
        unknowns['mIxx'] = 0.00747708*diameter**5-0.6928725*diameter**4+28.82739*diameter**3-504.5127*diameter**2+3021.946*diameter
        unknowns['mIyy'] = -0.000318379*diameter**5+0.1597748*diameter**4-4.548246*diameter**3+45.46149*diameter**2-141.1889*diameter
        unknowns['mIzz'] = -0.0003929626*diameter**5+0.150502*diameter**4-3.647634*diameter**3+24.14188*diameter**2+10.99387*diameter
        unknowns['mIxy'] = 0.0
        unknowns['mIxz'] = 0.002006358*diameter**5-0.4540418*diameter**4+30.57293*diameter**3-666.8528*diameter**2+4595.203*diameter
        unknowns['mIyz'] = 0.0

        # # --- loading case 1: max Thrust ---
        unknowns['Fx1'] = 108.5329*diameter**2-3554.449*diameter
        unknowns['Fy1'] = 0.
        unknowns['Fz1'] = m*-9.81


        unknowns['Mxx1'] = -0.03850724*diameter**3+180.3352*diameter**2+9179.51*diameter
        unknowns['Myy1'] = -0.1077067*diameter**3+46.49623*diameter**2-22155.54*diameter
        unknowns['Mzz1'] = -16.37457*diameter**2-673.7798*diameter
        # # ---------------

        # # --- loading case 2: max wind speed ---
        unknowns['Fx2'] = 78.58158*diameter**2-2573.546*diameter
        unknowns['Fy2'] = 0.
        unknowns['Fz2'] = m*-9.81


        unknowns['Mxx2'] = -73.38825*diameter**2-4043.893*diameter
        unknowns['Myy2'] = -17989.42*diameter - 248612.5
        unknowns['Mzz2'] = 6.955537*diameter**2+286.1839*diameter


    def linearize(Self, params, unknowns, resids):
        J = {}
        J['AEPobj', 'AEP'] = -1.

        return J


def Weibull(x):

    a = 1.8
    f = np.zeros(len(x))
    for i in range(len(x)):
        f[i] = a*(x[i])**(a-1)*np.exp(-(x[i])**a)

    return f


def speedFreq(n):
    hi = 30.0/7.02954403167
    speedsX = np.linspace(0,hi,100*n)
    speedsDist = Weibull(speedsX)
    freq = np.zeros(n)

    for i in range(n):
        x = speedsX[i*100:(i+1)*100]
        dist = speedsDist[i*100:(i+1)*100]
        freq[i] = np.trapz(dist,x=x)

    return freq


def actualSpeeds(n, b):
    hi = 30.0/7.02954403167*b
    speeds = np.linspace(0,hi-hi/n,n)+hi/(2*n)

    return speeds


def randomStart(nTurbs, xlow, xhigh, ylow, yhigh, rotor_diameter):
    turbineX = np.zeros(nTurbs)
    turbineY = np.zeros(nTurbs)
    for i in range(nTurbs):
        good = 0
        while good == 0:
            turbineX[i] = float(np.random.rand(1))*(xhigh-xlow)+xlow
            turbineY[i] = float(np.random.rand(1))*(yhigh-ylow)+xlow

            if i == 0:
                break

            for j in range(i):
                print 'i: ', i
                print 'j: ', j
                if i == j:
                    good = 1
                else:
                    d = np.sqrt((turbineX[i]-turbineX[j])**2+(turbineY[i]-turbineY[j])**2)
                    if d > rotor_diameter:
                        good = 1
                    else:
                        good = 0
                        break

    return turbineX, turbineY



if __name__ == "__main__":
    # x = np.linspace(0,30,1000)
    # y = np.zeros(len(x))
    # y = myWeibull(x)
    # A = np.trapz(y,x,0.01)
    # print A
    # import matplotlib.pyplot as plt
    # plt.plot(x,y)
    # plt.show()

    xmin = 100.
    xmax = 1500.
    ymin = 100.
    ymax = 1500.
    rotor_diameter = 126.4
    nTurbs = 10

    turbineX, turbineY = randomStart(nTurbs, xmin, xmax, ymin, ymax, rotor_diameter)

    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig = plt.gcf()
    ax = fig.gca()

    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    color = (0,0.6,0.8)
    for j in range(nTurbs):
        ax.add_artist(Circle(xy=(turbineX[j],turbineY[j]),
                  radius=rotor_diameter/2., fill=False, edgecolor=color))

    ax.axis([xmin-100., xmax+100., ymin-100., ymax+100.]
    )

    plt.axis('off')
    plt.title('Start')
    plt.show()

    """
    import matplotlib.pyplot as plt

    AmaliaLocationsAndHull = loadmat('Amalia_locAndHull.mat')
    print AmaliaLocationsAndHull.keys()
    turbineX = AmaliaLocationsAndHull['turbineX'].flatten()
    turbineY = AmaliaLocationsAndHull['turbineY'].flatten()

    print turbineX.size

    nTurbines = len(turbineX)
    locations = np.zeros([nTurbines, 2])
    for i in range(0, nTurbines):
        locations[i] = np.array([turbineX[i], turbineY[i]])

    # get boundary information
    vertices, unit_normals = calculate_boundary(locations)

    print vertices, unit_normals

    # define point of interest
    resolution = 100
    x = np.linspace(min(turbineX), max(turbineX), resolution)
    y = np.linspace(min(turbineY), max(turbineY), resolution)
    xx, yy = np.meshgrid(x, y)
    xx = xx.flatten()
    yy = yy.flatten()
    nPoints = len(xx)
    p = np.zeros([nPoints, 2])
    for i in range(0, nPoints):
        p[i] = np.array([xx[i], yy[i]])

    # calculate distance from each point to each face
    face_distance, inside = calculate_distance(p, vertices, unit_normals, return_bool=True)

    print inside.shape
    # reshape arrays for plotting
    xx = np.reshape(xx, (resolution, resolution))
    yy = np.reshape(yy, (resolution, resolution))
    inside = np.reshape(inside, (resolution, resolution))

    # plot points colored based on inside/outside of hull
    plt.figure()
    plt.pcolor(xx, yy, inside)
    plt.plot(turbineX, turbineY, 'ow')
    plt.show()
    """

    # top = Problem()
    #
    # root = top.root = Group()
    #
    # root.add('p1', IndepVarComp('x', np.array([1.0, 1.0])))
    # root.add('p2', IndepVarComp('y', np.array([0.75, 0.25])))
    # root.add('p', WindFarmAEP(nDirections=2))
    #
    # root.connect('p1.x', 'p.power_directions')
    # root.connect('p2.y', 'p.windrose_frequencies')
    #
    # top.setup()
    # top.run()
    #
    # # should return 8760.0
    # print(root.p.unknowns['AEP'])
    # top.check_partial_derivatives()

    # top = Problem()
    #
    # root = top.root = Group()
    #
    # root.add('p1', IndepVarComp('x', 1.0))
    # root.add('p2', IndepVarComp('y', 2.0))
    # root.add('p', MUX(nElements=2))
    #
    # root.connect('p1.x', 'p.input0')
    # root.connect('p2.y', 'p.input1')
    #
    # top.setup()
    # top.run()
    #
    # # should return 8760.0
    # print(root.p.unknowns['Array'])
    # top.check_partial_derivatives()

    # top = Problem()
    #
    # root = top.root = Group()
    #
    # root.add('p1', IndepVarComp('x', np.zeros(2)))
    # root.add('p', DeMUX(nElements=2))
    #
    # root.connect('p1.x', 'p.Array')
    #
    # top.setup()
    # top.run()
    #
    # # should return 8760.0
    # print(root.p.unknowns['output0'])
    # print(root.p.unknowns['output1'])
    # top.check_partial_derivatives()

    # top = Problem()
    #
    # root = top.root = Group()
    #
    # root.add('p1', IndepVarComp('x', np.array([0, 3])))
    # root.add('p2', IndepVarComp('y', np.array([1, 0])))
    # root.add('p', SpacingComp(nTurbines=2))
    #
    # root.connect('p1.x', 'p.turbineX')
    # root.connect('p2.y', 'p.turbineY')
    #
    # top.setup()
    # top.run()
    #
    # # print(root.p.unknowns['output0'])
    # # print(root.p.unknowns['output1'])
    # top.check_partial_derivatives()
