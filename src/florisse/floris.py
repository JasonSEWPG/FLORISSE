from __future__ import print_function, division, absolute_import

import numpy as np

import openmdao.api as om

import _floris
import _florisDiscontinuous


def add_floris_parameters(comp, use_rotor_components=False):
    # altering the values in this function will have no effect during optimization. To change defaults permanently,
    # alter the values in add_floris_IndepVarComps().

    # ##################   wake deflection   ##################

    # ## parameters
    # original model
    comp.add_discrete_input('floris_params:kd', 0.15 if not use_rotor_components else 0.17,
                            desc='model parameter that defines the sensitivity of the wake deflection to yaw')
    comp.add_discrete_input('floris_params:initialWakeDisplacement', -4.5,
                            desc='defines the wake at the rotor to be slightly offset from the rotor. This is'
                            'necessary for tuning purposes')
    comp.add_discrete_input('floris_params:bd', -0.01,
                            desc='defines rate of wake displacement if initialWakeAngle is not used')
    # added
    comp.add_discrete_input('floris_params:initialWakeAngle', 1.5,
                            desc='sets how angled the wake flow should be at the rotor')

    # ## flags
    comp.add_discrete_input('floris_params:useWakeAngle', False if not use_rotor_components else True,
                            desc='define whether an initial angle or initial offset should be used for wake center. '
                            'if True, then bd will be ignored and initialWakeAngle will'
                            'be used. The reverse is also true')

    # ##################   wake expansion   ##################

    # ## parameters
    # original model
    comp.add_discrete_input('floris_params:ke', 0.065 if not use_rotor_components else 0.05,
                            desc='parameter defining overall wake expansion')
    comp.add_discrete_input('floris_params:me', np.array([-0.5, 0.22, 1.0]) if not use_rotor_components else np.array([-0.5, 0.3, 1.0]),
                            desc='parameters defining relative zone expansion. Mixing zone (me[2]) must always be 1.0')

    # ## flags
    comp.add_discrete_input('floris_params:adjustInitialWakeDiamToYaw', False if not use_rotor_components else True,
                            desc='if True then initial wake diameter will be set to rotorDiameter*cos(yaw)')

    # ##################   wake velocity   ##################

    # ## parameters
    # original model
    comp.add_discrete_input('floris_params:MU', np.array([0.5, 1.0, 5.5]),
                            desc='velocity deficit decay rates for each zone. Middle zone must always be 1.0')
    comp.add_discrete_input('floris_params:aU', 5.0 if not use_rotor_components else 12.0,
                            desc='zone decay adjustment parameter independent of yaw (deg)')
    comp.add_discrete_input('floris_params:bU', 1.66 if not use_rotor_components else 1.3,
                            desc='zone decay adjustment parameter dependent yaw')
    # added
    comp.add_discrete_input('floris_params:cos_spread', 2.0,
                            desc='spread of cosine smoothing factor (multiple of sum of wake and rotor radii)')
    comp.add_discrete_input('floris_params:keCorrArray', 0.0,
                            desc='multiplies the ke value by 1+keCorrArray*(sum of rotors relative overlap with '
                                 'inner two zones for including array affects')
    comp.add_discrete_input('floris_params:keCorrCT', 0.0,
                            desc='adjust ke by adding a precentage of the difference of CT and ideal CT as defined in'
                                 'Region2CT')
    comp.add_discrete_input('floris_params:Region2CT', 4.0*(1.0/3.0)*(1.0-(1.0/3.0)),
                            desc='defines ideal CT value for use in adjusting ke to yaw adjust CT if keCorrCT>0.0')

    # flags
    comp.add_discrete_input('floris_params:axialIndProvided', True if not use_rotor_components else False,
                            desc='if axial induction is not provided, then it will be calculated based on CT')
    comp.add_discrete_input('floris_params:useaUbU', True,
                            desc='if True then zone velocity decay rates (MU) will be adjusted based on yaw')

    # ################   Visualization   ###########################
    # shear layer (only influences visualization)
    comp.add_discrete_input('floris_params:shearCoefficientAlpha', 0.10805)
    comp.add_discrete_input('floris_params:shearZh', 90.0)

    # ##################   other   ##################
    comp.add_discrete_input('floris_params:FLORISoriginal', False,
                            desc='override all parameters and use FLORIS as original in first Wind Energy paper')


def add_floris_params_IndepVarComps(openmdao_group, use_rotor_components=False):
    ivc = openmdao_group.add_subsystem('floris_params', om.IndepVarComp(), promotes_outputs=['*'])

    # permanently alter defaults here

    # ##################   wake deflection   ##################

    # ## parameters
    # original model

    ivc.add_discrete_output('floris_params:kd', 0.15 if not use_rotor_components else 0.17,
                            desc='model parameter that defines the sensitivity of the wake deflection '
                            'to yaw')

    ivc.add_discrete_output('floris_params:initialWakeDisplacement', -4.5,
                            desc='defines the wake at the rotor to be slightly offset from the rotor. '
                            'This is necessary for tuning purposes')

    ivc.add_discrete_output('floris_params:bd', -0.01,
                            desc='defines rate of wake displacement if initialWakeAngle is not used')

    # added
    ivc.add_discrete_output('floris_params:initialWakeAngle', 1.5,
                            desc='sets how angled the wake flow should be at the rotor')

    # ## flags
    ivc.add_discrete_output('floris_params:useWakeAngle', False if not use_rotor_components else True,
                            desc='define whether an initial angle or initial offset should be used for'
                            'wake center. If True, then bd will be ignored and initialWakeAngle '
                            'will be used. The reverse is also true')

    # ##################   wake expansion   ##################

    # ## parameters
    # original model
    ivc.add_discrete_output('floris_params:ke', 0.065 if not use_rotor_components else 0.05,
                            desc='parameter defining overall wake expansion')

    ivc.add_discrete_output('floris_params:me', np.array([-0.5, 0.22, 1.0]) if not use_rotor_components else np.array([-0.5, 0.3, 1.0]),
                            desc='parameters defining relative zone expansion. Mixing zone (me[2]) '
                            'must always be 1.0')

    # ## flags
    ivc.add_discrete_output('floris_params:adjustInitialWakeDiamToYaw', False,
                            desc='if True then initial wake diameter will be set to '
                            'rotorDiameter*cos(yaw)')

    # ##################   wake velocity   ##################

    # ## parameters
    # original model
    ivc.add_discrete_output('floris_params:MU', np.array([0.5, 1.0, 5.5]),
                            desc='velocity deficit decay rates for each zone. Middle zone must always '
                            'be 1.0')

    ivc.add_discrete_output('floris_params:aU', 5.0 if not use_rotor_components else 12.0,
                            desc='zone decay adjustment parameter independent of yaw (deg)')

    ivc.add_discrete_output('floris_params:bU', 1.66 if not use_rotor_components else 1.3,
                            desc='zone decay adjustment parameter dependent yaw')

    # added
    ivc.add_discrete_output('floris_params:cos_spread', 2.0,
                            desc='spread of cosine smoothing factor (multiple of sum of wake and '
                            'rotor radii)')

    ivc.add_discrete_output('floris_params:keCorrArray', 0.0,
                            desc='multiplies the ke value by 1+keCorrArray*(sum of rotors relative '
                            'overlap with inner two zones for including array affects')

    ivc.add_discrete_output('floris_params:keCorrCT', 0.0,
                            desc='adjust ke by adding a precentage of the difference of CT and ideal '
                            'CT as defined in Region2CT')

    ivc.add_discrete_output('floris_params:Region2CT', 4.0*(1.0/3.0)*(1.0-(1.0/3.0)),
                            desc='defines ideal CT value for use in adjusting ke to yaw adjust CT if '
                            'keCorrCT>0.0')

    # flags
    ivc.add_discrete_output('floris_params:axialIndProvided',
                            True if not use_rotor_components else False,
                            desc='if axial induction is not provided, then it will be calculated based '
                            'on CT')

    ivc.add_discrete_output('floris_params:useaUbU', True,
                            desc='if True then zone velocity decay rates (MU) will be adjusted based '
                            'on yaw')

    # ################   Visualization   ###########################
    # shear layer (only influences visualization)
    ivc.add_discrete_output('floris_params:shearCoefficientAlpha', 0.10805)

    ivc.add_discrete_output('floris_params:shearZh', 90.0)

    # ##################   other   ##################
    # this is currently not used. Defaults to original if use_rotor_components=False
    ivc.add_discrete_output('floris_params:FLORISoriginal', False,
                            desc='override all parameters and use FLORIS as original in Gebraad et al.'
                            '2014, Wind plant power optimization through yaw control using a '
                            'parametric model for wake effect-a CFD simulation study')


class Floris(om.ExplicitComponent):
    """
    OpenMDAO 2.x component wrapper for FLOw Redirection and Induction in Steady-state (FLORIS) wind
    farm wake model.
    """

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('nTurbines', types=int, default=0,
                             desc="Number of wind turbines.")
        self.options.declare('direction_id', types=int, default=0,
                             desc="Direction index.")
        self.options.declare('differentiable', types=bool, default=True,
                             desc="Differentiation flag, set to False to force finite difference.")
        self.options.declare('use_rotor_components', types=bool, default=False,
                             desc="Set to True to use rotor components.")
        self.options.declare('nSamples', types=int, default=0,
                             desc="Number of samples for the visualization arrays.")
        self.options.declare('verbose', types=bool, default=False,
                             desc="Set to True for some debug printing.")

    def setup(self):
        """
        Set up the Floris component.
        """
        opt = self.options
        direction_id = opt['direction_id']
        differentiable = opt['differentiable']
        nTurbines = opt['nTurbines']
        verbose = opt['verbose']
        nSamples = opt['nSamples']
        use_rotor_components = opt['use_rotor_components']

        # FLORIS parameters
        add_floris_parameters(self, use_rotor_components=use_rotor_components)

        # input arrays
        self.add_input('turbineXw', np.zeros(nTurbines), units='m',
                       desc='x coordinates of turbines in wind dir. ref. frame')
        self.add_input('turbineYw', np.zeros(nTurbines), units='m',
                       desc='y coordinates of turbines in wind dir. ref. frame')
        self.add_input('yaw%i' % direction_id, np.zeros(nTurbines), units='deg',
                       desc='yaw of each turbine wrt wind dir.')
        self.add_input('hubHeight', np.zeros(nTurbines), units='m',
                       desc='hub heights of all turbines')
        self.add_input('rotorDiameter', np.zeros(nTurbines) + 126.4, units='m', desc='rotor diameter of each turbine')
        self.add_input('Ct', np.zeros(nTurbines)+4.0*(1./3.)*(1.0-(1./3.)), desc='thrust coefficient of each turbine')
        self.add_input('wind_speed', 8.0, units='m/s', desc='free stream wind velocity')
        self.add_input('axialInduction', np.zeros(nTurbines)+1./3., desc='axial induction of all turbines')

        self.add_input('floris_params:shearExp', 0.15, desc='wind shear exponent')
        self.add_input('floris_params:z_ref', 90., units='m', desc='height at which wind_speed is measured')
        self.add_input('floris_params:z0', 0., units='m', desc='ground height')

        # output arrays
        self.add_output('wtVelocity%i' % direction_id, val=np.zeros(nTurbines), units='m/s',
                        desc='effective hub velocity for each turbine')
        self.add_output('wakeCentersYT', np.zeros(nTurbines*nTurbines), units='m',
                        desc='wake center y position at each turbine')
        self.add_output('wakeCentersZT', np.zeros(nTurbines*nTurbines), units='m',
                        desc='wake center z position at each turbine')
        self.add_output('wakeDiametersT', np.zeros(3*nTurbines*nTurbines), units='m',
                        desc='wake diameter of each zone of each wake at each turbine')
        self.add_output('wakeOverlapTRel', np.zeros(3*nTurbines*nTurbines),
                        desc='relative wake zone overlap to rotor area')

        # ############################ visualization arrays ##################################
        if nSamples > 0:
            # visualization input
            self.add_discrete_input('wsPositionXw', np.zeros(nSamples),
                           desc='downwind position of desired measurements in wind ref. frame (m)')
            self.add_discrete_input('wsPositionYw', np.zeros(nSamples),
                           desc='crosswind position of desired measurements in wind ref. frame (m)')
            self.add_discrete_input('wsPositionZ', np.zeros(nSamples),
                           desc='position of desired measurements in wind ref. frame (m)')

            # visualization output
            self.add_discrete_output('wsArray%i' % direction_id, np.zeros(nSamples),
                                     desc='wind speed at measurement locations (m/s)')

        # Derivatives
        if differentiable:
            self.declare_partials(of='*', wrt='*')

        else:
            self.declare_partials(of='*', wrt='*', method='fd', form='forward', step=1.0e-6,
                                  step_calc='rel')

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        opt = self.options
        direction_id = opt['direction_id']
        differentiable = opt['differentiable']
        verbose = opt['verbose']
        nSamples = opt['nSamples']

        # x and y positions w.r.t. the wind direction (wind dir. = +x)
        turbineXw = inputs['turbineXw']
        turbineYw = inputs['turbineYw']
        turbineZ = inputs['hubHeight']

        # yaw wrt wind dir.
        yawDeg = inputs['yaw%i' % direction_id]

        # turbine specs
        rotorDiameter = inputs['rotorDiameter']

        # air flow
        Vinf = inputs['wind_speed']
        Ct = inputs['Ct']
        shearExp = inputs['floris_params:shearExp']
        zref = inputs['floris_params:z_ref']
        z0 = inputs['floris_params:z0']

        # wake deflection
        kd = discrete_inputs['floris_params:kd']
        bd = discrete_inputs['floris_params:bd']
        initialWakeDisplacement = discrete_inputs['floris_params:initialWakeDisplacement']
        useWakeAngle = discrete_inputs['floris_params:useWakeAngle']
        initialWakeAngle = discrete_inputs['floris_params:initialWakeAngle']

        # wake expansion
        ke = discrete_inputs['floris_params:ke']
        adjustInitialWakeDiamToYaw = discrete_inputs['floris_params:adjustInitialWakeDiamToYaw']

        # velocity deficit
        MU = discrete_inputs['floris_params:MU']
        useaUbU = discrete_inputs['floris_params:useaUbU']
        aU = discrete_inputs['floris_params:aU']
        bU = discrete_inputs['floris_params:bU']
        me = discrete_inputs['floris_params:me']
        cos_spread = discrete_inputs['floris_params:cos_spread']

        # logicals
        Region2CT = discrete_inputs['floris_params:Region2CT']
        axialInduction = inputs['axialInduction']
        keCorrCT = discrete_inputs['floris_params:keCorrCT']
        keCorrArray = discrete_inputs['floris_params:keCorrArray']
        axialIndProvided = discrete_inputs['floris_params:axialIndProvided']

        # visualization
        # shear layer (only influences visualization)
        shearCoefficientAlpha = discrete_inputs['floris_params:shearCoefficientAlpha']
        shearZh = discrete_inputs['floris_params:shearZh']

        if nSamples > 0:
            wsPositionXYZw = np.array([discrete_inputs['wsPositionXw'],
                                       discrete_inputs['wsPositionYw'],
                                       discrete_inputs['wsPositionZ']])
        else:
            wsPositionXYZw = np.zeros([3, 1])

        # print option
        #if verbose:
        #    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        #    print "free-stream wind speed %s" % Vinf
        #    print "axial induction of turbines %s" % axialInduction
        #    print "C_T of turbines %s" % Ct
        #    print "yaw of turbines %s" % yawDeg

        if differentiable:
            # call to fortran code to obtain output values
            wtVelocity, wsArray, wakeCentersYT, wakeCentersZT, wakeDiametersT, wakeOverlapTRel = \
                _floris.floris(turbineXw, turbineYw, turbineZ, yawDeg, rotorDiameter, Vinf, shearExp, zref, z0,
                                               Ct, axialInduction, ke, kd, me, initialWakeDisplacement, bd,
                                               MU, aU, bU, initialWakeAngle, cos_spread, keCorrCT,
                                               Region2CT, keCorrArray, useWakeAngle,
                                               adjustInitialWakeDiamToYaw, axialIndProvided, useaUbU, wsPositionXYZw,
                                               shearCoefficientAlpha, shearZh)

            outputs['wakeCentersZT'] = wakeCentersZT

        else:
            # call to fortran code to obtain output values
            # print rotorDiameter.shape
            wtVelocity, wakeCentersYT, wakeDiametersT, wakeOverlapTRel = \
                _florisDiscontinuous.floris(turbineXw, turbineYw, yawDeg, rotorDiameter, Vinf,
                                                           Ct, axialInduction, ke, kd, me, initialWakeDisplacement, bd,
                                                           MU, aU, bU, initialWakeAngle, keCorrCT,
                                                           Region2CT, keCorrArray, useWakeAngle,
                                                           adjustInitialWakeDiamToYaw, axialIndProvided, useaUbU)

        # pass outputs to self
        outputs['wtVelocity%i' % direction_id] = wtVelocity
        outputs['wakeCentersYT'] = wakeCentersYT
        outputs['wakeDiametersT'] = wakeDiametersT
        outputs['wakeOverlapTRel'] = wakeOverlapTRel
        if nSamples > 0:
            discrete_outputs['wsArray%i' % direction_id] = wsArray

    def compute_partials(self, inputs, partials, discrete_inputs):
        opt = self.options
        direction_id = opt['direction_id']
        nTurbines = opt['nTurbines']

        # x and y positions w.r.t. the wind dir. (wind dir. = +x)
        turbineXw = inputs['turbineXw']
        turbineYw = inputs['turbineYw']
        turbineZ = inputs['hubHeight']

        # yaw wrt wind dir. (wind dir. = +x)
        yawDeg = inputs['yaw%i' % direction_id]

        # turbine specs
        rotorDiameter = inputs['rotorDiameter']

        # air flow
        Vinf = inputs['wind_speed']
        Ct = inputs['Ct']
        shearExp = inputs['floris_params:shearExp']
        zref = inputs['floris_params:z_ref']
        z0 = inputs['floris_params:z0']

        # wake deflection
        kd = discrete_inputs['floris_params:kd']
        bd = discrete_inputs['floris_params:bd']
        initialWakeDisplacement = discrete_inputs['floris_params:initialWakeDisplacement']
        useWakeAngle = discrete_inputs['floris_params:useWakeAngle']
        initialWakeAngle = discrete_inputs['floris_params:initialWakeAngle']

        # wake expansion
        ke = discrete_inputs['floris_params:ke']
        adjustInitialWakeDiamToYaw = discrete_inputs['floris_params:adjustInitialWakeDiamToYaw']

        # velocity deficit
        MU = discrete_inputs['floris_params:MU']
        useaUbU = discrete_inputs['floris_params:useaUbU']
        aU = discrete_inputs['floris_params:aU']
        bU = discrete_inputs['floris_params:bU']
        me = discrete_inputs['floris_params:me']
        cos_spread = discrete_inputs['floris_params:cos_spread']

        # logicals
        Region2CT = discrete_inputs['floris_params:Region2CT']
        axialInduction = inputs['axialInduction']
        keCorrCT = discrete_inputs['floris_params:keCorrCT']
        keCorrArray = discrete_inputs['floris_params:keCorrArray']
        axialIndProvided = discrete_inputs['floris_params:axialIndProvided']

        # define jacobian size
        nDirs = nTurbines

        # define input array to direct differentiation
        wtVelocityb = np.eye(nDirs, nTurbines)

        # call to fortran code to obtain output values
        turbineXwb, turbineYwb, turbineZb, yawDegb, rotorDiameterb, Vinfb, Ctb, axialInductionb = \
                        _floris.floris_bv(turbineXw, turbineYw, turbineZ, yawDeg, rotorDiameter, Vinf, shearExp, zref, z0,
                                                         Ct, axialInduction, ke, kd, me, initialWakeDisplacement, bd,
                                                         MU, aU, bU, initialWakeAngle, cos_spread, keCorrCT,
                                                         Region2CT, keCorrArray, useWakeAngle,
                                                         adjustInitialWakeDiamToYaw, axialIndProvided, useaUbU,
                                                         wtVelocityb)

        # collect values of the Jacobian
        partials['wtVelocity%i' % direction_id, 'turbineXw'] = turbineXwb
        partials['wtVelocity%i' % direction_id, 'turbineYw'] = turbineYwb
        partials['wtVelocity%i' % direction_id, 'hubHeight'] = turbineZb
        partials['wtVelocity%i' % direction_id, 'yaw%i' % direction_id] = yawDegb
        partials['wtVelocity%i' % direction_id, 'rotorDiameter'] = rotorDiameterb
        partials['wtVelocity%i' % direction_id, 'wind_speed'] = Vinfb
        partials['wtVelocity%i' % direction_id, 'Ct'] = Ctb
        partials['wtVelocity%i' % direction_id, 'axialInduction'] = axialInductionb
