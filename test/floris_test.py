from __future__ import print_function, division, absolute_import

import numpy as np

import openmdao.api as om

from florisse.floris import Floris as floris


if __name__ == '__main__':

    # Setup the turbine locations
    rotor_diameter = 126.4
    turbineX = np.array([0., 200., 400., 600.])
    turbineY = np.array([0., 0., 0., 0.])
    turbineZ = np.array([100., 150., 45., 55.])

    nTurbs = len(turbineX)

    rotorDiameter = np.ones(nTurbs) * 100.

    nDirections = 1

    shearExp = 0.15
    z0 = 0.
    zref = 50.
    Uref = 8.

    windArray = np.zeros(nTurbs)
    for turbine_id in range(nTurbs):
        windArray[turbine_id] = Uref*((turbineZ[turbine_id]-z0)/(zref-z0))**shearExp

    # OpenMDAO
    prob = om.Problem()
    model = prob.model

    floris_comp = floris(nTurbines=nTurbs, differentiable=True, use_rotor_components=False, nSamples=0,
                         verbose=False)
    model.add_subsystem('floris', floris_comp,
                        promotes=['turbineXw','turbineYw','hubHeight','rotorDiameter','wind_speed',
                                  'floris_params:shearExp','floris_params:z_ref'])

    prob.setup(check=True)

    prob['turbineXw'] = turbineX
    prob['turbineYw'] = turbineY
    prob['hubHeight'] = turbineZ
    prob['floris_params:z_ref'] = zref
    prob['floris_params:shearExp'] = shearExp

    prob.run_model()

    correct = np.array([8.87655578, 6.58018746, 6.02151567, 3.63604448])
    print('Answer should be:', correct)
    print(prob['floris.wtVelocity0'])
