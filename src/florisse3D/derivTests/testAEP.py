import unittest
from openmdao.api import pyOptSparseDriver, ExecComp, IndepVarComp, Problem
from FLORISSE3D.setupOptimization import *
from FLORISSE3D.floris import *
from FLORISSE3D.GeneralWindFarmComponents import calculate_boundary, SpacingComp,\
            BoundaryComp, get_z, getTurbineZ, AEPobj, DeMUX, hGroups, randomStart,\
            getRotorDiameter
import numpy as np


class test_AEPgroup(unittest.TestCase):

    def setUp(self):

        self.rtol = 1.E-3
        self.atol = 1.E-3

        use_rotor_components = False
        datasize = 0
        rotor_diameter = 126.4

        nRows = 4
        nTurbs = nRows**2
        spacing = 5.

        """Define wind flow"""
        air_density = 1.1716    # kg/m^3

        # windSpeeds, windFrequencies, windDirections, nDirections = amaliaWind()

        windSpeeds = np.array([25.])
        windFrequencies = np.array([1.])
        windDirections = np.array([0.])
        nDirections = 1

        axialInduction = np.zeros(nTurbs)
        Ct = np.zeros(nTurbs)
        Cp = np.zeros(nTurbs)
        generatorEfficiency = np.zeros(nTurbs)
        yaw = np.zeros((nDirections, nTurbs))

        # define initial values
        for turbI in range(0, nTurbs):
            axialInduction[turbI] = 1.0/3.0
            Ct[turbI] = 4.0*axialInduction[turbI]*(1.0-axialInduction[turbI])
            # Cp[turbI] = 0.7737/0.944 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
            Cp[turbI] = 0.7737 * 4.0 * 1.0/3.0 * np.power((1. - 1.0/3.0), 2)
            generatorEfficiency[turbI] = 1.0#0.944

        minSpacing = 2.0

        points = np.linspace(start=spacing*rotor_diameter, stop=nRows*spacing*rotor_diameter, num=nRows)
        xpoints, ypoints = np.meshgrid(points, points)
        turbineX_bounds = np.ndarray.flatten(xpoints)
        turbineY_bounds = np.ndarray.flatten(ypoints)

        # generate boundary constraint
        locations = np.zeros((len(turbineX_bounds),2))
        for i in range(len(turbineX_bounds)):
            locations[i][0] = turbineX_bounds[i]
            locations[i][1] = turbineY_bounds[i]
        # print locations
        boundaryVertices, boundaryNormals = calculate_boundary(locations)
        nVertices = boundaryVertices.shape[0]

        turbineX = np.random.rand(nTurbs)*(max(turbineX_bounds)-min(turbineX_bounds))+min(turbineX_bounds)
        turbineY = np.random.rand(nTurbs)*(max(turbineY_bounds)-min(turbineY_bounds))+min(turbineY_bounds)

        shearExp = 0.08

        rotorDiameter = np.random.rand(nTurbs)*120.*40.
        turbineZ = np.random.rand(nTurbs)*120.*40.
        turbineRating = np.random.rand(nTurbs)*9000.+1000.
        """OpenMDAO"""

        prob = Problem()
        root = prob.root = Group()

        # for i in range(nGroups):
        #     root.add('turbineH%s'%i, IndepVarComp('turbineH%s'%i, float(turbineZ[i])), promotes=['*'])
        #     root.add('rotorDiameter%s'%i, IndepVarComp('rotorDiameter%s'%i, float(rotorDiameter[i])), promotes=['*'])

        root.add('turbineZ', IndepVarComp('turbineZ', turbineZ), promotes=['*'])
        root.add('rotorDiameter', IndepVarComp('rotorDiameter', rotorDiameter), promotes=['*'])
        root.add('ratedPower', IndepVarComp('ratedPower', turbineRating), promotes=['*'])

        root.add('AEPGroup', AEPGroup(nTurbs, nDirections=nDirections,
                    use_rotor_components=False, datasize=datasize, differentiable=True,
                    optimizingLayout=False, nSamples=0), promotes=['*'])


        # set up optimizer
        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'SNOPT'
        prob.driver.add_objective('AEP')

        # select design variables
        prob.driver.add_desvar('turbineX')
        prob.driver.add_desvar('turbineY')
        prob.driver.add_desvar('turbineZ')
        prob.driver.add_desvar('rotorDiameter')
        prob.driver.add_desvar('ratedPower')

        # root.connect('rotorDiameter','getRating.rotorDiameter')
        # root.connect('getRating.ratedPower','ratedPower')

        # root.add('spacing_comp', SpacingComp(nTurbines=nTurbs), promotes=['*'])
        #
        # # add constraint definitions
        # root.add('spacing_con', ExecComp('sc = wtSeparationSquared-(minSpacing*rotorDiameter[0])**2',
        #                              minSpacing=minSpacing, rotorDiameter=np.zeros(nTurbs),
        #                              sc=np.zeros(((nTurbs-1)*nTurbs/2)),
        #                              wtSeparationSquared=np.zeros(((nTurbs-1)*nTurbs/2))),
        #                              promotes=['*'])

        # if nVertices > 0:
        #     # add component that enforces a convex hull wind farm boundary
        #     root.add('boundary_con', BoundaryComp(nVertices=nVertices, nTurbines=nTurbs), promotes=['*'])

        prob.root.ln_solver.options['single_voi_relevance_reduction'] = True

        prob.setup(check=True)

        for i in range(nDirections):
            prob['yaw%s'%i] = yaw[i]

        prob['turbineX'] = turbineX
        prob['turbineY'] = turbineY

        # prob['ratedPower'] = np.ones(nTurbs)*2098.77 # in kw

        # prob['boundaryVertices'] = boundaryVertices
        # prob['boundaryNormals'] = boundaryNormals

        # assign values to constant inputs (not design variables)
        # prob['rotorDiameter'] = rotorDiameter
        # prob['rotor_diameter'] = rotor_diameter
        prob['axialInduction'] = axialInduction
        prob['generatorEfficiency'] = generatorEfficiency
        prob['air_density'] = air_density
        prob['windDirections'] = windDirections
        prob['windFrequencies'] = windFrequencies
        prob['Uref'] = windSpeeds

        prob['Ct_in'] = Ct
        prob['Cp_in'] = Cp
        prob['floris_params:cos_spread'] = 1E12 # turns off cosine spread (just needs to be very large)


        prob['shearExp'] = shearExp
        prob['zref'] = 50.
        prob['z0'] = 0.

        prob['turbineX'] = turbineX
        prob['turbineY'] = turbineY

        prob.run_once()

        self.J = prob.check_total_derivatives(out_stream=None)

        print prob['AEP']

        print 'REV'
        print self.J[('AEP', 'ratedPower')]['J_rev']

        print 'FD'
        print self.J[('AEP', 'ratedPower')]['J_fd']


    def test_rated(self):
        np.testing.assert_allclose(self.J[('AEP', 'ratedPower')]['J_rev'], self.J[('AEP', 'ratedPower')]['J_fd'], self.rtol, self.atol)

    # def test_X(self):
    #     np.testing.assert_allclose(self.J[('AEP', 'turbineX')]['J_rev'], self.J[('AEP', 'turbineX')]['J_fd'], self.rtol, self.atol)
    #
    # def test_Y(self):
    #     np.testing.assert_allclose(self.J[('AEP', 'turbineY')]['J_rev'], self.J[('AEP', 'turbineY')]['J_fd'], self.rtol, self.atol)

    # def test_Z(self):
    #     np.testing.assert_allclose(self.J[('AEP', 'turbineZ')]['J_rev'], self.J[('AEP', 'turbineZ')]['J_fd'], self.rtol, self.atol)

    # def test_D(self):
    #     np.testing.assert_allclose(self.J[('AEP', 'rotorDiameter')]['J_rev'], self.J[('AEP', 'rotorDiameter')]['J_fd'], self.rtol, self.atol)

if __name__ == "__main__":
    unittest.main()
