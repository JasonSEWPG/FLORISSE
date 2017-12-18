from openmdao.api import Component, Group, Problem, IndepVarComp
import numpy as np
from scipy.interpolate import RectBivariateSpline, LinearNDInterpolator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class SimpleRotorSE(Component):
    def __init__(self):

        super(SimpleRotorSE, self).__init__()

        self.add_param('turbineRating', 0.0, units='kW', desc='turbine rating (kW)')
        self.add_param('rotorDiameter', 0.0, units='m', desc='rotor diameter (m)')

        self.add_output('ratedT', 0., desc='rated thrust')
        self.add_output('ratedQ', 0., desc='rated torque')
        self.add_output('blade_mass', 0., desc='blade mass')
        self.add_output('Vrated', 0., desc='rated wind speed')
        self.add_output('extremeT', 0., desc='extreme thrust')
        self.add_output('I', np.zeros(6), desc='I all blades')

    def solve_nonlinear(self, params, unknowns, resids):
        # # global ratedT_func
        # # global ratedQ_func
        # # global blade_mass_func
        # # global Vrated_func
        # # global extremeT_func
        #
        # num = 100
        # x = np.linspace(500.,10000.,num)
        # y = np.linspace(25.,160.,num)
        #
        # filename = 'src/florisse3D/optRotor/ratedT100.txt'
        # openedFile = open(filename)
        # loadedData = np.loadtxt(openedFile)
        # ratedT = loadedData[:]
        #
        # filename = 'src/florisse3D/optRotor/ratedQ100.txt'
        # openedFile = open(filename)
        # loadedData = np.loadtxt(openedFile)
        # ratedQ = loadedData[:]
        #
        # filename = 'src/florisse3D/optRotor/blade_mass100.txt'
        # openedFile = open(filename)
        # loadedData = np.loadtxt(openedFile)
        # blade_mass = loadedData[:]
        #
        # filename = 'src/florisse3D/optRotor/Vrated100.txt'
        # openedFile = open(filename)
        # loadedData = np.loadtxt(openedFile)
        # Vrated = loadedData[:]
        #
        # filename = 'src/florisse3D/optRotor/extremeT100.txt'
        # openedFile = open(filename)
        # loadedData = np.loadtxt(openedFile)
        # extremeT = loadedData[:]
        #
        # filename = 'src/florisse3D/optRotor/I1_100.txt'
        # openedFile = open(filename)
        # loadedData = np.loadtxt(openedFile)
        # I1 = loadedData[:]
        #
        # filename = 'src/florisse3D/optRotor/I2_100.txt'
        # openedFile = open(filename)
        # loadedData = np.loadtxt(openedFile)
        # I2 = loadedData[:]
        #
        # filename = 'src/florisse3D/optRotor/I3_100.txt'
        # openedFile = open(filename)
        # loadedData = np.loadtxt(openedFile)
        # I3 = loadedData[:]

        # ratedT_func = RectBivariateSpline(x, y, ratedT)
        # ratedQ_func = RectBivariateSpline(x, y, ratedQ)
        # blade_mass_func = RectBivariateSpline(x, y, blade_mass)
        # Vrated_func = RectBivariateSpline(x, y, Vrated)
        # extremeT_func = RectBivariateSpline(x, y, extremeT)
        # I1_func = RectBivariateSpline(x, y, I1)
        # I2_func = RectBivariateSpline(x, y, I2)
        # I3_func = RectBivariateSpline(x, y, I3)

        filename = 'src/florisse3D/optRotor/rotor/OPTIMIZED.txt'
        opedRatedT = open(filename)
        ratedTdata = np.loadtxt(opedRatedT)
        "ratedPower, rotorDiameter, ratedQ, blade_mass, Vrated, I1, I2, I3, ratedT, extremeT"
        ratedPower = ratedTdata[:,0]
        rotorDiameter = ratedTdata[:,1]
        ratedQ = ratedTdata[:,2]
        blade_mass = ratedTdata[:,3]
        Vrated = ratedTdata[:,4]
        I1 = ratedTdata[:,5]
        I2 = ratedTdata[:,6]
        I3 = ratedTdata[:,7]
        ratedT = ratedTdata[:,8]
        extremeT = ratedTdata[:,9]

        cartcoord = list(zip(ratedPower,rotorDiameter))
        interp_spline = LinearNDInterpolator(cartcoord,Vrated)

        ratedT_func = LinearNDInterpolator(cartcoord, ratedT)
        ratedQ_func = LinearNDInterpolator(cartcoord, ratedQ)
        blade_mass_func = LinearNDInterpolator(cartcoord, blade_mass)
        Vrated_func = LinearNDInterpolator(cartcoord, Vrated)
        extremeT_func = LinearNDInterpolator(cartcoord, extremeT)
        I1_func = LinearNDInterpolator(cartcoord, I1)
        I2_func = LinearNDInterpolator(cartcoord, I2)
        I3_func = LinearNDInterpolator(cartcoord, I3)




        rating = params['turbineRating']
        diam = params['rotorDiameter']

        # print 'IN COMP: RATING: ', rating
        # print 'IN COMP: DIAMETER: ', diam

        unknowns['ratedT'] = ratedT_func(rating,diam)
        unknowns['ratedQ'] = ratedQ_func(rating,diam)
        unknowns['blade_mass'] = blade_mass_func(rating,diam)
        unknowns['Vrated'] = Vrated_func(rating,diam)
        unknowns['extremeT'] = extremeT_func(rating,diam)
        unknowns['I'] = np.array([I1_func(rating,diam),I2_func(rating,diam),I3_func(rating,diam),0.,0.,0.])
