from openmdao.api import Component, Group, Problem, IndepVarComp
import numpy as np
from scipy.optimize import fsolve
from math import sin, cos, sinh, cosh, sqrt, pi
from scipy.optimize import root
import _shellbuckling

class calcMass(Component):
    """
    Calculate the mass of the cylinder tower
    """

    def __init__(self):

        super(calcMass, self).__init__()

        # self.deriv_options['form'] = 'forward'
        # self.deriv_options['step_size'] = 1.0E-6
        # self.deriv_options['step_calc'] = 'absolute'

        self.add_param('turbineH', 0.0, desc='Height of the Tower')
        self.add_param('d_param', np.zeros(3), desc='parameterized diameter')
        self.add_param('t_param', np.zeros(3), desc='parameterized thickness')
        self.add_param('rho', 0.0, desc='density of the material')

        self.add_output('mass', 0.0, desc='tower mass')


    def solve_nonlinear(self, params, unknowns, resids):
        H = params['turbineH']
        r = params['d_param']/2.
        t = params['t_param']
        rho = params['rho']

        bottom_outer = 1./3.*3.141592653589793*(r[0]**2+r[0]*r[1]+r[1]**2)*H/2.
        bottom_inner = 1./3.*3.141592653589793*((r[0]-t[0])**2+(r[0]-t[0])*(r[1]-t[1])+(r[1]-t[1])**2)*H/2.
        top_outer = 1./3.*3.141592653589793*(r[1]**2+r[1]*r[2]+r[2]**2)*H/2.
        top_inner = 1./3.*3.141592653589793*((r[1]-t[1])**2+(r[1]-t[1])*(r[2]-t[2])+(r[2]-t[2])**2)*H/2.

        unknowns['mass'] = (bottom_outer + top_outer - bottom_inner - top_inner)*rho

    def linearize(self, params, unknowns, resids):
        H = params['turbineH']
        d = params['d_param']
        t = params['t_param']
        rho = params['rho']
        mass = unknowns['mass']

        J = {}
        # J['mass', 'turbineH'] = 1./3.*3.141592653589793*((d[0]/2.)**2+(d[0]/2.)*(d[1]/2.)+(d[1]/2.)**2)/2.+ \
        #     1./3.*3.141592653589793*((d[1]/2.)**2+(d[1]/2.)*(d[2]/2.)+(d[2]/2.)**2)/2. - \
        #     1./3.*3.141592653589793*((d[0]/2.-t[0])**2+(d[0]/2.-t[0])*(d[1]/2.-t[1])+(d[1]/2.-t[1])**2)/2. - \
        #     1./3.*3.141592653589793*((d[1]/2.-t[1])**2+(d[1]/2.-t[1])*(d[2]/2.-t[2])+(d[2]/2.-t[2])**2)/2.
        J['mass', 'turbineH'] = mass/H

        dmass_dD0 = H*3.141592653589793/24.*((2.*d[0]+d[1])-(2.*(d[0]-2.*t[0])+(d[1]-2.*t[1])))
        dmass_dD1 = H*3.141592653589793/24.*((d[0]+2.*d[1])+(2.*d[1]+d[2])-((d[0]-2.*t[0])+2.*(d[1]-2.*t[1]))-(2.*(d[1]-2.*t[1])+(d[2]-2.*t[2])))
        dmass_dD2 = H*3.141592653589793/24.*((2.*d[2]+d[1])-(2.*(d[2]-2.*t[2])+(d[1]-2.*t[1])))
        # J['mass', 'd_param'] = np.array([dmass_dD0, dmass_dD1, dmass_dD2])
        J['mass', 'd_param'] = np.zeros((1,3))
        J['mass', 'd_param'][0][0] = dmass_dD0*rho
        J['mass', 'd_param'][0][1] = dmass_dD1*rho
        J['mass', 'd_param'][0][2] = dmass_dD2*rho

        dmass_dt0 = -1.*H*3.141592653589793/24.*(-4.*(d[0]-2.*t[0])-2.*(d[1]-2.*t[1]))
        dmass_dt1 = -1.*H*3.141592653589793/24.*(-4.*(d[1]-2.*t[1])-2.*(d[0]-2.*t[0])+-4.*(d[1]-2.*t[1])-2.*(d[2]-2*t[2]))
        dmass_dt2 = -1.*H*3.141592653589793/24.*(-4.*(d[2]-2.*t[2])-2.*(d[1]-2.*t[1]))
        # J['mass', 't_param'] = np.array([dmass_dt0, dmass_dt1, dmass_dt2])
        J['mass', 't_param'] = np.zeros((1,3))
        J['mass', 't_param'][0][0] = dmass_dt0*rho
        J['mass', 't_param'][0][1] = dmass_dt1*rho
        J['mass', 't_param'][0][2] = dmass_dt2*rho

        return J


class TowerDiscretization(Component):
    """discretize geometry into finite element nodes"""

    #inputs

    def __init__(self, nPoints, nFull):

        super(TowerDiscretization, self).__init__()

        self.nFull = nFull
        self.nPoints = nPoints

        self.deriv_options['check_form'] = 'central'
        self.deriv_options['check_step_size'] = 1E-6

        self.deriv_options['form'] = 'central'
        self.deriv_options['step_size'] = 1E-6
        self.deriv_options['step_calc'] = 'relative'
        self.deriv_options['type'] = 'fd'

         # variables
        self.add_param('z_param', np.zeros(nPoints), units='m', desc='parameterized locations along tower, linear lofting between')
        self.add_param('d_param', np.zeros(nPoints), units='m', desc='tower diameter at corresponding locations')
        self.add_param('t_param', np.zeros(nPoints), units='m', desc='shell thickness at corresponding locations')
        self.add_param('z_full', np.zeros(nFull), units='m', desc='locations along tower')

        #out
        self.add_output('d_full', np.zeros(nFull), units='m', desc='tower diameter at corresponding locations')
        self.add_output('t_full', np.zeros(nFull), units='m', desc='shell thickness at corresponding locations')


    def solve_nonlinear(self, params, unknowns, resids):

        unknowns['d_full'] = np.interp(params['z_full'], params['z_param'], params['d_param'])
        unknowns['t_full'] = np.interp(params['z_full'], params['z_param'], params['t_param'])

    # def linearize(self, params, unknowns, resids):



class hoopStressEurocode(Component):
    """Hoop stress at each point"""

    #inputs

    def __init__(self, nFull):

        super(hoopStressEurocode, self).__init__()

        self.nFull = nFull

        self.add_param('d_full', np.zeros(nFull), desc='diameter at each point')
        self.add_param('t_full', np.zeros(nFull), desc='thickness at each point')
        self.add_param('L_reinforced', np.zeros(nFull), units='m')
        self.add_param('rhoAir', 1.225, desc='density of air')
        self.add_param('Vel', 0.0, desc='wind speed')

        self.add_output('hoop_stress', np.zeros(nFull), desc='hoop stress at each point')

    def solve_nonlinear(self, params, unknowns, resids):

        d = params['d_full']
        t = params['t_full']
        L_reinforced = params['L_reinforced']
        rho = params['rhoAir']
        Vel = params['Vel']

        q_dyn = 0.5*rho*Vel**2 #TODO THIS NEEDS TO BE FIXED
        r = d/2.0-t/2.0  # radius of cylinder middle surface
        omega = L_reinforced/np.sqrt(r*t)

        C_theta = 1.5  # clamped-clamped
        k_w = 0.46*(1.0 + 0.1*np.sqrt(C_theta/omega*r/t))
        Peq = k_w*q_dyn
        hoop_stress = -Peq*r/t

        print 'MINE: '
        print 'q_dyn: ', q_dyn
        unknowns['hoop_stress'] = hoop_stress

    def linearize(self, params, unknowns, resids):

        nFull = self.nFull

        d = params['d_full']
        t = params['t_full']
        L_reinforced = params['L_reinforced']
        rho = params['rhoAir']
        Vel = params['Vel']

        C_theta = 1.5
        r = d/2.0-t/2.0
        omega = L_reinforced/np.sqrt(r*t)

        hoop_stress = unknowns['hoop_stress']

        J = {}
        dhoop_dD_full = np.zeros((nFull, nFull))
        dhoop_dT_full = np.zeros((nFull, nFull))
        for i in range(nFull):
            di = d[i]
            ti = t[i]
            ri = (di/2.-ti/2.)
            L = L_reinforced[i]

            #dHoop_dD
            p1 = -0.0140846*ri*Vel**2*rho/(ti*np.sqrt(ri*np.sqrt(ri*ti)/(L*ti)))
            p2 = ri/(4.*L*np.sqrt(ri*ti))+np.sqrt(ri*ti)/(2.*L*ti)
            p3 = 0.115*Vel**2*rho/ti
            p4 = 1.+.1*np.sqrt(1.5)*np.sqrt(ri*np.sqrt(ri*ti)/(L*ti))
            dhoop_dD_full[i] = p1*p2-p3*p4

            #dHoop_dT
            a1 = -0.0140846*ri*Vel**2*rho/(ti*np.sqrt(ri*np.sqrt(ri*ti)/(L*ti)))
            a2 = (di/2.-ti)*(ri)/(2.*L*ti*np.sqrt(ri*ti))
            a3 = ri*np.sqrt(ri*ti)/(L*ti**2)
            a4 = np.sqrt(ri*ti)/(2.*L*ti)

            a5 = 0.23*ri*Vel**2*rho/(ti**2)
            a6 = 1.+0.1*np.sqrt(1.5)*np.sqrt(ri*np.sqrt(ri*ti)/(L*ti))

            a7 = 0.115*Vel**2*rho/ti
            a8 = 1.+0.1*np.sqrt(1.5)*np.sqrt(ri*np.sqrt(ri*ti)/(L*ti))
            dhoop_dT_full[i] = a1*(a2-a3-a4) + a5*a6 + a7*a8


        J['hoop_stress', 'd_full'] = dhoop_dD_full
        J['hoop_stress', 't_full'] = dhoop_dT_full

        return J



class axialStress(Component):
    """axial stress at each point"""

    def __init__(self, nFull):

        super(axialStress, self).__init__()

        self.nFull = nFull

        self.add_param('d_full', np.zeros(nFull), desc='diameter at each point')
        self.add_param('t_full', np.zeros(nFull), desc='thickness at each point')
        self.add_param('z_full', np.zeros(nFull), desc='location on tower')
        self.add_param('Fx', 0.0, desc='fx force at top of the tower')
        self.add_param('Fz', 0.0, desc='z force at top of tower')
        self.add_param('Mxx', 0.0, desc='moments at the top of the tower, xx')
        self.add_param('Myy', 0.0, desc='moments at the top of the tower, yy')
        self.add_param('shearExp', 0.15, desc='Shear exponent')
        self.add_param('rhoAir', 1.225, desc='density of air')
        self.add_param('V', 0.0, desc='wind speed at reference height (90 m for NREL 5 MW reference turbine)')
        self.add_param('zref', 90., desc='height corresponding to wind speed V')

        self.add_output('axial_stress', np.zeros(nFull), desc='hoop stress at each point')

    def solve_nonlinear(self, params, unknowns, resids):

        d_full = params['d_full']
        t_full = params['t_full']
        z_full = params['z_full']
        Fz = params['Fz']
        Fx = params['Fx']
        Mxx = params['Mxx']
        Myy = params['Myy']
        rho = params['rhoAir']
        V = params['V']
        shearExp = params['shearExp']
        zref = params['zref']
        pi = 3.1415926
        nFull = self.nFull

        ztop = z_full[-1]
        Az = np.zeros(nFull)
        momentY = np.zeros(nFull)
        for i in range(nFull):
            Az[i] = 0.25*pi*(d_full[i]**2 - (d_full[i]-2.*t_full[i])**2)
            p1 = 0.5*rho*V**2/(zref**(2*shearExp))
            DD = d_full[i :]
            D = np.sum(DD)/len(DD)
            drag = p1*D*(ztop**(2.*shearExp+1)-z_full[i]**(2.*shearExp+1))
            windMoment = drag*(ztop-z_full[i])/2.
            momentY[i] = Myy-windMoment-Fx*(ztop-z_full[i]) #TODO A LITTLE OFF?

        Iyy = pi/4.*((d_full/2.)**4-((d_full-2.*t_full)/2.)**4)
        print 'My Axial: '
        print 'Mxx: ', Mxx
        print 'Myy: ', momentY
        axial_stress = Fz/Az + np.sqrt(Mxx**2+momentY**2)/Iyy*d_full/2.0

        unknowns['axial_stress'] = axial_stress




    # def linearize(self, params, unknowns, resids):


class shellBuckling(Component):
    def __init__(self, nFull):

        super(shellBuckling, self).__init__()

        self.nFull = nFull
        self.add_param('d_full', np.zeros(nFull), desc='diameter at specified locations')
        self.add_param('t_full', np.zeros(nFull), desc='thickness at specified locations')
        self.add_param('axial_stress', np.zeros(nFull), desc='axial stress at specified locations')
        self.add_param('hoop_stress', np.zeros(nFull), desc='shoop stress at specified locations')
        self.add_param('shear_stress', np.zeros(nFull), desc='shear stress at specified locations')
        self.add_param('L_reinforced', np.zeros(nFull), units='m')
        self.add_param('E', np.zeros(nFull), units='N/m**2', desc='modulus of elasticity')
        self.add_param('sigma_y', np.zeros(nFull), units='N/m**2', desc='yield stress')
        self.add_param('gamma_f', 1.35)
        self.add_param('gamma_b', 1.1)

        self.add_output('shell_buckling', np.zeros(nFull), desc='shell buckling at each point')


    def solve_nonlinear(self, params, unknowns, resids):
        d_full = params['d_full']
        t_full = params['t_full']
        axial_stress = params['axial_stress']
        hoop_stress = params['hoop_stress']
        shear_stress = params['shear_stress']
        L_reinforced = params['L_reinforced']
        E = params['E']
        sigma_y = params['sigma_y']
        gamma_f = params['gamma_f']
        gamma_b = params['gamma_b']

        unknowns['shell_buckling'] = _shellbuckling.shellbucklingeurocode(d_full, t_full, axial_stress, hoop_stress, shear_stress, L_reinforced, E, sigma_y, gamma_f, gamma_b)







class calcFreq(Component):

    def __init__(self):

        super(calcFreq, self).__init__()

        self.add_param('turbineH', 0.0, desc='Height of the Tower')
        self.add_param('E', 0.0, desc='Modulus of Elasticity')
        self.add_param('t_param', np.zeros(3), desc='parameterized thickness')
        self.add_param('d_param', np.zeros(3), desc='parameterized dimeter')
        self.add_param('rho', 0.0, desc='tower density')
        self.add_param('topM', 0.0, desc='Point mass at top of tower')
        self.add_param('topI', 0.0, desc='moment of inertia at top of tower')

        self.add_output('freq', 0.0, desc='first natural frequency')


    def solve_nonlinear(self, params, unknowns, resids):

        rho = params['rho']
        L = params['turbineH']
        E = params['E']
        t1 = params['t_param'][0]
        t2 = params['t_param'][1]
        t3 = params['t_param'][2]
        d1 = params['d_param'][0]
        d2 = params['d_param'][1]
        d3 = params['d_param'][2]

        d = (d1+d2+d3)/3.
        t = (t1+t2+t3)/3.

        outer = 1./3.*3.141592653589793*((d/2.)**2+(d/2.)*(d/2.)+(d/2.)**2)*L
        inner = 1./3.*3.141592653589793*(((d/2.)-t)**2+((d/2.)-t)*((d/2.)-t)+((d/2.)-t)**2)*L
        mass = (outer - inner)*rho
        m = mass/L

        def func(x):
            M = params['topM']
            I = params['topI']

            p1 = 1.
            p2 = np.cos(x)*np.cosh(x)
            p3 = x*M/(m*L)*(np.cos(x)*np.sinh(x)-np.sin(x)*np.cosh(x))
            p4 = x**3*I/(m*L**3)*(np.cosh(x)*np.sin(x)+np.sinh(x)*np.cos(x))
            p5 = x**4*M*I/(m**2*L**4)*(1.-np.cos(x)*np.cosh(x))

            return p1+p2+p3-p4+p5

        x0 = fsolve(func, 0.5)

        di = d-2.*t
        I = 3.1415926*(d**4-di**4)/64.

        omega  = x0**2*(E*I/(m*L**4))**0.5

        unknowns['freq'] = omega / (2.*3.1415926)


class freq(Component):

    def __init__(self):

        super(freq, self).__init__()

        self.add_param('L', 0.0, desc='Height of the Tower')
        self.add_param('m', 0.0, desc='Modulus of Elasticity')
        self.add_param('I', 0.0, desc='Modulus of Elasticity')
        self.add_param('E', 0.0, desc='Modulus of Elasticity')
        self.add_param('Mt', 0.0, desc='parameterized thickness')
        self.add_param('It', 0.0, desc='parameterized dimeter')

        self.add_output('freq', 0.0, desc='first natural frequency')


    def solve_nonlinear(self, params, unknowns, resids):

        L = params['L']
        m = params['m']
        I = params['I']
        E = params['E']
        Mt = params['Mt']
        It = params['It']

        # # constant
        # Mt = 3.4
        # It = 6.4
        # E = 1.2
        #
        # # variable
        # m = 2.6
        # L = 5.3
        # I = 3.6

        def R(lam):
            return 1 + cos(lam)*cosh(lam) + lam*Mt/(m*L)*(cos(lam)*sinh(lam) - sin(lam)*cosh(lam)) \
            - lam**3*It/(m*L**3)*(cosh(lam)*sin(lam) + sinh(lam)*cos(lam)) \
            + lam**4*Mt*It/(m**2*L**4)*(1 - cos(lam)*cosh(lam))

        def freq(m, L, I):

            l0 = 1.0
            l = root(R, l0)['x'][0]
            f = l**2/(2*pi)*sqrt(E*I/(m*L**4))  # divided by 2pi to give Hz
            return f, l

        f, l = freq(m, L, I)
        unknowns['freq'] = f

    def linearize(self, params, unknowns, resids):

        L = params['L']
        m = params['m']
        I = params['I']
        E = params['E']
        Mt = params['Mt']
        It = params['It']

        def R(lam):
            return 1 + cos(lam)*cosh(lam) + lam*Mt/(m*L)*(cos(lam)*sinh(lam) - sin(lam)*cosh(lam)) \
            - lam**3*It/(m*L**3)*(cosh(lam)*sin(lam) + sinh(lam)*cos(lam)) \
            + lam**4*Mt*It/(m**2*L**4)*(1 - cos(lam)*cosh(lam))

        def freq(m, L, I):

            l0 = 1.0
            l = root(R, l0)['x'][0]
            f = l**2/(2*pi)*sqrt(E*I/(m*L**4))  # divided by 2pi to give Hz
            return f, l

        f, l = freq(m, L, I)

        sl = sin(l)
        cl = cos(l)
        shl = sinh(l)
        chl = cosh(l)

        pfpm = -l**2/(4*pi*m)*sqrt(E*I/(m*L**4))
        pfpl = l/pi*sqrt(E*I/(m*L**4))
        prpm = -l*Mt/(m**2*L)*(cl*shl - sl*chl) \
            + l**3*It/(m**2*L**3)*(chl*sl + shl*cl) \
            - 2*l**4*Mt*It/(m**3*L**4)*(1 - cl*chl)
        prpl = 0.00407951671291345*l**4*(sl*chl - cl*shl) + 0.0163180668516538*l**3*(-cl*chl + 1) - 0.0330680825317337*l**3*cl*chl - 0.0496021237976006*l**2*(sl*chl + cl*shl) - 0.493468795355588*l*sl*shl - 1.24673439767779*sl*chl + 1.24673439767779*cl*shl

        pfpL = -l**2/(pi*L)*sqrt(E*I/(m*L**4))
        prpL = -l*Mt/(m*L**2)*(cl*shl - sl*chl) \
            + 3*l**3*It/(m*L**4)*(chl*sl + shl*cl) \
            - 4*l**4*Mt*It/(m**2*L**5)*(1 - cl*chl)

        pfpI = l**2/(4*pi*I)*sqrt(E*I/(m*L**4))

        dfdm = pfpm - pfpl*prpm/prpl
        dfdL = pfpL - pfpl*prpL/prpl
        dfdI = pfpI

        J = {}
        J['freq', 'L'] = dfdL
        J['freq', 'm'] = dfdm
        J['freq', 'I'] = dfdI
