import numpy as np

# class calcAxialStress(Component):
#     """axial stress at each point"""
#
#     def __init__(self, nFull):
#
#         super(calcAxialStress, self).__init__()
#
#         self.nFull = nFull
#
#         self.add_param('d_full', np.zeros(nFull), desc='diameter at each point')
#         self.add_param('t_full', np.zeros(nFull), desc='thickness at each point')
#         self.add_param('z_full', np.zeros(nFull), desc='location on tower')
#         self.add_param)'F', np.zeros(3), desc='forces at the top of the tower (x,y,z)')
#         self.add_param)'M', np.zeros(3), desc='moments at the top of the tower (x,y,z)')
#         self.add_param('shearExp', 0.15, desc='Shear exponent')
#         self.add_param('rhoAir', 1.225, desc='density of air')
#         self.add_param('V', 0.0, desc='wind speed at reference height (90 m for NREL 5 MW reference turbine)')
#         self.add_param('zref', 90., desc='height corresponding to wind speed V')
#
#         self.add_output('axial_stress', np.zeros(nFull), desc='hoop stress at each point')
#
#     def solve_nonlinear(self, params, unknowns, resids):
#
#         d_full = params['d_full']
#         t_full = params['t_full']
#         z_full = params['z_full']
#         F = params['F']
#         M = params['M']
#         rho = params['rhoAir']
#         V = params['V']
#         shearExp = params['shearExp']
#         zref = params['zref']
#
#         nFull = self.nFull
#
#         ztop = z[-1]
#         drag = np.zeros(nFull)
#         for i in range(nFull):
#             p1 = 0.5*rho*V**2/(zref**(2*shearExp))
#             D = np.sum(d_full[i :])/np.len(d_full[i :])

if __name__=="__main__":
    d_full = np.array([2.,2.5,3.,3.5,4.,4.5,5.])
    t_full = np.array([0.05,0.045,0.04,0.035,0.03,0.025,0.02])
    z_full = np.array([0.,15.,30.,45.,60.,75.,90.])
    Fx = np.array([1284744.19620519])
    Fy = np.array([0.])
    Fz = np.array([-2914124.84400512])
    Mx = np.array([3963732.76208099])
    My = np.array([-2275104.79420872])
    Mz = np.array([-346781.68192839])
    rho = 1.225
    V = 70.
    shearExp = 0.12
    zref = 90.
    pi = 3.1415926

    nFull = len(d_full)

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
        momentY[i] = My-windMoment-Fx*(ztop-z_full[i])
        print momentY

    Iyy = pi/2.*(d_full**4-(d_full-2.*t_full)**4)
    axial_stress = Fz/Az - np.sqrt(Mx**2+momentY**2)/Iyy*d_full/2.0

    print "Stress: ", axial_stress
