# import _shellbuckling
# from setupOptimization import *
# import numpy as np
#
# def hoopStressEurocode(z, d, t, L_reinforced, q_dyn):
#     """default method for computing hoop stress using Eurocode method"""
#
#     r = d/2.0-t/2.0  # radius of cylinder middle surface
#     omega = L_reinforced/np.sqrt(r*t)
#
#     C_theta = 1.5  # clamped-clamped
#     k_w = 0.46*(1.0 + 0.1*np.sqrt(C_theta/omega*r/t))
#     Peq = k_w*q_dyn
#     hoop_stress = -Peq*r/t
#
#     return hoop_stress
#
# if __name__=="__main__":
#     d_param = np.array([6.0, 4.935, 3.87])
#     t_param = np.array([0.027*1.3, 0.023*1.3, 0.019*1.3])
#     z_param = np.array([0.,50.,100.])
#     sigma_z = np.ones(3)*1.0E6
#     sigma_t = np.ones(3)*1.0E6
#     tau_zt = np.ones(3)*1.0E6
#     n = 3
#
#     L_reinforced, Toweryaw, E, G, rho, sigma_y, kidx, kx, ky, kz, ktx, kty, ktz, nK, \
#                 midx, m, mIxx, mIyy, mIzz, mIxy, mIxz, mIyz, mrhox, mrhoy, mrhoz, nMass, \
#                 addGravityLoadForExtraMass, wind_zref, wind_z0, shearExp, wind_Uref1, \
#                 plidx1, Fx1, Fy1, Fz1, Mxx1, Myy1, Mzz1, nPL, wind_Uref2, \
#                 plidx2, Fx2, Fy2, Fz2, Mxx2, Myy2, Mzz2, gamma_f, gamma_m, gamma_n, \
#                 gamma_b, z_DEL, M_DEL, nDEL, gamma_fatigue, life, m_SN, min_d_to_t, min_taper \
#                 = setupTower(n)
#
#     maxSpeed = 70.0
#
#     rhoAir = 1.225
#     qdyn = 0.5*rhoAir*maxSpeed**2
#
#     hoop_stress = hoopStressEurocode(z_param,d_param,t_param,L_reinforced, qdyn)
#     print hoop_stress
#
#     nPoints = len(d_param)
#     nFull = n
#
#     axial_stress = np.ones(n)*1.0E6
#     shear_stress = np.ones(n)*1.0E6
#     # axial_stress = Fz/params['Az'] - np.sqrt(Mxx**2+Myy**2)/params['Iyy']*params['d']/2.0  #More conservative, just use the tilted bending and add total max shear as well at the same point, if you do not like it go back to the previous lines
#     #
#     # shear_stress = 2. * np.sqrt(Vx**2+Vy**2) / params['Az'] # coefficient of 2 for a hollow circular section, but should be conservative for other shapes
#
#     # EU_utilization = np.zeros(n)
#     shell = _shellbuckling.shellbucklingeurocode(d_param, t_param, axial_stress, hoop_stress, shear_stress, L_reinforced, E, sigma_y, gamma_f, gamma_b)
#
#     print shell


import numpy as np
from FLORISSE3D.simpleTower import shellBuckling, hoopStressEurocode, axialStress
from openmdao.api import Component, Group, Problem, IndepVarComp
from setupOptimization import setupTower
from towerse.tower import TowerSE
from FLORISSE3D.GeneralWindFarmComponents import get_z, getTurbineZ


if __name__ == '__main__':
    nTurbs = 1

    """set up 3D aspects of wind farm"""
    H1_H2 = np.array([])
    for i in range(nTurbs/2):
        H1_H2 = np.append(H1_H2, 0)
        H1_H2 = np.append(H1_H2, 1)
    if len(H1_H2) < nTurbs:
        H1_H2 = np.append(H1_H2, 0)

    """Define tower structural properties"""
    # --- geometry ----
    d_param = np.array([6.0, 5.0, 4.0])
    t_param = np.array([0.027*1.3, 0.025*1.3, 0.018*1.3])
    n = 3

    turbineH1 = 120.

    L_reinforced, Toweryaw, E, G, rho, sigma_y, kidx, kx, ky, kz, ktx, kty, ktz, nK, \
                midx, m, mIxx, mIyy, mIzz, mIxy, mIxz, mIyz, mrhox, mrhoy, mrhoz, nMass, \
                addGravityLoadForExtraMass, wind_zref, wind_z0, shearExp, wind_Uref1, \
                plidx1, Fx1, Fy1, Fz1, Mxx1, Myy1, Mzz1, nPL, wind_Uref2, \
                plidx2, Fx2, Fy2, Fz2, Mxx2, Myy2, Mzz2, gamma_f, gamma_m, gamma_n, \
                gamma_b, z_DEL, M_DEL, nDEL, gamma_fatigue, life, m_SN, min_d_to_t, min_taper \
                = setupTower(n)

    nPoints = len(d_param)
    nFull = n
    wind = 'PowerWind'
    Vel = wind_Uref2
    rhoAir = 1.225


    prob = Problem()
    root = prob.root = Group()

    root.add('d_full', IndepVarComp('d_full', d_param), promotes=['*'])
    root.add('t_full', IndepVarComp('t_full', t_param), promotes=['*'])
    root.add('shellBuckling', shellBuckling(n), promotes=['*'])
    root.add('hoopStressEurocode', hoopStressEurocode(n), promotes=['*'])
    root.add('axialStress', axialStress(n), promotes=['*'])
    root.add('getTurbineZ', getTurbineZ(nTurbs), promotes=['*'])
    root.add('get_z_paramH1', get_z(nPoints))
    root.add('get_z_fullH1', get_z(n))
    #For Constraints
    root.add('TowerH1', TowerSE(nPoints, nFull, nK, nMass, nPL, nDEL, wind=wind), promotes=['L_reinforced',
                        'E','G','sigma_y','kidx','kx','ky','kz','ktx','kty',
                        'ktz','midx','m','mIxx','mIyy','mIzz','mIxy','mIxz','mIyz',
                        'mrhox','mrhoy','mrhoz','addGravityLoadForExtraMass','g',
                        'gamma_f','gamma_m','gamma_n','gamma_b','life','m_SN',
                        'gc.min_d_to_t','gc.min_taper','M_DEL','gamma_fatigue'])

    root.connect('turbineH1', 'get_z_paramH1.turbineZ')
    root.connect('turbineH1', 'get_z_fullH1.turbineZ')
    root.connect('get_z_paramH1.z_param', 'TowerH1.z_param')
    root.connect('get_z_fullH1.z_param', 'TowerH1.z_full')
    root.connect('TowerH1.z_full', 'z_full')

    prob.setup()

    prob['L_reinforced'] = L_reinforced
    prob['rhoAir'] = rhoAir
    prob['Fx'] = Fx2
    prob['Fz'] = Fz2
    prob['Mxx'] = Mxx2
    prob['Myy'] = Myy2
    prob['Vel'] = Vel
    prob['zref'] = 90.
    prob['shearExp'] = 0.15
    prob['E'] = E
    prob['gamma_f'] = gamma_f
    prob['gamma_b'] = gamma_b
    prob['sigma_y'] = sigma_y

    prob['TowerH1.wind1.shearExp'] = shearExp
    prob['TowerH1.wind2.shearExp'] = shearExp
    prob['TowerH1.d_param'] = d_param
    prob['TowerH1.t_param'] = t_param

    prob['turbineH1'] = turbineH1
    prob['H1_H2'] = H1_H2

    """tower structural properties"""
    # --- geometry ----
    prob['L_reinforced'] = L_reinforced
    prob['TowerH1.yaw'] = Toweryaw

    # --- material props ---
    prob['E'] = E
    prob['G'] = G
    prob['TowerH1.tower1.rho'] = rho
    prob['sigma_y'] = sigma_y

    # --- spring reaction data.  Use float('inf') for rigid constraints. ---
    prob['kidx'] = kidx
    prob['kx'] = kx
    prob['ky'] = ky
    prob['kz'] = kz
    prob['ktx'] = ktx
    prob['kty'] = kty
    prob['ktz'] = ktz

    # --- extra mass ----
    prob['midx'] = midx
    prob['m'] = m
    prob['mIxx'] = mIxx
    prob['mIyy'] = mIyy
    prob['mIzz'] = mIzz
    prob['mIxy'] = mIxy
    prob['mIxz'] = mIxz
    prob['mIyz'] = mIyz
    prob['mrhox'] = mrhox
    prob['mrhoy'] = mrhoy
    prob['mrhoz'] = mrhoz
    prob['addGravityLoadForExtraMass'] = addGravityLoadForExtraMass
    # -----------

    # --- wind ---
    prob['TowerH1.zref'] = wind_zref
    prob['TowerH1.z0'] = wind_z0
    # ---------------

    # # --- loading case 1: max Thrust ---
    prob['TowerH1.wind1.Uref'] = wind_Uref1
    prob['TowerH1.tower1.plidx'] = plidx1
    prob['TowerH1.tower1.Fx'] = Fx1
    prob['TowerH1.tower1.Fy'] = Fy1
    prob['TowerH1.tower1.Fz'] = Fz1
    prob['TowerH1.tower1.Mxx'] = Mxx1
    prob['TowerH1.tower1.Myy'] = Myy1
    prob['TowerH1.tower1.Mzz'] = Mzz1
    # # ---------------

    # # --- loading case 2: max Wind Speed ---
    prob['TowerH1.wind2.Uref'] = wind_Uref2
    prob['TowerH1.tower2.plidx'] = plidx2
    prob['TowerH1.tower2.Fx'] = Fx2
    prob['TowerH1.tower2.Fy'] = Fy2
    prob['TowerH1.tower2.Fz'] = Fz2
    prob['TowerH1.tower2.Mxx'] = Mxx2
    prob['TowerH1.tower2.Myy'] = Myy2
    prob['TowerH1.tower2.Mzz'] = Mzz2
    # # ---------------

    # --- safety factors ---
    prob['gamma_f'] = gamma_f
    prob['gamma_m'] = gamma_m
    prob['gamma_n'] = gamma_n
    prob['gamma_b'] = gamma_b
    # --------------


    prob.run()

    print 'MINE'
    print 'Shell Bucking: ', prob['shell_buckling']
    print 'Axial Stress: ', prob['axial_stress']
    print 'Hoop Stress: ', prob['hoop_stress']

    print 'TOWERSE'
    print 'Shell Buckling max thrust: ', prob['TowerH1.tower1.shell_buckling']
    print 'Shell Buckling max wind speed: ', prob['TowerH1.tower2.shell_buckling']
    print prob['TowerH1.z_full']
