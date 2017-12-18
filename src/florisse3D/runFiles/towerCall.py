import numpy as np
from openmdao.api import Group, Problem, IndepVarComp, pyOptSparseDriver, view_tree, profile
from FLORISSE3D.simpleTower import Tower


if __name__ == '__main__':

    nPoints = 3
    nFull = 15

    d_param = np.array([6.3,5.3,4.3])
    t_param = np.array([0.02,0.015,0.01])
    z_param = np.linspace(0.,120., nPoints)
    z_full = np.linspace(0.,120., nFull)
    rho = np.ones(nFull)*8500.0

    shearExp = 0.15
    rotorDiameter = np.array([126.4, 70.,150.,155.,141.])
    turbineZ = np.array([120., 70., 100., 120., 30.])

    """OpenMDAO"""

    prob = Problem()
    root = prob.root = Group()

    root.add('Tower', Tower(nPoints, nFull), promotes=['*'])

    prob.root.ln_solver.options['single_voi_relevance_reduction'] = True
    prob.setup(check=True)

    prob['d_param'] = d_param
    prob['t_param'] = t_param
    prob['z_param'] = z_param
    prob['z_full'] = z_full
    prob['rho'] = 8500.0*np.ones(nFull)
    prob['Vel'] = 10.
    prob['zref'] = 50.
    prob['z0'] = 0.
    prob['shearExp'] = 0.15
    prob['rhoAir'] = 1.225
    prob['rotor_mass'] = np.array([10000.])
    prob['nacelle_mass'] = np.array([20000.])
    prob['L_reinforced'] = 30.0*np.ones(nFull)
    prob['Myy'] = np.array([-2275104.79420872])
    prob['Myy'] = np.array([0.])
    prob['mrhox'] = np.array([-1.13197635])
    prob['Fx'] = np.array([1284744.])
    prob['Fy'] = np.array([0.])
    prob['E'] = 210.e9*np.ones(nFull)
    prob['sigma_y'] = 450.0e6*np.ones(nFull)
    prob['gamma_f'] = 1.35
    prob['gamma_b'] = 1.1
    prob['L'] = 90.
    prob['It'] = np.array([1.87597425e+07])

    prob.run()

    print 'Frequency: ', prob['freq']
    print 'Shell Buckling: ', prob['shell_buckling']
    print 'Axial Stress: ', prob['axial_stress']
    print 'Shear Stress: ', prob['shear_stress']
    print 'Hoop Stress: ', prob['hoop_stress']
    print 'Tower Mass: ', prob['mass']
