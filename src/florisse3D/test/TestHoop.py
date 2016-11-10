import numpy as np
from FLORISSE3D.simpleTower import hoopStressEurocode
from openmdao.api import Component, Group, Problem, IndepVarComp



if __name__ == '__main__':
    d_full = np.array([4.])
    t_full = np.array([0.05])
    L_reinforced = np.array([10.])
    rhoAir = 1.225
    Vel = 75.
    step = 1.0E-6

    prob = Problem()
    root = prob.root = Group()

    root.add('d_full', IndepVarComp('d_full', d_full), promotes=['*'])
    root.add('t_full', IndepVarComp('t_full', t_full), promotes=['*'])
    root.add('hoopStressEurocode', hoopStressEurocode(1), promotes=['*'])

    prob.setup()

    prob['L_reinforced'] = L_reinforced
    prob['rhoAir'] = rhoAir
    prob['Vel'] = Vel

    prob.run()

    hs1 = prob['hoop_stress']

    prob = Problem()
    root = prob.root = Group()

    root.add('d_full', IndepVarComp('d_full', d_full+step), promotes=['*'])
    root.add('t_full', IndepVarComp('t_full', t_full), promotes=['*'])
    root.add('hoopStressEurocode', hoopStressEurocode(1), promotes=['*'])

    prob.setup()

    prob['L_reinforced'] = L_reinforced
    prob['rhoAir'] = rhoAir
    prob['Vel'] = Vel

    prob.run()

    hsD = prob['hoop_stress']

    prob = Problem()
    root = prob.root = Group()

    root.add('d_full', IndepVarComp('d_full', d_full), promotes=['*'])
    root.add('t_full', IndepVarComp('t_full', t_full+step), promotes=['*'])
    root.add('hoopStressEurocode', hoopStressEurocode(1), promotes=['*'])

    prob.driver.add_objective('hoop_stress')
    prob.driver.add_desvar('d_full', lower=None, upper=None)
    prob.driver.add_desvar('t_full', lower=None, upper=None)

    prob.setup()

    prob['L_reinforced'] = L_reinforced
    prob['rhoAir'] = rhoAir
    prob['Vel'] = Vel

    prob.run()

    hsT = prob['hoop_stress']


    ds_dd = (hsD-hs1)/step
    ds_dt = (hsT-hs1)/step

    print 'D: ', ds_dd
    print 'T: ', ds_dt

    print 'Hoop Stress: ', prob['hoop_stress']
