import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

def func(x):
    M = 20000.
    I = 15000.
    m = 10000.
    L = 140.

    p1 = 1.
    p2 = np.cos(x)*np.cosh(x)
    p3 = x*M/(m*L)*(np.cos(x)*np.sinh(x)-np.sin(x)*np.cosh(x))
    p4 = x**3*I/(m*L**3)*(np.cosh(x)*np.sin(x)+np.sinh(x)*np.cos(x))
    p5 = x**4*M*I/(m**2*L**4)*(1.-np.cos(x)*np.cosh(x))

    return p1+p2+p3-p4+p5
if __name__=="__main__":


    # x0 = fsolve(func, 1.76)
    # print x0
    x = np.linspace(0.,100.,100)
    print x[98 :]
