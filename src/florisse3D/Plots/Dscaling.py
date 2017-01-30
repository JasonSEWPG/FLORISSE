import numpy as np
import matplotlib as mpl
# from pyoptsparse import Optimization, SNOPT, SLSQP, NSGA2, ALPSO



# def rosenbrock(xdict):
#
#     x = xdict['x']
#     dim = len(x)
#
#     f = 0.0
#     for i in range(dim - 1):
#         f += (1 - x[i])**2 + 100*(x[i+1] - x[i]**2)**2
#
#     funcs = {}
#     funcs['obj'] = f
#     funcs['con'] = []
#     fail = False
#
#     global f_iter
#     f_iter += 1
#
#     return funcs, fail
#
#
# def grad_rosenbrock(xdict, fdict):
#
#     x = xdict['x']
#     dim = len(x)
#
#     g = np.zeros((1, dim))
#     for i in range(dim - 1):
#         g[0, i] += -2*(1 - x[i]) + 200*(x[i+1] - x[i]**2)*-2*x[i]
#         g[0, i+1] += 200*(x[i+1] - x[i]**2)
#
#     grad = {}
#     grad['obj'] = {}
#     grad['obj']['x'] = g
#     grad['con'] = []
#     fail = False
#
#
#     return grad, fail
#
#
# def run_opt(optimizer, sens_type, dim):
#
#     opt_prob = Optimization('nd_rosenbrock', rosenbrock)
#     opt_prob.addObj('obj')
#     # opt_prob.addVarGroup('x', dim, type='c', lower=-5.0, upper=5.0, value=2.0)
#     opt_prob.addVarGroup('x', dim, type='c', lower=-2.0, upper=2.0, value=2.0)
#
#     sol = optimizer(opt_prob, sens=sens_type)
#     print sol
#
#
#
#
if __name__ == '__main__':
#
#     snopt = SNOPT()
#     # snopt.setOption('Major feasibility tolerance', 1e-5)
#     # snopt.setOption('Minor feasibility tolerance', 1e-5)
#     # snopt.setOption('Major optimality tolerance', 1e-5)
#     # snopt.setOption('Function precision', 1e-12)
#     # snopt.setOption('Iterations limit', 5000)
#
#     slsqp = SLSQP()
#     # slsqp.setOption('MAXIT', 5000)
#     # slsqp.setOption('IPRINT', 1)
#     # slsqp(opt_prob, sens_type=sens_type)
#     # print opt_prob.solution(0)
#
#     nsga_opt = NSGA2()
#     # nsga2.setOption('maxGen', 1000)
#     # nsga2.setOption('PopSize', 80)
#     # nsga2.setOption('pCross_real', 1.0)
#     # nsga2.setOption('pMut_real', 1.0)
#
#     alpso_opt = ALPSO()
#
#
#     # dimvec = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
#     # dimvec = [2, 4, 8, 16, 32]
#     dimvec = [64]
#     # dimvec = [4]
#
#     snopt_grad = []
#     snopt_fd = []
#     slsqp_grad = []
#     slsqp_fd = []
#     nsga = []
#     alpso = []
#
#
#     for dim in dimvec:
#
#         # f_iter = 0
#         # snopt = SNOPT()
#         # run_opt(snopt, grad_rosenbrock, dim)
#         # snopt_grad.append(f_iter)
#
#         # f_iter = 0
#         # snopt = SNOPT()
#         # run_opt(snopt, None, dim)
#         # snopt_fd.append(f_iter)
#
#         # f_iter = 0
#         # run_opt(slsqp, grad_rosenbrock, dim)
#         # slsqp_grad.append(f_iter)
#
#         # f_iter = 0
#         # run_opt(slsqp, 'FD', dim)
#         # slsqp_fd.append(f_iter)
#
#         f_iter = 0
#         nsga_opt = NSGA2()
#         nsga_opt.setOption('maxGen', 1000*dim)
#         nsga_opt.setOption('PopSize', 20*dim)
#         nsga_opt.setOption('pMut_real', 0.01)
#         nsga_opt.setOption('pCross_real', 1.0)
#         run_opt(nsga_opt, None, dim)
#         nsga.append(f_iter)
#
#         # f_iter = 0
#         # alpso_opt = ALPSO()
#         # alpso_opt.setOption('SwarmSize', 5*dim)
#         # alpso_opt.setOption('dtol', 1e-4)
#         # alpso_opt.setOption('atol', 1e-4)
#         # alpso_opt.setOption('rtol', 1e-4)
#         # alpso_opt.setOption('maxOuterIter', 10000)
#         # run_opt(alpso_opt, None, dim)
#         # alpso.append(f_iter)


    dimvec2 = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    snopt_grad = np.array([31, 35, 36, 45, 64, 68, 157, 183, 195, 244])
    slsqp_grad = np.array([33, 30, 43, 76, 135, 271, 535, 271, 445, 577])
    snopt_fd = np.array([102, 248, 375, 1017, 3054, 6315, 60142, 151453, 212337, 776704])
    slsqp_fd = np.array([84, 118, 227, 621, 1646, 5324, 16877, 36639, 46939, 152124])
    dimvec = [2, 4, 8, 16, 32, 64]
    alpso = np.array([1150, 32780, 108040, 488240, 2649760, 12301760])
    mpl.rcParams.update({'font.size': 20})



    # from myutilities import printArray
    # printArray(snopt_grad, name='snopt_grad')
    # printArray(slsqp_grad, name='slsqp_grad')
    # printArray(snopt_fd, name='snopt_fd')
    # printArray(slsqp_fd, name='slsqp_fd')
    # printArray(nsga, name='nsga', numpy=False)
    # printArray(alpso, name='alpso')

    # from myutilities import plt
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,8))
    plt.loglog(dimvec2, snopt_grad, '-bo')
    # plt.loglog(dimvec2, slsqp_grad)
    plt.loglog(dimvec2, snopt_fd, '-ro')
    # plt.loglog(dimvec2, slsqp_fd)
    # plt.loglog(dimvec, nsga)
    plt.loglog(dimvec, alpso, '-ko')
    # vec = np.concatenate(([1], dimvec2, [2000])
    # plt.loglog(vec, vec*200, 'k--')
    # plt.loglog(vec, vec**2*2e3, 'k--')
    plt.xlim([1, 2000])
    plt.ylim([1, 3e7])
    # plt.grid()
    plt.text(10, 10, 'analytic gradients', color='blue')
    plt.text(40, 1*1e3, 'finite difference gradients', color='red')
    plt.text(30, 7e5, 'gradient-free', color='black')
    # plt.text(1e3, 4e4, 'linear')
    # plt.text(150, 2e7, 'quadratic')
    plt.title('Optimization Scaling of the Multi-Dimensional \nRosenbrock Equation')
    plt.xlabel('Number of design variables')
    plt.ylabel('Number of function evaluations')
    plt.gcf().subplots_adjust(bottom=0.16)
    # plt.gca().tight_layout()
    # plt.save('/Users/andrewning/Dropbox/NREL/Downwind/paper/images/convergence2.pdf')
    plt.grid()
    plt.show()
