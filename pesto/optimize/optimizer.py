import scipy.optimize
import re
import sys

class Optimizer:

    def __init__(self, solver='SciPy_L-BFGS-B'):

        self.solver = solver
        self.tol = 1e-9
        self.options = {'maxiter': 1000, 'disp': False}

    def minimize(self, problem, x0):

        lb = problem.upper_parameter_bounds
        ub = problem.lower_parameter_bounds

        if re.match('^(?i)(scipy_)',self.solver):

            scipy_method = self.solver[6:]

            if re.match('^(?i)(ls_)',scipy_method):

                ls_method = scipy_method[3:]
                bounds = (lb[0, :], ub[0, :])

                res = scipy.optimize.least_squares(
                    problem.objective.get_res,
                    x0,
                    method=ls_method,
                    jac=problem.objective.get_sres,
                    bounds=bounds,
                    ftol=self.tol,
                    tr_solver='exact',
                    loss='linear',
                )

            else:

                bounds = scipy.optimize.Bounds(lb[0, :], ub[0, :])

                res = scipy.optimize.minimize(
                    problem.objective.get_fval,
                    x0,
                    method=scipy_method,
                    jac=problem.objective.get_grad,
                    hess=problem.objective.get_hess,
                    hessp=problem.objective.get_hessp,
                    bounds=bounds,
                    tol=self.tol,
                    options=self.options,
                )

        elif re.match('^(?i)(dlib_)',self.solver):

            if 'dlib' not in sys.modules:
                try:
                    import dlib
                except ImportError:
                    print('No installation of dlib was found, which is required for the ' + self.solver + ' method.')

            dlib_method = self.solver[5:]

            res = dlib.find_min_global(
                problem.objective.get_fval_vararg,
                list(lb[0, :]),
                list(ub[0, :]),
                int(self.options['maxiter']),
                0.002,
            )

        elif re.match('^(?i)(pyopt_)',self.solver):

            if 'pyopt' not in sys.modules:
                try:
                    import pyopt
                except ImportError:
                    print('No installation of pyopt was found, which is required for the ' + self.solver + ' method.')

            pyopt_method = self.solver[6:]

            opt_prob = pyopt.pyOpt_optimization.Optimization('pyPesto',problem.objective.get_fval_pyopt)
            opt_prob.addObj('f')

            for index, name in enumerate(problem.parameter_names):
                opt_prob.addVar(name, type='c', value=x0[index], lower=lb[index], upper=ub[index])

            if pyopt_method == 'SNOPT':
                opt = pyopt.SNOPT(options={
                })

            if pyopt_method == 'NLPQL':
                opt = pyopt.NLPQL(options={
                    'maxIt':self.options['maxiter'],
                    'lql':False,
                })

            if pyopt_method == 'NLPQLP':
                opt = pyopt.NLPQLP(options={
                    'MAXIt':self.options['maxiter'],
                    'LQL':False,
                })

            if pyopt_method == 'FSQP':
                opt = pyopt.FSQP(options={
                })


            opt(opt_prob, sens_type=problem.objective.get_grad_pyopt, disp_opts=False)
            res = opt_prob.solution[0]

        return res