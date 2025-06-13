from typing import List, Callable

import numpy as np

from base import Function, Info, Benchmark
from utils import seed_every_where


def sphere(x):
    return np.sum(x ** 2)


def rastrigin(x):
    return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


def ellipsoid(x):
    return np.sum(np.arange(1, len(x) + 1) * x ** 2)


def griewank(x):
    sum_part = np.sum(x ** 2) / 400
    cos_part = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return 1 + sum_part - cos_part


def rosenbrock(x):
    if len(x) == 1:
        return sphere(x)
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1]) ** 2)


def ackley(x):
    p = len(x)
    term1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / p))
    term2 = -np.exp(np.sum(np.cos(2 * np.pi * x)) / p)
    return 20 + np.e + term1 + term2


Z_functions = [sphere, ellipsoid, rastrigin, griewank, rosenbrock, ackley]

F_functions = {
    'x': lambda x: x,
    'x^2': lambda x: x ** 2,
    'x^3': lambda x: x ** 3,
    '1/(1.1+x)': lambda x: 1 / (1.1 + x),
    'e^x': lambda x: np.exp(x),
    'sin(x)': lambda x: np.sin(x),
    'tan(x)': lambda x: np.tan(x),
    'log(1.1+x)': lambda x: np.log(1.1 + x),
    '|x|': lambda x: np.abs(x)
}


class JLCallable:
    def __call__(self, x):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError


class ZeroMin(JLCallable):

    def __init__(self, z=None):
        if z is None:
            z = np.random.choice(Z_functions)
        self.z = z

    def __call__(self, x):
        return self.z(x)

    def __repr__(self):
        return self.z.__name__


class RandF(JLCallable):
    def __init__(self, dim):
        self.dim = dim
        self.f_ks = np.random.choice(list(F_functions.keys()), dim)
        self.r_f_ks = [f_k.replace('x', f'x{i}') for i, f_k in enumerate(self.f_ks)]
        self.t = np.random.uniform(0, 1, dim)

    def __call__(self, x):
        fs = [F_functions[f_k] for f_k in self.f_ks]
        fvs = np.asarray([f(x) for f, x in zip(fs, x)])
        return np.sum(fvs * self.t)

    def __repr__(self):
        return '+'.join([f'{t}*{f_k}' for t, f_k in zip(self.t, self.r_f_ks)])


class RandYstar(JLCallable):
    def __init__(self, x_dim, y_dim):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.f_ks = np.random.choice(list(F_functions.keys()), x_dim)
        self.r_f_ks = [f_k.replace('x', f'x{i}') for i, f_k in enumerate(self.f_ks)]
        self.vs = []
        for i in range(y_dim):
            v = np.random.randn(x_dim)
            v = v / np.linalg.norm(v)
            self.vs.append(v)

    def __call__(self, x):
        fs = [F_functions[f_k] for f_k in self.f_ks]
        fvs = np.asarray([f(x) for f, x in zip(fs, x)])
        return np.asarray([np.cos(np.sum(fvs * v)) for v in self.vs])

    def __repr__(self):
        vs = np.asarray(self.vs).T
        return f'cos([{",".join(self.r_f_ks)}]*\n{vs})'


class JLBilevel(JLCallable):
    def __init__(self, u_dim, l_dim, F_star=None, zl=None):
        self.u_dim = u_dim
        self.l_dim = l_dim
        self.F_star = ZeroMin(F_star)
        self.zl = ZeroMin(zl)
        self.y_star = RandYstar(u_dim, l_dim)
        self.f_star = RandF(u_dim)
        self.f_r = RandF(u_dim + l_dim)

    def f(self, xu, xl):
        return self.zl(self.y_star(xu) - xl) + self.f_star(xu)

    def F(self, xu, xl):
        x = np.concatenate((xu, xl))
        return self.F_star(xu) + np.sum((self.y_star(xu) - xl) * self.f_r(x))

    def to_function(self, name='Not named yet', _c=None):
        if _c is None:
            constraints = None
            c = None
        else:
            c = RandF(self.u_dim + self.l_dim)
            x_star = lambda x: np.concatenate((x[:self.u_dim], self.y_star(x[self.u_dim:])))
            constraints = [
                lambda x: c(x_star(x)) - c(x) - _c,
                lambda x: c(x) - c(x_star(x)) - _c
            ]
        mFe = 500 * self.u_dim ** 2
        mfe = 3e5 * (self.u_dim + self.l_dim) ** 2
        info = JLInfo(name, self.u_dim, self.l_dim, np.asarray([[-1, 1]] * (self.u_dim + self.l_dim)).T,
                      constraints=constraints, F_star=self.F_star, zl=self.zl, y_star=self.y_star, f_star=self.f_star,
                      f_r=self.f_r, mFe=mFe, mfe=mfe)
        info.c = c

        return Function(f=self.f, F=self.F, info=info)


class _MultiLowerOpt(JLCallable):
    def __init__(self, u_dim, l_dim, F_star=None, zl=None):
        if l_dim < 3:
            raise ValueError('only supports l_dim>=3')

        self.u_dim = u_dim
        self.l_dim = l_dim - 2
        self.F_star = ZeroMin(F_star)
        self.zl = ZeroMin(zl)
        self.zml = ZeroMin()
        self.zmu = ZeroMin()
        self.y_star = RandYstar(u_dim, l_dim)
        self.f_star = RandF(u_dim)
        self.f_r = RandF(u_dim + l_dim)

    def f(self, xu, xl):
        y0, y1, y2 = xl[:-2], xl[-2], xl[-1]
        return self.zl(self.y_star(xu) - y0) + self.f_star(xu) + self.zml(y2 - y1)

    def F(self, xu, xl):
        x = np.concatenate((xu, xl))
        y0, y1 = xl[:-2], xl[-2]
        return self.F_star(xu) + np.sum((self.y_star(xu) - y0) * self.f_r(x)) + self.zmu(y1)

    def to_function(self, name):
        mFe = 500 * self.u_dim ** 2
        mfe = 3e5 * (self.u_dim + self.l_dim) ** 2
        info = JLInfo(name, self.u_dim, self.l_dim, np.asarray([[-1, 1]] * (self.u_dim + self.l_dim)).T,
                      F_star=self.F_star, zl=self.zl, y_star=self.y_star, f_star=self.f_star, f_r=self.f_r, mFe=mFe,
                      mfe=mfe)
        info.zml = self.zml
        info.zmu = self.zmu

        return Function(f=self.f, F=self.F, info=info)


class JLInfo(Info):
    def __init__(self, name: str, u_dim: int, l_dim: int, bounds, constraints: List[Callable] = None,
                 description: str = '',
                 F_star=None, zl=None, y_star=None, f_star=None, f_r=None, mFe=int(1e12), mfe=int(1e13)):
        super().__init__(name, u_dim, l_dim, bounds, constraints, description, ul_fe_max=mFe, ll_fe_max=mfe)
        self.description = f'F_star={F_star}\nzl={zl}\ny_star={y_star}\nf_star={f_star}\nf_r={f_r}'
        self.zu = F_star
        self.zl = zl
        self.y_star = y_star
        self.f_star = f_star
        self.f_r = f_r


class JLBenchmark(Benchmark):
    st_functions = None
    general_functions = None

    def get_standard_functions(self) -> List[Function]:
        if self.st_functions is None:
            seed_every_where(0)
            u, l = self.u_dim, self.l_dim
            self.st_functions = [
                JLBilevel(u, l, F_star=sphere, zl=sphere).to_function('JL0'),
                JLBilevel(u, l, F_star=ellipsoid, zl=rosenbrock).to_function('JL1'),
                JLBilevel(u, l, F_star=ackley, zl=sphere).to_function('JL2'),
                JLBilevel(u, l, F_star=rastrigin, zl=griewank).to_function('JL3'),
            ]
            if l >= 3:
                self.st_functions.append(_MultiLowerOpt(u, l).to_function('JL4'))
                self.st_functions.append(JLBilevel(u, l, F_star=sphere, zl=sphere).to_function('JL5', _c=0.5))

        return self.st_functions

    def generate_random_function(self, seed) -> Function:
        seed_every_where(seed)
        return JLBilevel(self.u_dim, self.l_dim).to_function()


if __name__ == '__main__':
    funs = JLBenchmark(2, 4).get_test_functions(5)
    for fun in funs:
        print(fun.info.description)
        print(fun.info.name)
