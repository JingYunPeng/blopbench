from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import List

import numpy as np
from scipy.stats import qmc

from utils import cur_seed


class Level:
    lower = 'lower level'
    upper = 'upper level'


class ExhaustedCallException(Exception):
    def __init__(self, level: str):
        super().__init__(f'Exhausted function evaluations on {level}.')


class Info:
    def __init__(self, name: str, u_dim: int, l_dim: int, bounds, constraints: List[Callable] = None,
                 description: str = '', ll_fe_max: int = 1e12, ul_fe_max: int = 1e12):
        self.name = name
        self.description = description
        self.u_dim = u_dim
        self.l_dim = l_dim
        self.dim = u_dim + l_dim

        self.bounds = bounds
        self.lb, self.ub = bounds
        self.ulb, self.uub = bounds[:, :u_dim]
        self.llb, self.lub = bounds[:, u_dim:]

        self.constraints = constraints or []

        self.ll_fe_max = ll_fe_max
        self.ul_fe_max = ul_fe_max

    def uniform_x(self, num_samples: int):
        samples = np.random.uniform(self.ulb, self.uub, (num_samples, self.u_dim))
        return samples

    def lhc_x(self, num_samples: int):
        sampler = qmc.LatinHypercube(d=self.u_dim, seed=cur_seed())
        samples = sampler.random(n=num_samples) * (self.uub - self.ulb) + self.ulb
        return samples

    def uniform_y(self, num_samples: int):
        samples = np.random.uniform(self.llb, self.lub, (num_samples, self.u_dim))
        return samples

    def lhc_y(self, num_samples: int):
        sampler = qmc.LatinHypercube(d=self.u_dim, seed=cur_seed())
        samples = sampler.random(n=num_samples) * (self.lub - self.llb) + self.llb
        return samples


class Function:
    def __init__(self, f: Callable, F: Callable, info: Info):
        self.ll_fe_left = info.ll_fe_max
        self.ul_fe_left = info.ul_fe_max
        self.f_function = f
        self.F_function = F
        self.info = info

    def reset_max_fe(self):
        self.ll_fe_left = self.info.ll_fe_max
        self.ul_fe_left = self.info.ul_fe_max

    def f(self, x, y) -> float:
        if self.ll_fe_left <= 0:
            raise ExhaustedCallException(Level.lower)
        self.ll_fe_left -= 1
        return self.f_function(x, y)

    def F(self, x, y) -> float:
        if self.ul_fe_left <= 0:
            raise ExhaustedCallException(Level.upper)
        self.ul_fe_left -= 1
        return self.F_function(x, y)

    @property
    def nfev(self):
        return self.info.ul_fe_max - self.ul_fe_left, self.info.ll_fe_max - self.ll_fe_left

    @property
    def name(self):
        return self.info.name

    def landscape2D(self, N):
        x = np.linspace(self.info.ulb[0], self.info.uub[0], N)
        y = np.linspace(self.info.llb[0], self.info.lub[0], N)
        X, Y = np.meshgrid(x, y)

        f, F = np.zeros((N, N)), np.zeros((N, N))

        for i in range(N):
            for j in range(N):
                u = np.asarray([X[i, j]])
                l = np.asarray([Y[i, j]])
                f[i, j] = self.f_function(u, l)
                F[i, j] = self.F_function(u, l)

        return X, Y, f, F


class Benchmark(ABC):
    functions = {}

    def __init__(self, u_dim=1, l_dim=1):
        self.u_dim = u_dim
        self.l_dim = l_dim

    @abstractmethod
    def generate_random_function(self, seed) -> Function:
        raise NotImplementedError

    def get_function(self, seed):
        name = f'JLr{seed}({self.u_dim}+{self.l_dim})'
        if name not in self.functions:
            fun = self.generate_random_function(seed)
            fun.info.name = name
            self.functions[name] = fun
        return self.functions[name]

    def get_test_functions(self, amount) -> List[Function]:
        functions = [self.get_function(seed) for seed in range(amount)]
        return functions

    def get_train_functions(self, amount) -> List[Function]:
        functions = [self.get_function(seed) for seed in range(10000, 10000 + amount)]
        return functions


class HyperParams:
    def suffix(self):
        repr = str(self)
        return repr[repr.index('('):].replace(' ', '')


class Algorithm(ABC):

    def __init__(self, hyper_params: HyperParams = None):
        name = self.__class__.__name__
        if hyper_params is None:
            self.name = name
        else:
            self.name = f'{name}{hyper_params.suffix()}'
        self.hp = hyper_params

        self.fun: Function = None

    def bind(self, fun: Function):
        self.fun = fun
        fun.reset_max_fe()
        return self

    @abstractmethod
    def optimize(self):
        raise NotImplementedError
