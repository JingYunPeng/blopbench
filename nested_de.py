import numpy as np

from base import Algorithm, HyperParams, ExhaustedCallException
from jl_benchmark import JLBenchmark
from utils import seed_every_where


class DeHp(HyperParams):
    def __init__(self,
                 u_pop=15,
                 l_pop=20,
                 uF=0.7,
                 uCR=0.8,
                 lF=0.8,
                 lCR=0.9):
        self.u_pop = u_pop
        self.l_pop = l_pop
        self.uF = uF
        self.uCR = uCR
        self.lF = lF
        self.lCR = lCR


class NestedDe(Algorithm):

    def de(self, fun, lb, ub, dim, pop_size, max_fe, F, CR, args=()):
        pop = np.random.uniform(lb, ub, size=(pop_size, dim))
        fit = np.asarray([fun(*args, x) for x in pop])
        fe = 0
        exhausted = False
        while not exhausted and fe < max_fe:
            for i in range(pop_size):
                idxs = [idx for idx in range(pop_size) if idx != i]
                r1, r2, r3 = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(r1 + F * (r2 - r3), lb, ub)
                cp = np.random.rand(dim) < CR
                if not np.any(cp):
                    cp[np.random.randint(0, dim)] = True
                trial = np.where(cp, mutant, pop[i])
                try:
                    fv = fun(*args, trial)
                except ExhaustedCallException:
                    exhausted = True
                    break
                if fv < fit[i]:
                    pop[i], fit[i] = trial, fv
            fe += pop_size
        best_idx = np.argmin(fit)
        return pop[best_idx]

    def optimize(self):
        info = self.fun.info
        self.hp = DeHp()
        l_max = 1000 * info.l_dim ** 2

        def _inner(x):
            y = self.de(
                self.fun.f,
                info.llb,
                info.lub,
                info.l_dim,
                self.hp.l_pop,
                l_max,
                self.hp.lF,
                self.hp.lCR,
                args=(x,),
            )
            return self.fun.F(x, y)

        return self.de(
            _inner,
            info.ulb,
            info.uub,
            info.u_dim,
            self.hp.u_pop,
            int(1e20),
            self.hp.uF,
            self.hp.uCR,
        )


if __name__ == '__main__':
    solver = NestedDe()
    funs = JLBenchmark(2, 3).get_test_functions(12)
    seed_every_where(0)
    for f in funs:
        solver.bind(f)
        r = solver.optimize()
        print(f.nfev)
        print(r)
        print()
