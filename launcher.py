import multiprocessing
from itertools import product

from jl_benchmark import JLBenchmark
from nested_de import NestedDe
from db import save
from utils import seed_every_where


def worker(fun, alg, seed):
    seed_every_where(seed)
    fun.reset_max_fe()
    best_x = alg.bind(fun).optimize()
    value = fun.F(best_x, fun.info.y_star(best_x))
    save(fun.info.name, alg.name, seed, value)


def main(u_dim, l_dim):
    jobs = []
    funcs = JLBenchmark(u_dim, l_dim).get_test_functions(12)
    algs = [NestedDe()]
    seeds = list(range(31))

    for fun, alg, seed in product(funcs, algs, seeds):
        p = multiprocessing.Process(target=worker, args=(fun, alg, seed))
        jobs.append(p)
        p.start()

    for j in jobs:
        j.join()


if __name__ == '__main__':
    main(2, 3)
    main(5, 5)
    main(10, 10)
