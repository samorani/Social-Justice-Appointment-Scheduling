# -*- coding: utf-8 -*-
__author__ = "Michele Samorani"

import pandas as pd
import cplex
import time
import random

TIME_LIMIT_SECONDS = 60
def build_scenarios(show_probs, max_scenarios,seed):
    """
    Builds the scenarios
    :param show_probs:
    :type show_probs: list[float]
    :return: a list of (probability, 0-1 show list)
    """
    random.seed(seed)
    n = len(show_probs)
    if 2 ** n <= max_scenarios:
        import itertools
        lst = [list(i) for i in itertools.product([0, 1], repeat=n)]
        for s in lst:
            p = 1
            for j in range(n):
                p *= (show_probs[j] if s[j] == 1 else 1 - show_probs[j])

            yield p,s
    else:

        s = show_probs.copy()
        for i in range(max_scenarios):
            for j in range(n):
                p2 = random.uniform(0, 1)
                s[j] = 1 if p2 < show_probs[j] else 0
            yield 1 / max_scenarios, s.copy()

        # s = show_probs.copy()
        # for i in range(max_scenarios):
        #     for j in range(n):
        #         p2 = random.uniform(0, 1)
        #         s[j] = 1 if p2 < show_probs[j] else 0
        #     p = 1
        #     for j in range(n):
        #         p *= (show_probs[j] if s[j] == 1 else 1 - show_probs[j])
        #
        #     # input(f'returning {str(p)}->{str(s)}')
        #     yield p, s.copy()


def optimally_schedule(show_probs, wtc, otc, nslots,seed, max_scenarios = 100000, delta_sim = 0):
    print_steps = False
    # First, find the scenarios
    qs = [] # a list of sets of patients that show under a scenario
    ps = [] # a list of probabilities
    init = time.time()

    ser = pd.Series(data=show_probs)
    sorted_indices = list(ser.sort_values().index)

    # Similar index (for each index i, the index of the other patient for constraint 4)
    similar = {}
    for iii in range(len(sorted_indices)-1):
        i = sorted_indices[iii]
        j = sorted_indices[iii+1]
        # check whether i is similar to j
        if show_probs[j] - show_probs[i] <= delta_sim + 0.00000001:
            similar[i] = j
        else:
            similar[i] = -1

    similar[sorted_indices[-1]] = -1

    if print_steps:
        print('Building scenarios')
    totp = 0
    for p,s in build_scenarios(show_probs, max_scenarios,seed):
        qs.append(set()) # set of showing indices
        ps.append(p)
        totp+=p
        for i in range(len(s)):
            if s[i] == 1:
                qs[-1].add(i)
    #print(f'totp={totp}')
    # if abs(totp-1) > 0.01:
    #     input('TOT P < 1!!!!!!')
    S = len(qs) # number of scenarios
    F = nslots # number of slots
    N = len(show_probs) # number of patients
    F_max = N
    if print_steps:
        print(f'Done in {time.time() - init}. Built {S} scenarios. Setting up problem...')
    c = cplex.Cplex()
    # variables
    c.variables.add(names=[f'x{i}_{j}' for i in range(N) for j in range(F)],types=[c.variables.type.binary for i in range(N) for j in range(F)])
    c.variables.add(names=[f'b{s}_{j}' for j in range(F_max) for s in range(S)],lb=[0 for j in range(F_max) for s in range(S)])
    c.set_log_stream(None)
    c.set_results_stream(None)
    c.set_warning_stream(None)
    c.parameters.timelimit.set(TIME_LIMIT_SECONDS)
    # objective
    if print_steps:
        print(f'Setting up objective...')
    for s in range(S):
        tot_shows = len(qs[s]) #N^s
        #print(f'Scenario {s} with probability {ps[s]} and tot_shows = {tot_shows}:')
        #print(qs[s])

        if tot_shows == 0:
            continue
        for j in range(F_max):
            #print(f'scenario {s}, j={j}: adding b{s}_{j} * (ps_s={ps[s]}) * (wtc={wtc}) / (tot_shows={tot_shows})')
            c.objective.set_linear(f'b{s}_{j}',ps[s] * wtc)

        c.objective.set_linear(f'b{s}_{F-1}', ps[s] * (otc + wtc))
        #print(f'scenario {s}: adding b{s}_{F-1} * (ps_s={ps[s]}) * (otc={otc})')

    # constraint set (1)
    if print_steps:
        print(f'Setting up constraint set 1...')
    for i in range(N):
        c.linear_constraints.add(lin_expr=[cplex.SparsePair(
            ind = [f'x{i}_{j}' for j in range(F)], val = [1.0 for j in range(F)])],
            senses = ['E'],
            rhs=[1],
            names=[f'(1_{i})'])

    # constraint set (2)
    if print_steps:
        print(f'Setting up constraint set 2...')
    for s in range(S):
        if print_steps and s % 1000 == 0:
            print(f'Built constraints for {s} scenarios')
        for j in range(0,F_max):
            expr = []
            if j < F:
                expr = [f'x{i}_{j}' for i in qs[s]]
            expr.append(f'b{s}_{j}')
            if j >= 1:
                expr.append(f'b{s}_{j-1}')
            vals = []
            if j <F:
                vals = [-1.0 for i in qs[s]]
            vals.append(1)
            if j >=1 :
                vals.append(-1)
            c.linear_constraints.add(lin_expr=[cplex.SparsePair(expr,vals)],
                senses=['G'],
                rhs=[-1],
                names=[f'(2_{s}_{j})'])

    # constraint set (3)
    if print_steps:
        print(f'Setting up constraint set 3...')


    # original constraint 3
    if (N >= F):
        for j in range(0, F):
            c.linear_constraints.add(lin_expr=[cplex.SparsePair(
                ind=[f'x{i}_{j}' for i in range(N)], val=[1.0 for i in range(N)])],
                senses=['G'],
                rhs=[1],
                names=[f'(3_{j})'])

    # constraint set (4)
    if print_steps:
        print(f'Setting up constraint set 4...')
    for i1 in range(N):
        i2 = similar[i1]
        if i2 == -1:
            continue
        for j_prime in range(F-1):
            expr = []
            vals = []

            # old and faster
            expr = [f'x{i1}_{j}' for j in range(j_prime+1,F)]

            # new and slower
            #expr = [f'x{i1}_{j_prime}']

            # expr.extend([f'x{i2}_{j}' for j in range(0,j_prime+1)])
            # vals = [1 for i in range(len(expr))]
            # c.linear_constraints.add(lin_expr=[cplex.SparsePair(expr, vals)],
            #                          senses=['L'],
            #                          rhs=[1],
            #                          names=[f'(4_{i1}_{j_prime})'])

    #c.write(filename='model.txt', filetype='lp')
    if print_steps:
        print(f'Solving...')
    c.solve()
    time_taken = time.time() - init

    # c.solution.write('solution.txt')
    #print(f'Value = {c.solution.get_objective_value()}')
    solution = []
    try:
        for i in range(N):
            sols = c.solution.get_values([f'x{i}_{j}' for j in range(F)])
            for j in range(F):
                if sols[j] >= .9:
                    solution.append(j)
                    break
    except:
        import numpy as np
        return np.nan, np.nan, np.nan, np.nan
    return c.solution.get_objective_value(),c.solution.MIP.get_mip_relative_gap(), solution, time_taken
