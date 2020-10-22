# -*- coding: utf-8 -*-
__author__ = "Michele Samorani"

import pandas as pd
import numpy as np
import cplex
import time
import random
import stochastic2 as stoch2
from sklearn.cluster import KMeans

def optimally_schedule_race_unaware(unaware_strategy, show_probs, groups, wtc, otc,bc, nslots,seed, max_scenarios = 100000, delta_sim = 0):
    
    # UOF N
    if unaware_strategy == 'N':
        new_groups = [str(q) for q in range(len(show_probs))]
    elif unaware_strategy == '2':
        # UOF 2
        # new_groups = pd.Series(show_probs).sort_values()
        # mid_point = int(len(new_groups)/2)
        # new_groups.iloc[:mid_point]= 'L'
        # new_groups.iloc[mid_point:]= 'H'
        # new_groups.sort_index(inplace=True)
        X = np.array([[n] for n in show_probs])
        kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
        new_groups = [str(i) for i in kmeans.labels_]
    else:
        raise('set unware_strategy')

    ofv, opt_gap, sol,wts, time = stoch2.optimally_schedule_reduced_bias(show_probs,new_groups, wtc, otc, bc, nslots, seed,
                                            max_scenarios, delta_sim)

    return ofv, opt_gap, sol,wts, time
