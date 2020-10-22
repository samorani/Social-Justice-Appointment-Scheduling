from sklearn.cluster import KMeans
import cplex as cp
import stochastic as stoch
import stochastic2 as stoch2
import race_unaware_stochastic as race_unaware_stoch
import pandas as pd
import numpy as np
np.random.seed(0)



###################################################
######### UTILITY FUNCTIONS ########################
###################################################


def compute_actual_expected_value(solution, actual_show_probs,id_to_group, wtc,otc,itc,gapc, n_slots):
    """
    Generate all scenarios and then compute the weighted avg of the actual value of each scenario
    :param solution:
    :param actual_show_probs:
    :param wtc:
    :param otc:
    :param itc:
    :param id_to_group: a list, for each id the group
    :param gapc: gapcost
    :param n_slots:
    :return:
    """

    # build scenarios
    n = len(actual_show_probs)
    actual_value = 0
    actual_wt = 0
    actual_ot = 0
    actual_it = 0
    group_to_ids = {}
    actual_wts = [0 for i in range(n)]
    for id in range(len(id_to_group)):
        g = id_to_group[id]
        if g not in group_to_ids:
            group_to_ids[g] = []
        group_to_ids[g].append(id)

    for p_s, actual_shows in stoch.build_scenarios(actual_show_probs, 2 ** n,1):
        value, wt, ot, it ,wts = compute_actual_value(solution, actual_shows,id_to_group,group_to_ids, wtc,otc,itc,gapc, n_slots)
        #print('==============\np_s='+str(p_s)+'\tvalue='+str(value))
        #print (actual_shows)
        actual_value += p_s * value
        actual_wt += p_s * wt
        actual_ot += p_s * ot
        actual_it += p_s * it
        for i in range(n):
            if ~np.isnan(wts[i]):
                actual_wts[i] += p_s * wts[i]
    return actual_value, actual_wt, actual_ot,actual_it, actual_wts

def compute_actual_value(solution, actual_shows,id_to_group,group_to_ids, wtc,otc,itc,gapc, n_slots):
    """
    Computes the actual value of a schedule
    :param solution: a vector of assignment of people into slots
    :param wtc: the waiting time cost incurred if every showing patient waits one time unit
    :param actual_shows: a vector of patients. 1=show, 0=no-show
    :param id_to_group: a vector of groups, one per patient
    :group_to_ids: a dict: for each group, the list of patients
    :return:
    """
    
    # first convert to schedule format
    schedule = [[] for i in range(n_slots)]
    tot_shows = 0
    for p in range(len(solution)):
        if actual_shows[p] == 0:
            continue
        slot_p = solution[p]
        tot_shows+=1
        schedule[slot_p].append(p)
    #print(schedule)

    # if tot_shows == 0:
    #     return 0, 0, 0,0,[np.nan for i in range(len(solution))]

    # then, for each slot, calculate the queue, etc
    tot_w = 0 # total waiting time over all slots and patients
    tot_it = 0 # total idle time
    queue = []
    wts = [0 for i in range(len(solution))]
    for i in range(n_slots):
        if len(queue) + len(schedule[i]) < 0.001:
            tot_it += 1 * itc
        # patients arrive
        queue.extend(schedule[i])
        # one patient is seen
        if len(queue) > 0:
            queue.pop(0)
        # whoever is in queue, pays wt
        for p in queue:
            wts[p] += 1 
        tot_w += len(queue) * wtc
        tot_ot = len(queue) * otc
    remaining_patients = len(queue)
    for i in range(0,remaining_patients):
        queue.pop(0)
        for p in queue:
            wts[p] += 1 
        tot_w += len(queue) * wtc
        #print(f'wtc = {tot_w}, otc = {tot_ot}')

    for p in range(len(solution)):
        if actual_shows[p] == 0:
            wts[p] = np.nan
    # for each group of patient in the same slot, divide their waiting time
    for slot in range(len(schedule)):
        sum_wt_in_slot = 0
        shows_in_slot = 0
        for p in schedule[slot]:
            if actual_shows[p]:
                sum_wt_in_slot += wts[p]
                shows_in_slot += 1
        for p in schedule[slot]:
            if actual_shows[p]:
                wts[p] = sum_wt_in_slot / shows_in_slot
            
    tot_cost = tot_ot + tot_w + tot_it
    return tot_cost,  tot_w ,tot_ot,  tot_it, wts


def vector_to_string(l):
    s = ''
    for e in l:
        try:
            s += str(round(e,3)) + ','
        except:
            return l
    return s[:-1]

###################################################
######### TUTORIAL ########################
###################################################

# A tutorial on how to schedule appointments given individual show probabilities
# This problem is about scheduling 5 patients into 4 slots

wtc = .5 # waiting time cost coefficient (i.e., the cost of one time unit of patient waiting time)
otc = 1.5 # overtime cost coefficient (i.e., the cost of one time unit of provider overtime)
itc = 1 # idle time cost coefficient (i.e., the cost of one time unit of provider idle time)

# the patients' individual show probabilities. Edit them with your patients' predicted show probabilities 
show_probs =[.85,.75,.7,.7, .6]
print(f'The patients\' show probabilities are {show_probs}')

F = 4 # F is the number of slots
N = len(show_probs) # N is the number of patients

# the racial groups of the patients (G1 = racial group at higher risk of no-show, G2 = the other racial group)
groups = ['G2','G2','G1','G1', 'G1']

unique_groups = pd.Series(groups)
X = np.array([n for n in show_probs])

for method in ['state-of-the-art','race-aware','race-unaware']:
    print(f'============ {method} ============')
    if method == 'state-of-the-art':
        # will likely result in lowest cost but highest racial disparity
        ofv, opt_gap, sol, time = stoch.optimally_schedule(show_probs, wtc, otc + itc, F, 0,2**N, 0)
    elif method =='race-aware':
        ofv, opt_gap, sol,wts, time = stoch2.optimally_schedule_reduced_bias(show_probs,groups, 0.0001, otc,wtc+itc,  F, 0,
                                                   2 ** N, 0)
    else:
        ofv, opt_gap, sol,wts, time = race_unaware_stoch.optimally_schedule_race_unaware('N', show_probs,groups, 0.0001, otc,wtc+itc,  F, 0,
                                                   2 ** N, 0)
    actual_cost, actual_wt, actual_ot,actual_it,actual_wts  = compute_actual_expected_value(sol, show_probs,groups, wtc,otc,itc,0, F)


    print(f'Solution = {sol} (the i-th element is the slot number 0 to {F-1} where to schedule the i-th patient). \nTot wt = {round(actual_wt,3)} (the sum of all patients\' waiting time, with time unit = one appointment slot), \nwts = {[round(i,3) for i in actual_wts]} (waiting time of the individual patients),\n ot = {round(actual_ot,3)} (the overtime), cost = {round(actual_cost,3)} (the clinic cost)')

    actual_cost, actual_wt, actual_ot,actual_it,actual_wts  = compute_actual_expected_value(sol, show_probs,groups, wtc,otc,itc,0, F)
    wt_by_group = {}
    for gr in unique_groups:
        wt_by_group[gr] = 0
        n_gr=0
        for i in range(N):
            if groups[i] == gr:
                n_gr+=show_probs[i]
                wt_by_group[gr] += actual_wts[i]
        wt_by_group[gr] = round(wt_by_group[gr] / n_gr,3)

    print (f'Average times by race: {wt_by_group}')
exit(0)
