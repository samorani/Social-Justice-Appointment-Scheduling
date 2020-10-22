# -*- coding: utf-8 -*-
__author__ = "Michele Samorani"
debug = False
from queue import Queue
from sortedcontainers import SortedDict
import numpy as np
import pandas as pd

# class Appointment_Set:
#     # a set of appointments with the same show probability and group
#     def __init__(self, probability, group):
#         self.probability = probability
#         self.group = group
#         self.indices = set()
#         self.n = 0

#     def copy(self):
#         a = Appointment_Set(self.probability, self.group)
#         a.indices = self.indices.copy()
#         a.n = self.n
#         return a        

#     def add(self, index):
#         self.indices.add(index)
#         self.n += 1
 
#     def remove(self, how_many):
#         # if how_many > len(self.indices):
#         #     print('error')
#         # toret = self.indices[:how_many].copy()
#         # self.indices = self.indices[how_many:]
#         toret = [self.indices.pop() for i in range(how_many)]
#         self.n = len(self.indices)
#         return toret

#     def remove_id(self, id):
#         if id in self.indices:
#             self.indices.remove(id)
#             self.n -= 1



# def enumerate_patient_subsets(cur_result, left_to_choose, sets, min_prob = 0):
#     """ pick patients among the sets. cur_result is a list of how many patients for each set. cur_result is the index of the set to consider next """
#     # if it is the last set, pick left_to_choose from the current set
#     cur_index = len(cur_result)
#     if cur_index == len(sets) - 1:
#         if sets[cur_index].n >= left_to_choose and sets[cur_index].probability >= min_prob:
#             cur_result.append(left_to_choose)
#             yield cur_result.copy()
#             cur_result = cur_result[:-1]
#         else:
#             pass
#             # input(f'cur_index = {cur_index}, cur_result = {" ".join([str(s) for s in cur_result])}, cannot complete with {left_to_choose} left to choose')


#     else:
#         curset = sets[cur_index]
#         max_iterations = 1 if curset.probability < min_prob else 1000000
#         n = curset.n
#         # if it is NOT the last set, pick at most left_to_choose from the current set
#         for r in range(0,min([n+1,left_to_choose+1,max_iterations])):
#             # input(f'cur_index = {cur_index}, cur_result = {" ".join([str(s) for s in cur_result])}, thinking of adding {r}')
#             # pick r from the next set
#             curset.n -= r
#             cur_result.append(r)
#             left_to_choose -= r
#             for completed in enumerate_patient_subsets(cur_result.copy(), left_to_choose, sets,min_prob):
#                 yield completed
#             left_to_choose += r
#             curset.n = n
#             cur_result = cur_result[:-1]


# def make_horizontal_segment(patient_ids, cur_patient_assignment, cur_schedule,patient_to_set_index, set_list, j):
#     """ make a horizontal segment starting at j """
#     k = j
#     for id in patient_ids:
#         cur_schedule[k].append(id)
#         set_list[patient_to_set_index[id]].remove_id(id)
#         cur_patient_assignment[id] = k
#         k+=1

# def undo_horizontal_segment(n_patients, cur_patient_assignment, cur_schedule,patient_to_set_index, set_list, j):
#     """undo the segment that starts at j and contains n_patients"""
#     for k in range(j,j+n_patients):
#         id = cur_schedule[k][0]
#         if len(cur_schedule[k]) > 1:
#             input('error! Trying to delete a horizontal segment but there is a vertical segment here')
#         cur_schedule[k].remove(id)
#         set_list[patient_to_set_index[id]].add(id)
#         cur_patient_assignment[id] = -1
#         k+=1

# def make_vertical_segment(patient_ids, cur_patient_assignment, cur_schedule,patient_to_set_index, set_list, j):
#     """ make a vertical segment at j """
#     for id in patient_ids:
#         cur_schedule[j].append(id)
#         set_list[patient_to_set_index[id]].remove_id(id)
#         cur_patient_assignment[id] = j
        
# def undo_vertical_segment(cur_patient_assignment, cur_schedule,patient_to_set_index, set_list, j):
#     """undo the vertical segment at j """
#     for id in cur_schedule[j].copy():
#         cur_schedule[j].remove(id)
#         set_list[patient_to_set_index[id]].add(id)
#         cur_patient_assignment[id] = -1
        

# def extract_ids_from_sets(selection,set_list):
#     """return the list of ids given the selection on set_list and removes them from the set_list
#     selection[i] has the number of patients from set i"""
#     ids = []
#     for i in range(len(selection)):
#         n = selection[i]
#         if n == 0:
#             continue
#         s = set_list[i]
#         removed = s.remove(n)
#         ids.extend(removed)
#     return ids

# def copy_list_of_sets(list_of_sets):
#     ret = []
#     for s in list_of_sets:
#         ret.append(s.copy())
#     return ret

# def print_set_list(set_list):
#     for i in range(len(set_list)):
#         print(f'set {i}. Indices = {",".join([str(s) for s in set_list[i].indices])}, n = {set_list[i].n}')

# def complete_schedule(cur_patient_assignment, cur_schedule,show_probs, patient_to_set_index, set_list,  state,min_prob, remaining_patients, j,F):
#     debug = False
#     if j == F:
#         yield cur_patient_assignment
#     else:
#         if state == 'init':
#             sorted_indices = pd.Series(show_probs).sort_values(ascending=False).index
#             for r in range(1,F):
#                 # make an initial  horizontal segment of r patients
#                 if debug:
#                     print_set_list(set_list)
#                     print(f'j={j}, state = {state}, assignment={",".join([str(s) for s in cur_patient_assignment])}, remaining patients = {remaining_patients}. About to make a horizontal segment of length {r}')

#                 # print(f'j = {j}, before making an init-horizontal segment ' + str(','.join([str(i) for i in sorted_indices[:r]])+ f', remaining patients = {remaining_patients}'))
#                 # print_set_list(set_list)
#                 make_horizontal_segment(sorted_indices[:r], cur_patient_assignment, cur_schedule,patient_to_set_index, set_list, 0)
#                 for sc in complete_schedule(cur_patient_assignment, cur_schedule,show_probs, patient_to_set_index, set_list,  'horizontal',0, remaining_patients - r, r,F):
#                     yield sc
#                 undo_horizontal_segment(r,cur_patient_assignment, cur_schedule,patient_to_set_index, set_list, 0)
#                 # print(f'j = {j}, after undoing the init-horizontal segment ' + str(','.join([str(i) for i in sorted_indices[:r]])+ f', remaining patients = {remaining_patients}'))
#                 # print_set_list(set_list)
#         if state == 'vertical':
#             # make a horizontal segment of r
#             if debug:
#                  print(f'j={j}, state = {state}, assignment={",".join([str(s) for s in cur_patient_assignment])}, remaining patients = {remaining_patients}. Checking whether I can make a horizontal segment')
            
#             max_length = F-j-1 if remaining_patients > F-j else F-j
#             # print(f'j = {j}, before attempting to generate horizontal segment, remaining patients = {remaining_patients}, max length = {max_length}')
#             # print_set_list(set_list)
#             for r in range(1,max_length+1):
#                 # pick r patients with minimum probability min_prob
#                 for se in enumerate_patient_subsets([], r, copy_list_of_sets(set_list),min_prob):
#                     # se[i] contains the number of patients from the i-th set
#                     # print(f'j = {j}, making a horizontal with r ={r}, before extracting ids')
#                     # print_set_list(set_list)
#                     patient_ids = extract_ids_from_sets(se,set_list)
#                     # print(f'j = {j}, after extracting ' + str(','.join([str(i) for i in patient_ids])))
#                     # print_set_list(set_list)
#                     ser = pd.Series(show_probs)[patient_ids].sort_values(ascending=False)
#                     sorted_indices = ser.index
#                     prob = ser.max()

#                     # print(f'j = {j}, after making horizontal vertical segment ' + str(','.join([str(i) for i in patient_ids])+ f', remaining patients = {remaining_patients}'))
#                     # print_set_list(set_list)
#                     make_horizontal_segment(sorted_indices, cur_patient_assignment, cur_schedule,patient_to_set_index, set_list, j)
#                     for sc in complete_schedule(cur_patient_assignment, cur_schedule,show_probs, patient_to_set_index, set_list,  'horizontal',prob, remaining_patients - r, j+r,F):
#                         yield sc
#                     undo_horizontal_segment(r,cur_patient_assignment, cur_schedule,patient_to_set_index, set_list, j)
#                     # print(f'j = {j}, after undoing horizontal segment ' + str(','.join([str(i) for i in patient_ids])+ f', remaining patients = {remaining_patients}'))
#                     # print_set_list(set_list)
            
#         # make a vertical if state is vertical or horizontal
#         # then this is the second vertical segment in a row, hence we need to enforce the probability check
#         min_prob_enforced = min_prob if state == 'vertical' else 0
#         if debug:
#             print(f'j={j}, state = {state}, assignment={",".join([str(s) for s in cur_patient_assignment])}, remaining patients = {remaining_patients}. Checking whether I can make a vertical segment with min_prob_enforced = {min_prob_enforced}')
#         min_range = max(2,remaining_patients) if j == F - 1 else 2
#         # print(f'j = {j}, before attempting to generate vertical segment with an enforced probability of ' + str(min_prob_enforced)+ f', remaining patients = {remaining_patients}')
#         # print_set_list(set_list)
#         for r in range(min_range,remaining_patients - (F - j) + 2):
#             # pick r patients with minimum probability min_prob
#             for se in enumerate_patient_subsets([], r, copy_list_of_sets(set_list),min_prob_enforced):
#                 # se[i] contains the number of patients from the i-th set
#                 # print(f'j = {j}, before extracting ids')
#                 # print_set_list(set_list)
#                 patient_ids = extract_ids_from_sets(se,set_list)
#                 # print(f'j = {j}, after extracting ' + str(','.join([str(i) for i in patient_ids])))
#                 # print_set_list(set_list)
#                 if debug:
#                     print(f'j={j}, state = {state}, assignment={",".join([str(s) for s in cur_patient_assignment])}, remaining patients = {remaining_patients}. About to make a vertical segment of length {r}')
#                 max_prob = pd.Series(show_probs)[patient_ids].max()
#                 make_vertical_segment(patient_ids,cur_patient_assignment,cur_schedule,patient_to_set_index, set_list, j)
#                 # print(f'j = {j}, after making vertical segment ' + str(','.join([str(i) for i in patient_ids])+ f', remaining patients = {remaining_patients}'))
#                 # print_set_list(set_list)
#                 for sc in complete_schedule(cur_patient_assignment, cur_schedule,show_probs, patient_to_set_index, set_list, 'vertical',max_prob, remaining_patients - r, j+1,F):
#                     yield sc
#                 # print(f'j = {j}, before undoing vertical segment ' + str(','.join([str(i) for i in patient_ids])+ f', remaining patients = {remaining_patients}'))
#                 # print_set_list(set_list)
#                 undo_vertical_segment(cur_patient_assignment, cur_schedule,patient_to_set_index, set_list, j)
#                 # print(f'j = {j}, after undoing vertical segment ' + str(','.join([str(i) for i in patient_ids])+ f', remaining patients = {remaining_patients}'))
#                 # print_set_list(set_list)

# def enumerate_schedules(N,F,show_probabilities,patient_groups):
#     # first, make the classes
#     distinct_group_labels = {}
#     for i in range(N):
#         label = patient_groups[i] + '_' + str(show_probabilities[i])
#         if label not in distinct_group_labels:
#             distinct_group_labels[label] = Appointment_Set(show_probabilities[i], patient_groups[i])
#         distinct_group_labels[label].add(i)

#     set_list = list(distinct_group_labels.values())
#     patient_to_set_index = {} # for each patient id, the set index (-1 if already scheduled)
#     for i in range(len(set_list)):
#         s = set_list[i]
#         for id in s.indices:
#             patient_to_set_index[id] = i

#     cur_patient_assignment = [-1 for i in show_probabilities]
#     cur_schedule = [[] for i in range(F)]

#     for sc in complete_schedule(cur_patient_assignment, cur_schedule,show_probabilities,patient_to_set_index,set_list,'init',0,N,0,F):
#         yield sc
        
        
class AppointmentRealization:
    def __init__(self, id, scheduled_time, arrival_time, service_time, arg=None):
        self.scheduled_time = scheduled_time
        self.arrival_time = arrival_time
        self.service_time = service_time
        self.arg = arg

class ApptEvent:
    def __init__(self, time, id, type):
        self.time = time
        self.id = id
        self.type = type

    def __str__(self):
        return f'{self.time}: appt {self.id} {self.type}'

class Events:
    def __init__(self):
        self.list_of_events = SortedDict() # for each time, a list of events

    def add_event(self, event):
        if not self.list_of_events.get(event.time):
            self.list_of_events[event.time] = [event]
        else:
            self.list_of_events[event.time].append(event)

    def count_arrivals_at(self, time):
        try:
            li = self.list_of_events[time]
            n = 0
            for e in li:
                if li.type == 'arrival':
                    n+=1
            return n
        except:
            return 0

    def pop_next_event(self):
        li = self.list_of_events.peekitem(0)[1]
        #if debug:
        #    print(self.list_of_events)
        if (len(li) > 1):
            toret = li.pop(0)
            #if debug:
            #    print(f'returning {toret}')
            return toret
        else:
            lil= self.list_of_events.popitem(0)[1]
            toret = lil[0]
            #if debug:
            #    print(f'returning {toret}')
            return toret

    def count(self):
        return len(self.list_of_events)

    def __str__(self):
        s = ''
        for k in self.list_of_events.keys():
            s += (str(k)+": ")
            for e in self.list_of_events[k]:
                s+= str(e) + ', '
        return s

def simulate_events(nominal_end_time, appt_realizations, speed_up_factor = 0.0 ):
    """
    appt_realization: for each id, an appintmnetrealization
    Returns (for each id its waiting time, for each id the queue while in service, overtime) 
    service time is reduced by a speed_up_factor for each patient in the queue
    """

    events = Events()
    last_end = 0
    waiting_times = {}
    queue_while_in_service = {}

    for id in appt_realizations:
        waiting_times[id] = 0
        queue_while_in_service [id] = -1

    # create arrival events
    for id in appt_realizations:
        real = appt_realizations[id]
        ev = ApptEvent(real.arrival_time,id,"arrival")
        events.add_event(ev)
    
    queue = []
    busy_provider = False
    avg_queue_during_service = 0
    last_arrival_time = 0
    # start simulation
    while (events.count() > 0):
        # take the first even and process it
        ev = events.pop_next_event()
        
        #arrival and busy -> add to queue
        if ev.type == "arrival" and busy_provider:
            avg_queue_during_service += (ev.time - last_arrival_time) * len(queue)
            queue.append(ev.id)
            if debug:
                print(f'{ev.time}: {ev.id} has arrived and has been added to queue. Between {last_arrival_time} and {ev.time} there was a queue of {len(queue)-1}. Now the queue has {len(queue)} elements')
            last_arrival_time = ev.time

        #arrival and not busy -> start
        elif ev.type == "arrival" and not busy_provider:
            last_arrival_time = ev.time
            busy_provider = True
            serv_time = appt_realizations[ev.id].service_time * (1 - speed_up_factor * len(queue))
            events.add_event(ApptEvent(ev.time + serv_time, ev.id,"end"))
            waiting_times[ev.id] = 0
            # count arrivals
            if debug:
                print(f'{ev.time}: {ev.id} arrived and started. His waiting time is 0; the queue is 0')

        elif ev.type == "end":
            # check whether there is anybody else in the queue
            last_end = ev.time

            # update average queue
            avg_queue_during_service += (ev.time - last_arrival_time) * len(queue)
            avg_queue_during_service /= appt_realizations[ev.id].service_time
            queue_while_in_service[ev.id] = avg_queue_during_service
            avg_queue_during_service = 0
            last_arrival_time = ev.time

            if len(queue) > 0:
                id_next = queue.pop(0)
                serv_time = appt_realizations[id_next].service_time * (1 - speed_up_factor * len(queue))
                events.add_event(ApptEvent(ev.time + serv_time, id_next,"end"))
                wt = max(ev.time - max(appt_realizations[id_next].scheduled_time, appt_realizations[id_next].arrival_time),0)
                waiting_times[id_next] = wt
                if debug:
                    print(f'{ev.time}: {ev.id} has finished. {id_next} started, his waiting time is {wt}. The queue has {len(queue)} elements')
            else:
                busy_provider = False
                if debug:
                    print(f'{ev.time}: {ev.id} has finished. The provider is idle')


    ot = max(0,last_end - nominal_end_time)

    return waiting_times, queue_while_in_service, ot





def generate_lognormal_samples(mean, stdev, n=1):
    """
    Returns n samples taken from a lognormal distribution, based on mean and
    standard deviation calaculated from the original non-logged population.
    
    Converts mean and standard deviation to underlying lognormal distribution
    mu and sigma based on calculations desribed at:
        https://blogs.sas.com/content/iml/2014/06/04/simulate-lognormal-data-
        with-specified-mean-and-variance.html
        
    Returns a numpy array of floats if n > 1, otherwise return a float
    """
    
    # Calculate mu and sigma of underlying lognormal distribution
    phi = (stdev ** 2 + mean ** 2) ** 0.5
    mu = np.log(mean ** 2 / phi)
    sigma = (np.log(phi ** 2 / mean ** 2)) ** 0.5
    
    # Generate lognormal population
    generated_pop = np.random.lognormal(mu, sigma , n)
    
    # Convert single sample (if n=1) to a float, otherwise leave as array
    generated_pop = generated_pop[0] if len(generated_pop) == 1 else generated_pop
        
    return generated_pop


class Simulation_Result:
    """It represents the result simulations on one solution. It has the average waiting time, crowding, overtime, and idle time observed.
    It also has for each patient id, the average wait time and crowding.
    """
    def __init__(self, patient_assignments, patient_groups):
        self.n_iterations = 0
        self.patient_groups = patient_groups
        self.idle_time_observations = []
        self.overtime_observations = []
        self.wait_time_observations = [] # sum of WT per iteration
        self.wait_time_observations_by_patient = {}
        self.crowding_level_observations = {}
        for id in range(len(patient_assignments)):
            self.wait_time_observations_by_patient[id] = []
            self.crowding_level_observations[id] = []

    def add_observed_clinic_session(self, waiting_times, crowding, idle_time, ot):
        """[summary]
        
        :param waiting_times: the waiting time of each patient
        :type waiting_times: dict
        :param crowding: the crowding experienced by each patient
        :type crowding: dict
        :param idle_time: the total idle time experienced by the provider
        :type idle_time: float
        :param ot: the overtime eperienced by the provider
        :type ot: float
        """
        self.n_iterations += 1
        sumWT = 0
        for id in waiting_times:
            self.wait_time_observations_by_patient[id].append(waiting_times[id])
            self.crowding_level_observations[id].append(crowding[id])
            sumWT += waiting_times[id]
        self.idle_time_observations.append(idle_time)
        self.overtime_observations.append(ot)
        self.wait_time_observations.append(sumWT)

    def make_summary(self):
        """[summary]
        returns the sum of the WT, the OT, the IT, the WT for each group, the crowding for each group
        """       
        sumWT = 0
        OT = 0
        IT = 0
        WT_by_group = {}
        crowding_by_group = {}
        for g in self.patient_groups:
            WT_by_group[g] = 0
            crowding_by_group = 0
        for iter in range(self.n_iterations):
            IT += self.idle_time_observations[iter]
            OT += self.overtime_observations[iter]
        for id in self.wait_time_observations_by_patient:
            obs = self.wait_time_observations_by_patient[id]
            g = self.patient_groups[id]




def compute_simulated_performance (patient_assignments,show_probabilities, patient_groups,slot_length, F, mean_service_time, std_service_time,
    mean_lateness, std_lateness, seed, n_iterations):
    """[summary]
    Compute the simulated performance of a schedule, using lognormal service times and normal lateness
    :param patient_assignments: a vector of length N with the slot number of each patient i=0,...,N-1
    :type patient_assignments: list
    :param patient_groups: a vector of length N with the group of each patient i=0,...,N-1
    :type patient_assignments: list
    :param F: the number of slots
    :type F: int
    """
    np.random.seed(seed)
    res = Simulation_Result(patient_assignments, patient_groups)
    for it in range(n_iterations):
        realizations = {}
        for id in range(len(patient_assignments)):
            realizations[id] = AppointmentRealization(id,slot_length*patient_assignments[id],max(0,slot_length*patient_assignments[id]+
                np.random.normal(mean_lateness, std_lateness)),generate_lognormal_samples(mean_service_time,std_service_time))
        wt, crowding, idle_time, overtime = simulate_events(F * slot_length, realizations)
        res.add_observed_clinic_session(wt,crowding,idle_time,overtime)
    return res

# def find_best_simulated_schedule(n_iterations, show_probabilities, patient_groups, otc, wtc, F):
#     for sc in complete_schedule()