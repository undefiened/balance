"""
File description:
-----------------
This file contains functions that can be used to perform sanity checks on solutions
found by the two methods. This is necessary because a method may produce a solution
where it is difficult to manually ensure its correctness, and it may be wrong.
"""
import collections
from typing import List

# function to check that times are strictly increasing
#  to check that time + edge.weight = next_time

# function to check that vertiport capacity not exceeded

# function calling both above functions to perform sanity check


import numpy as np

from constants import DEBUG
import utils


class ValidSolution:
    """An object that represents whether a method has a valid solution."""
    __slots_ = ['method', 'time_correct', 'capacity_correct', 'cycles_exists']

    def __init__(self, method, time_correct, capacity_correct, cycles):
        self.method = method
        self.time_correct = time_correct
        self.capacity_correct = capacity_correct
        self.cycles_exists = cycles

    def __repr__(self):
        return (f'ValidSolution(method={self.method}, time_correct={self.time_correct}, '
                f'capacity_correct={self.capacity_correct}, cycles={self.cycles_exists})')

    def __str__(self):
        return f"{self.time_correct and self.capacity_correct and not self.cycles_exists}"


def time_correctness(path: list, edges: dict, time_delta: int) -> bool:
    """
    Performs checks to ensure that the departing and arriving times are correct,
    that's that time is strictly increasing and departure time + edge weight equals
    next arrival time.

    Parameters
    ----------
    path: list
        A list of utils.Link objects representing a drone path. Each tuple is of the form
        (node, arrival_layer, arrival_time, right_most_reserved_layer)
    edges: dict
        The edges dict.
    time_delta: int
        Time discretization delta.

    Returns
    -------
        time_correct: bool
            Boolean, whether times are correct or not.

    """
    arrival_times = np.array([link.travel_time for link in path[1:]])
    strictly_increasing = np.all(np.diff(arrival_times) > 0)

    valid_arrivals = True
    travel_time = 0

    for i in range(len(path[:-1])):
        dep_link = path[i]
        arr_link = path[i+1]
        dep_node, arr_node = dep_link.name, arr_link.name
        dep_time, arr_time = dep_link.travel_time, arr_link.travel_time

        if dep_node == arr_node:
            w = time_delta
        else:
            try:
                w = edges[(dep_node, arr_node)].weight
            except KeyError as _:
                # no such edge exists
                return False
            k, r = divmod(w, time_delta)
            w = (k + (r > 0)) * time_delta

        travel_time += w
        valid_arrivals = valid_arrivals and travel_time == arr_time

    time_correct = strictly_increasing and valid_arrivals
    return time_correct


def capacity_correctness(intents: dict, nodes: dict, time_delta: int, time_horizon: int) -> bool:
    """
    Checks whether any vertiport is overcapacitated at any point in the time planning.

    Parameters
    ----------
    intents: dict
        The intents dict.
    nodes: dict
        The nodes dict.
    time_delta: int
        Time discretization delta.
    time_horizon: int
        Planning time horizon.

    Returns
    -------
        greedy_capacity_correct: bool
            Boolean, whether capacities for greedy are valid.
        ip_capacity_correct: bool
            Boolean, whether capacities for ip are valid.

    """

    n_vertiports, n_layers = len(nodes), time_horizon//time_delta

    reservations_greedy = np.zeros((n_vertiports, n_layers+1))
    reservations_ip = np.zeros((n_vertiports, n_layers+1))

    names = ['GREEDY', 'INTEGER PROGRAMMING']
    J = 0

    for idx, intent in enumerate(intents.values()):
        path_greedy, path_ip = intent.path_greedy, intent.path_ip

        for path, reservations in zip([path_greedy, path_ip], [reservations_greedy, reservations_ip]):
            if DEBUG:
                print(names[J])
                J = (J+1)%2
                print(f"{idx}: Drone={intent}", flush=True)
                print(f"path={path}", flush=True)
                print(f"{'='*100}")
            
            for i in range(len(path[:-1])):
                dep_link, arr_link = path[i], path[i + 1]
                dep_node, arr_node = dep_link.name, arr_link.name

                left = arr_link.left_reserved_layer
                right = arr_link.right_reserved_layer

                # no reservation for ground delay
                if dep_node == arr_node:
                    left = arr_link.layer
                    right = arr_link.layer-1

                vertiport_id = int(arr_node[1:])

                # in case left reserved layer is not determined (which can happen for the last intent), set manually
                if not left and right:
                    left = dep_link.layer + 1

                reservations[vertiport_id][range(left, right+1)] += 1

    vertiport_capacities = np.expand_dims(np.array([node.capacity for node in nodes.values()]), 1)
    greedy_capacity_correct = np.all(vertiport_capacities - reservations_greedy >= 0)
    ip_capacity_correct = np.all(vertiport_capacities - reservations_ip >= 0)

    return greedy_capacity_correct, ip_capacity_correct


def cycles_exists(path: List[utils.Link]):
    """
    Checks whether cycles occur in a path. Ground delay at source node is not considered as a cycle.
    For each other vertiport, if it occurs twice in the path it means that a cycle exists.

    Parameters
    ----------
    path: List[utils.Link]
        Path of an intent.

    Returns
    -------
    cycle_exist: bool

    """
    vertiports = [link.name for link in path]
    vertiports_occurrence_count = collections.Counter(vertiports)
    source_vertiport = path[0].name

    # check for cycles among non-source vertiports
    for vertiport, occurrence in vertiports_occurrence_count.items():
        if vertiport != source_vertiport and occurrence > 1:
            return True

    # check for a cycle for the source vertiport, which could occurr
    # if the drone goes back to source from some other vertiport.
    for i in range(1, len(path)):
        prev, curr = path[i-1].name, path[i].name
        if prev != source_vertiport and curr == source_vertiport:
            return True

    return False


def sanity_check(intents: dict, nodes: dict, edges: dict, time_delta: int, time_horizon: int) -> bool:
    """
    Performs sanity check on a drone path found by some method. The checks
    are both time and capacity related. If any of these checks fails, then
    False is returned, else True.

    Parameters
    ----------
    intents: dict
        The intents dict.
    nodes: dict
        The nodes dict.
    edges: dict
        The edges dict.
    time_delta: int
        Time discretization delta.
    time_horizon: int
        Planning time horizon.

    Returns
    -------
        greedy_vs: ValidSolution
            An instance of ValidSolution.
        ip_vs: ValidSolution
           An instance of ValidSolution.

    """
    greedy_time_correct = True
    ip_time_correct = True

    greedy_cycles_not_exist = True
    ip_cycles_not_exist = True

    for intent in intents.values():
        greedy_correct = time_correctness(intent.path_greedy, edges, time_delta)
        ip_correct = time_correctness(intent.path_ip, edges, time_delta)
        greedy_time_correct = greedy_time_correct and greedy_correct
        ip_time_correct = ip_time_correct and ip_correct

        if len(intent.path_greedy) > 0:
            greedy_cycle = cycles_exists(intent.path_greedy)
            greedy_cycles_not_exist = greedy_cycles_not_exist and not greedy_cycle
        if len(intent.path_ip) > 0:
            ip_cycle = cycles_exists(intent.path_ip)
            ip_cycles_not_exist = ip_cycles_not_exist and not ip_cycle

    greedy_capacity_correct, ip_capacity_correct = capacity_correctness(intents, nodes, time_delta, time_horizon)

    greedy_vs = ValidSolution(method='Greedy', time_correct=greedy_time_correct,
                              capacity_correct=greedy_capacity_correct, cycles=not greedy_cycles_not_exist)
    ip_vs = ValidSolution(method='IP', time_correct=ip_time_correct,
                          capacity_correct=ip_capacity_correct, cycles=not ip_cycles_not_exist)

    return greedy_vs, ip_vs


def sanity_check_with_solutions(intents: dict, nodes: dict, edges: dict, time_delta: int, time_horizon: int,
                               greedy_solution, ip_solution) -> tuple:
    """
    Performs sanity check on drone paths found by the methods using solution objects.
    The checks are both time and capacity related. If any of these checks fails, then
    False is returned, else True.

    Parameters
    ----------
    intents: dict
        The intents dict.
    nodes: dict
        The nodes dict.
    edges: dict
        The edges dict.
    time_delta: int
        Time discretization delta.
    time_horizon: int
        Planning time horizon.
    greedy_solution: solution.GreedySolution
        The greedy solution object.
    ip_solution: solution.IPSolution
        The IP solution object.

    Returns
    -------
        greedy_vs: ValidSolution
            An instance of ValidSolution.
        ip_vs: ValidSolution
           An instance of ValidSolution.

    """
    greedy_time_correct = True
    ip_time_correct = True

    greedy_cycles_not_exist = True
    ip_cycles_not_exist = True

    for intent_key, intent in intents.items():
        intent_key_tuple = (intent.source.name, intent.destination.name, intent.start)
        
        # Get solutions for this intent
        greedy_sol = greedy_solution.get_solution(intent_key_tuple)
        ip_sol = ip_solution.get_solution(intent_key_tuple)
        
        # Check greedy solution
        if greedy_sol and greedy_sol.solution_found and len(greedy_sol.path) > 0:
            greedy_correct = time_correctness(greedy_sol.path, edges, time_delta)
            greedy_time_correct = greedy_time_correct and greedy_correct
            
            greedy_cycle = cycles_exists(greedy_sol.path)
            greedy_cycles_not_exist = greedy_cycles_not_exist and not greedy_cycle
        
        # Check IP solution
        if ip_sol and ip_sol.solution_found and len(ip_sol.path) > 0:
            ip_correct = time_correctness(ip_sol.path, edges, time_delta)
            ip_time_correct = ip_time_correct and ip_correct
            
            ip_cycle = cycles_exists(ip_sol.path)
            ip_cycles_not_exist = ip_cycles_not_exist and not ip_cycle

    # Check capacity correctness using solutions
    greedy_capacity_correct, ip_capacity_correct = capacity_correctness_with_solutions(
        intents, nodes, time_delta, time_horizon, greedy_solution, ip_solution)

    greedy_vs = ValidSolution(method='Greedy', time_correct=greedy_time_correct,
                              capacity_correct=greedy_capacity_correct, cycles=not greedy_cycles_not_exist)
    ip_vs = ValidSolution(method='IP', time_correct=ip_time_correct,
                          capacity_correct=ip_capacity_correct, cycles=not ip_cycles_not_exist)

    return greedy_vs, ip_vs


def capacity_correctness_with_solutions(intents: dict, nodes: dict, time_delta: int, time_horizon: int,
                                       greedy_solution, ip_solution) -> tuple:
    """
    Checks whether any vertiport is overcapacitated at any point in the time planning
    using solution objects.

    Parameters
    ----------
    intents: dict
        The intents dict.
    nodes: dict
        The nodes dict.
    time_delta: int
        Time discretization delta.
    time_horizon: int
        Planning time horizon.
    greedy_solution: solution.GreedySolution
        The greedy solution object.
    ip_solution: solution.IPSolution
        The IP solution object.

    Returns
    -------
        greedy_capacity_correct: bool
            Boolean, whether capacities for greedy are valid.
        ip_capacity_correct: bool
            Boolean, whether capacities for ip are valid.

    """
    n_vertiports, n_layers = len(nodes), time_horizon//time_delta

    reservations_greedy = np.zeros((n_vertiports, n_layers+1))
    reservations_ip = np.zeros((n_vertiports, n_layers+1))

    names = ['GREEDY', 'INTEGER PROGRAMMING']
    J = 0

    for idx, intent in enumerate(intents.values()):
        intent_key = (intent.source.name, intent.destination.name, intent.start)
        
        # Get solutions for this intent
        greedy_sol = greedy_solution.get_solution(intent_key)
        ip_sol = ip_solution.get_solution(intent_key)
        
        path_greedy = greedy_sol.path if greedy_sol and greedy_sol.solution_found else []
        path_ip = ip_sol.path if ip_sol and ip_sol.solution_found else []

        for path, reservations in zip([path_greedy, path_ip], [reservations_greedy, reservations_ip]):
            if DEBUG:
                print(names[J])
                J = (J+1)%2
                print(f"{idx}: Drone={intent}", flush=True)
                print(f"path={path}", flush=True)
                print(f"{'='*100}")
            
            if len(path) == 0:
                continue
                
            for i in range(len(path[:-1])):
                dep_link, arr_link = path[i], path[i + 1]
                dep_node, arr_node = dep_link.name, arr_link.name

                left = arr_link.left_reserved_layer
                right = arr_link.right_reserved_layer

                # no reservation for ground delay
                if dep_node == arr_node:
                    left = arr_link.layer
                    right = arr_link.layer-1

                vertiport_id = int(arr_node[1:])

                # in case left reserved layer is not determined (which can happen for the last intent), set manually
                if not left and right:
                    left = dep_link.layer + 1

                reservations[vertiport_id][range(left, right+1)] += 1

    vertiport_capacities = np.expand_dims(np.array([node.capacity for node in nodes.values()]), 1)
    greedy_capacity_correct = np.all(vertiport_capacities - reservations_greedy >= 0)
    ip_capacity_correct = np.all(vertiport_capacities - reservations_ip >= 0)

    return greedy_capacity_correct, ip_capacity_correct


# =============================================== END OF FILE ===============================================
