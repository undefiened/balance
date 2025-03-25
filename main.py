"""
File description:
-----------------
This file is used to read a problem, build a queue of operational intents,
use both the greedy approach and integer programming approach to solve the
routing problem and save the results.

"""
import argparse
import itertools
import json
import math
import os
import time
import tracemalloc
from typing import List, Dict, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import mip
import networkx as nx
import numpy as np
import pandas as pd

import checks
from constants import GREEDY_TIME_HORIZON_MULTIPLIER, IP_TIME_HORIZON_MULTIPLIER
import graph
import intent
import optimization
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--seed', required=False, default=2718, help="Setting a seed for reproducibility.")
parser.add_argument('--graph_name', required=True,
                    help=f"Name of graph. Assumes a graph with name `graph_name.json` exists in the graphs folder.")
parser.add_argument('--run_name', required=False, default='',
                    help="A name that can be used to distinguish different runs on the same graph.")
parser.add_argument('--random_intents', required=False, action='store_true', default=False,
                    help="Running the methods on a graph where intents are generated randomly.")
parser.add_argument('--debug', required=False, action='store_true', default=False,
                    help="Whether to run main function on a small graph for debugging purposes.")
parser.add_argument('--num_intents', required=True, help="Number of intents for example.")


# --- global variables ---
TIME_HORIZON = 0
TIME_DELTA = 0


class ResultsAnalysis:
    """Class for storing results related to a run, and plotting them."""
    __slots__ = ['name', 'dataframe', 'num_rows']

    def __init__(self, name: str):
        """Creates an instance. `name` is the example name used for file and plots name."""
        self.name = name
        self.num_rows = 0
        columns = ['num_intents', 'greedy_obj', 'ip_obj', 'greedy_runtime', 'ip_runtime', 'greedy_memory_usage',
                   'ip_memory_usage', 'intents_collection', 'ip_gap', 'greedy_solution_valid', 'ip_solution_valid']
        self.dataframe: pd.DataFrame = pd.DataFrame(columns=columns)

    def add_row(self, num_intents: int, greedy_obj: Union[int, None],
                ip_obj: Union[float, None], greedy_runtime: float, ip_runtime: float,
                greedy_memory_usage: float, ip_memory_usage: float,
                intents_collection: intent.IntentsCollection, ip_gap: float,
                greedy_solution_valid: bool, ip_solution_valid: bool):
        """
        Adds a new row to the dataframe.

        Parameters
        ----------
        num_intents: int
            Number of intents.
        greedy_obj: Union[int, None]
            Greedy objective
        ip_obj: Union[float, None]
            IP objective
        greedy_runtime: float
            Greedy runtime, in seconds.
        ip_runtime: float
            IP runtime, in seconds.
        greedy_memory_usage: float:
            Greedy memory usage, in bytes.
        ip_memory_usage: float
            IP memory usage, in bytes.
        intents_collection: intent.IntentsCollection
            An object storing the intents in the run.
        ip_gap: float
            The model gap of the run.
        greedy_solution_valid: bool
            Whether the solution found by greedy passed the sanity checks.
        ip_solution_valid: bool
            Whether the solution found by ip passed the sanity checks.

        Returns
        -------

        """
        # time is stored in minutes
        greedy_runtime, ip_runtime = round(greedy_runtime/60, 2), round(ip_runtime/60, 2)

        # memory usage is stored in GB
        greedy_memory_usage, ip_memory_usage = round(greedy_memory_usage/10e9, 2), round(ip_memory_usage/10e9, 2)

        self.dataframe.loc[self.num_rows] = [num_intents, greedy_obj, ip_obj, greedy_runtime,
                                             ip_runtime, greedy_memory_usage, ip_memory_usage,
                                             intents_collection, ip_gap, greedy_solution_valid,
                                             ip_solution_valid]

        self.num_rows += 1

    def save(self):
        """Pickles dataframe to local files."""
        if not os.path.exists(f'./results/{self.name}'):
            os.makedirs(f'./results/{self.name}')
            os.makedirs(f'./results/{self.name}/models')

        self.dataframe.to_pickle(f'./results/{self.name}/data.pickle')

        return

    def plot(self):
        """Plots the objectives, runtimes and memory usages of both methods side by side."""
        if not os.path.exists(f'./results/{self.name}'):
            os.makedirs(f'./results/{self.name}')

        plt.style.use("seaborn")

        greedy_color, ip_color = 'mediumaquamarine', 'violet'
        data = [[self.dataframe.greedy_obj, self.dataframe.ip_obj],
                [self.dataframe.greedy_runtime, self.dataframe.ip_runtime],
                [self.dataframe.greedy_memory_usage, self.dataframe.ip_memory_usage]]

        x_label = 'Number of intents'
        y_labels = ['Cost', 'Runtime (min)', 'Memory usage (GB)']
        titles = ['Objectives of the methods', 'Runtimes of the methods', 'Memory usages of the methods']
        plot_names = ['objective', 'runtime', 'memory_usage']
        x_axis = self.dataframe.num_intents.values
        fontsize = 16
        linewidth = 3

        for indx, elem in enumerate(data):
            mask0, mask1 = np.isfinite(elem[0]), np.isfinite(elem[1])
            fig, ax = plt.subplots(1, 1)
            ax.plot(x_axis[mask0], elem[0][mask0], marker='o', ls='-',
                    linewidth=linewidth, color=greedy_color, label='Greedy')

            ax.plot(x_axis[mask1], elem[1][mask1], marker='o', ls='-',
                    linewidth=linewidth, color=ip_color, label='IP')

            ax.set_xlabel(x_label, family='serif', fontsize=fontsize)
            ax.set_ylabel(y_labels[indx], family='serif', fontsize=fontsize)
            ax.set_title(titles[indx], family='serif', fontsize=fontsize)
            ax.set_xticks(x_axis)

            ax.legend()

            fig.savefig(f'./results/{self.name}/{plot_names[indx]}.png')

        return


def read_example(path: str, intents=None) \
        -> Tuple[int, int, int, Union[int, float], List[Dict], List[Dict], List[Dict]]:
    """
    Opens an example file, reads it and returns its content. Program exits if json data format is invalid.
    It will throw assertion errors if the actual data is of incorrect type or value.

    Parameters
    ----------
    path: str
        Path to an example json file. Ex. 'ex1.json'.
    intents: list (optional)
        A list of intents is passed for examples where intents are generated randomly.

    Returns
    -------
    return value: Tuple[int, int, int, List[Dict], List[Dict], List[Dict]]
        A tuple (start_time, time_horizon, time_delta, speed, nodes_list, edges_list, intents_list)

    """
    with open(path, 'r') as f:
        content = f.read()

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON example file at {path}:", e)
        exit()
    else:
        start = data["start"]
        assert start >= 0 and isinstance(start, int), "Start time is either negative or not an integer."

        time_horizon = data["time_horizon"]
        assert time_horizon > 0 and isinstance(time_horizon, int), "Time horizon must be a positive integer."

        time_delta = data["time_delta"]
        assert time_delta > 0 and isinstance(time_delta, int) and time_horizon % time_delta == 0, \
            ("Time delta must be a positive integer and a multiple of it equal the time horizon, "
             "i.e. T = k*delta for an integer k.")

        speed = data["speed"]
        assert speed > 0 and (isinstance(speed, int) or isinstance(speed, float))

        nodes = data["nodes"]
        for node in nodes:
            assert isinstance(node['name'], str) and isinstance(node['capacity'], int) and node['capacity'] >= 0, (
                f"{node}: Either syntax error in name, capacity being non integer or negative.")

        edges = data["edges"]
        for edge in edges:
            assert isinstance(edge['source'], str) and isinstance(edge['destination'], str) and \
                   isinstance(edge['weight'], int) and edge['weight'] >= 0, (
                f"{edge}: Either syntax error in edge names, weight being non integer or negative.")

        if intents is None:
            intents_list = data["intents"]
        else:
            intents_list = intents
        for op_intent in intents_list:
            assert isinstance(op_intent['source'], str) and isinstance(op_intent['destination'], str) \
                   and isinstance(op_intent['start'], int) and op_intent['start'] >= start \
                   and isinstance(op_intent['uncertainty'], int) and op_intent['uncertainty'] >= 0, (
                f"{op_intent}: Either syntax error in names, start time being non integer, it starts before "
                f"operations start time or time uncertainty is negative/non-integer.")

        return start, time_horizon, time_delta, speed, nodes, edges, intents_list


def create_dicts(nodes: List[Dict], edges: List[Dict], intents: List[Dict], time_horizon: int, time_delta: int,
                 speed: Union[int, float]) -> Tuple[Dict, Dict, Dict]:
    """
    Creates dictionaries of intents, nodes, edges given lists of these.
    Parameters
    ----------
    nodes: List[Dict]
        A list of nodes, each node being a dict.
    edges: List[Dict]
        A list of edges, each edge being a dict.
    intents: List[Dict]
        A list of intents, each intent being a dict.
    time_horizon: int
        Planning time horizon.
    time_delta: int
        Time delta.
    speed: Union[int, float]
        Common speed of all intents.

    Returns
    -------
    return value: Tuple[Dict, Dict, Dict]
        Three dictionaries (nodes_dict, edges_dict, intents_dict).

    """
    num_layers = time_horizon // time_delta

    nodes_dict = {v["name"]: graph.Node(v["name"], v["capacity"], num_layers) for v in nodes}
    edges_dict = {
        (e["source"], e["destination"]):
            graph.Edge(nodes_dict[e["source"]], nodes_dict[e["destination"]], math.ceil(e["weight"]/speed))
        for e in edges
    }

    # Add reverse edges
    edges_dict.update({
        (e["destination"], e["source"]): graph.Edge(nodes_dict[e["destination"]], nodes_dict[e["source"]], math.ceil(e["weight"] / speed))
        for e in edges
    })

    intents_dict = {
        (i["source"], i["destination"], i["start"]):
            intent.Intent(nodes_dict[i["source"]], nodes_dict[i["destination"]], i["start"], i["uncertainty"])
        for i in intents
    }

    return nodes_dict, edges_dict, intents_dict


def get_all_intents(path):
    """
    Given path to a json data file, this function creates and
    returns the list of all possible intents in the graph.
    """
    with open(path, 'r') as file:
        content = file.read()
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON example file at {path}:", e)
        exit()
    else:
        nodes, edges = data["nodes"], data["edges"]
        full_graph = nx.DiGraph([(edge["source"], edge["destination"]) for edge in edges])
        possible_intents = [(s, t) for s, t in itertools.product(nodes, nodes)
                            if s['name'] != t['name'] and nx.has_path(full_graph, s['name'], t['name'])]

        return possible_intents


def create_intent(all_intents: list, time_horizon: int, time_delta: int):
    """
    Selects an intent randomly from the list of all intents.
    Then generates random start time and time uncertainty.

    Parameters
    ----------
    all_intents: list
        A list of all possible intents
    time_horizon: int
        Planning time horizon.
    time_delta: int
        Time delta.


    Returns
    -------
        intent: dict
            The generated intent
    """
    random_index = np.random.randint(low=0, high=len(all_intents))
    source, destination = all_intents[random_index]
    source, destination = source['name'], destination['name']
    # U = np.random.randint(low=0, high=6) * time_delta
    # U = min(U, time_horizon // 10)
    U = 6*60
    start = min(np.random.randint(low=0, high=(time_horizon/time_delta)//2) * time_delta, time_horizon // 2)

    return {"source": source, "destination": destination, "start": start, "uncertainty": U}


def solve_greedy(intents_dict: Dict[str, intent.Intent], time_delta: int, nodes_dict: Dict[str, graph.Node]) \
        -> Tuple[Union[int, None], Union[graph.ExtendedNode, None]]:
    """
    Given data related to an operational intent, it runs the greedy algorithms on that intent,
    and adjusts the layer reservations by handling time uncertainty.

    Parameters
    ----------
    intents_dict: Dict[str, intent.Intent]
        Dictionary of intent objects.
    time_delta: int
        Time delta
    nodes_dict: Dict[str, graph.Node]
        Dictionary of node objects and their names as keys.

    Returns
    -------
    greedy_obj: Union[int, None]
        The objective. If no solution found, then None.

    """
    greedy_obj: Union[int, None] = 0

    for intent_name, operation_intent in intents_dict.items():
        # for each previous drone, get path, update vertiport capacities
        uncertainty_reservation_handling('increment', intent_name, operation_intent, nodes_dict, intents_dict,
                                         time_delta)

        # solve intent
        goal_node = optimization.find_shortest_path_extended(operation_intent, time_delta, nodes_dict)
        optimization.find_shortest_path(operation_intent, nodes_dict)
        time_difference = operation_intent.greedy_time_difference

        if goal_node and time_difference is not None:
            adjust_capacities(goal_node, nodes_dict)
            greedy_obj += time_difference
        else:
            # no solution found, so exit
            return None

        # for each previous drone, get path, update vertiport capacities
        uncertainty_reservation_handling('decrement', intent_name, operation_intent, nodes_dict, intents_dict,
                                         time_delta)

    return greedy_obj


def solve_ip(nodes: dict, edges: dict, intents: dict, time_steps: range, time_delta: int) -> Union[float, None]:
    """
    This function calls the ip optimization model to find a schedule for all the intents.

    Parameters
    ----------
    nodes: dict
        Dictionary of the nodes in a graph and their names.
    edges: dict
        Dictionary of the edges in a graph and their names.
    intents: dict
        Dictionary of the operational intents to be scheduled and their names.
    time_steps: range
        A range object for the discretized time steps.
    time_delta: int
        Time discretization step.

    Returns
    -------
    ip_obj: Union[float, None]
        The objective of the model or none if no solution was found.
    ip_model: mip.Model
        The optimization model.

    """
    ip_obj, ip_model = optimization.ip_optimization(nodes, edges, intents, time_steps, time_delta)
    return ip_obj, ip_model


def print_solutions(intents: dict) -> None:
    """
    Printing the solutions for each intent, in both greedy and ip parts.
    Also prints the operational time information.
    Parameters
    ----------
    intents: dict
        Dictionary of operational intents.

    Returns
    -------
        None
    """
    for name, operational_intent in intents.items():
        print(f"Intent {name}:")
        operational_intent.solution()


def adjust_capacities(goal_node: graph.ExtendedNode, nodes_dict: Dict[str, graph.Node]) -> None:
    """
    Given a dictionary of original graph nodes, it adjusts their capacities
    to include the capacities of the latest planned intent.

    Args:
        goal_node: graph.ExtendedNode
            The destination node in the extended graph.
        nodes_dict: Dict[str, graph.Node]
            Dictionary of the original graphs nodes.

    Returns:
        None
    """
    while goal_node:
        original_name = goal_node.name_original
        left = goal_node.left_reserve
        right = goal_node.right_reserve
        nodes_dict[original_name].decrement_capacity(left, right)
        goal_node = goal_node.previous


def increment_reservations(time_uncertainty: int, prev_intent, nodes_dict: dict,  delta: int,
                           indices: Sequence = None) -> None:
    """
    Increments reservations of the vertiports of a scheduled operational intent
    to mitigate the uncertainty of an operation after it in the queue.

    Args:
        time_uncertainty: int
            Time uncertainty of an operation down in the intents queue.
        prev_intent: intents.Intent
            A scheduled operational intent.
        nodes_dict: dict
            Dictionary of nodes objects and their names.
        delta: int
            Time discretization delta
        indices: Sequence (optinal)
            A sequence of indices used for looping over a path.

    Returns:

    """
    decrementor = int(math.copysign(1, time_uncertainty))  # sign of time_uncertainty
    path = prev_intent.path_greedy
    indices = indices if indices else range(1, len(path))

    for index in indices:
        name = path[index].name
        prev_name = path[index-1].name

        if name != prev_name:
            curr_left = path[index].left_reserved_layer
            left = math.ceil(decrementor*time_uncertainty/delta)
            new_left_layer = curr_left - left
            l, r = max(1, new_left_layer), curr_left
            # if previously set let layer, adjust it
            if prev_intent.path_greedy[index].probably_left_reserved_layer:
                prev_intent.path_greedy[index].probably_left_reserved_layer += left
            else:
                prev_intent.path_greedy[index].probably_left_reserved_layer = new_left_layer
            nodes_dict[name].layer_capacities[l:r] = [cap-decrementor for cap in nodes_dict[name].layer_capacities[l:r]]

    return None


def decrement_reservations(prev_intent: intent.Intent, curr_intent: intent.Intent, nodes_dict: dict, delta: int) \
        -> None:
    """
    Undoing `increment_reservations(...)` for the vertiports not being common of
    the two intents or that their uncertainty buffers don't intersect.

    Args:
        prev_intent: intent.Intent
            Previously scheduled intent
        curr_intent: intent.Intent
            An intent being scheduled currently.
        nodes_dict: dict
            Dictionary of nodes objects and their names.
        delta: int
            Time discretization delta

    Returns:
    """
    # find common vertiports, based on name as well as time intersection
    common_nodes = []
    u_prev, u_curr = [prev_intent.time_uncertainty, curr_intent.time_uncertainty]
    prev_intent_path: List[utils.Link] = prev_intent.path_greedy
    curr_intent_path: List[utils.Link] = curr_intent.path_greedy

    # find if the two intents have a common vertiport and their uncertainty buffers intersect at that vertiport
    for ind_p, prev_node in enumerate(prev_intent_path[1:]):
        for ind_c, curr_node in enumerate(curr_intent_path[1:]):
            if prev_node.name == curr_node.name:
                prev_intent_reaches_before = prev_node.right_reserved_layer <= curr_intent_path[ind_c].layer
                prev_intent_starts_after = prev_intent_path[ind_p].layer >= curr_node.right_reserved_layer
                if not (prev_intent_reaches_before or prev_intent_starts_after):
                    common_nodes.append(prev_node)
                break

    # for each vertiport in prev_intent_path, if it is not a common vertiport, decrement its cap
    indices = [i for i in range(1, len(prev_intent_path)) if prev_intent_path[i] not in common_nodes]
    increment_reservations(-u_curr, prev_intent, nodes_dict, delta, indices)

    return None


def uncertainty_reservation_handling(res_type: str, curr_intent_name: str, curr_intent: intent.Intent,
                                     nodes_dict: dict, intents_dict: dict, time_delta: int) -> None:
    """
    When planning an intent, previously scheduled intents must be safe from this current intent being delayed,
    hence the vertiports along all the paths of the previous intents are reserved for longer time.

    When the current intent is scheduled, then all the vertiports that were reserved for longer time but
    that are not included in the path of this current intent are being freed from that extra reservation.

    Depending on those two opposing cases, this function calls other functions to do the job.

    Args:
        res_type: str
            Either 'increment' or 'decrement', indicating whether a vertiport is reserved or freed at a layer.
        curr_intent_name: str
            Name of operational intent
        curr_intent: intent.Intent
            The operational intent
        nodes_dict: dict
            Dictionary of nodes objects and their names.
        intents_dict: dict
            Dictionary of intent objects and their names.
        time_delta: int
            Time discretization delta

    Returns:

    """
    curr_u = curr_intent.time_uncertainty

    for prev_intent_name, prev_intent in intents_dict.items():
        if prev_intent_name == curr_intent_name:
            break
        # p_path = prev_intent.path_greedy
        if res_type == 'increment':
            increment_reservations(curr_u, prev_intent, nodes_dict, time_delta)
        elif res_type == 'decrement':
            decrement_reservations(prev_intent, curr_intent, nodes_dict, time_delta)
    return None


def main(path: str, verbose: bool, intents_lst: List = None, analysis_obj: ResultsAnalysis = None) \
        -> Tuple[Union[int, None], Union[float, None]]:
    """
    The main function that solves each operational intent in sequence,
    as well as solving them all at once.

    Args:
        path: str
            Path to test file.
        verbose: bool
            Whether to print solutions.
        intents_lst: List (optional)
            This parameter can be used if intents are created on the fly,
            instead of being defined in a json example file.
        analysis_obj: DataAnalysis (optional)
            Object used to collect runs data.

    Returns:
        greedy_obj: int
            The greedy objective
        ip_obj: int
            The ip optimization objective

    """
    # --- Read data and create variables ---
    start, time_horizon, time_delta, speed, nodes, edges,  intents = read_example(path=path, intents=intents_lst)
    nodes_dict, edges_dict, intents_dict = create_dicts(nodes, edges, intents, max(GREEDY_TIME_HORIZON_MULTIPLIER, IP_TIME_HORIZON_MULTIPLIER)*time_horizon, time_delta, speed)
    time_steps = range(start, (IP_TIME_HORIZON_MULTIPLIER*time_horizon + 1), time_delta)

    # --- Solve IP ---
    ip_start = time.perf_counter()
    tracemalloc.start()
    ip_obj, ip_model = solve_ip(nodes_dict, edges_dict, intents_dict, time_steps, time_delta)
    ip_end = time.perf_counter()
    ip_runtime = (ip_end - ip_start)
    _, ip_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # --- Solve Greedy ---
    greedy_start = time.perf_counter()
    tracemalloc.start()
    greedy_obj = solve_greedy(intents_dict, time_delta, nodes_dict)
    greedy_end = time.perf_counter()
    greedy_runtime = (greedy_end - greedy_start)
    _, greedy_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # --- Check correctness of solutions ---
    print(f"\n\ngreedy={greedy_obj}, ip_obj={ip_obj}\n\n", flush=True)
    greedy_valid_solution, ip_valid_solution = (
        checks.sanity_check(intents_dict, nodes_dict, edges_dict, time_delta, max(GREEDY_TIME_HORIZON_MULTIPLIER, IP_TIME_HORIZON_MULTIPLIER)*time_horizon))
    print(f"greedy_valid_solution={greedy_valid_solution}\nip_valid_solution={ip_valid_solution}", flush=True)

    if verbose:
        print_solutions(intents_dict)

    ideal_times_sum = sum(op_intent.ideal_time for op_intent in intents_dict.values())
    if ip_obj:
        ip_obj -= ideal_times_sum
        ip_obj = round(ip_obj, 1)

    if analysis_obj:
        greedy_obj, ip_obj = greedy_obj if greedy_obj else np.nan, ip_obj if ip_obj else np.nan
        analysis_obj.add_row(num_intents=len(intents), greedy_obj=greedy_obj,
                             ip_obj=ip_obj, greedy_runtime=greedy_runtime, ip_runtime=ip_runtime,
                             greedy_memory_usage=greedy_memory, ip_memory_usage=ip_memory,
                             intents_collection=intent.IntentsCollection(list(intents_dict.values())),
                             ip_gap=round(ip_model.gap, 3), greedy_solution_valid=greedy_valid_solution,
                             ip_solution_valid=ip_valid_solution)

    print(f"Objectives: Greedy={greedy_obj} IP={ip_obj}\n", flush=True)

    return greedy_obj, ip_obj


if __name__ == "__main__":
    # --- Extract command line arguments ---
    args = parser.parse_args()
    seed = args.seed
    graph_name = args.graph_name
    run_name = args.run_name
    random_intents = args.random_intents
    num_intents = int(args.num_intents)

    np.random.seed(seed=seed)

    if args.debug:
        graph_path = graph_name
    else:
        graph_path = f"./graphs/{graph_name}.json"
    _, TIME_HORIZON, TIME_DELTA, _, _, _, _ = read_example(graph_path, None)

    if random_intents:
        print(f"\nExample with {num_intents} intents:\n{'=' * 100}", flush=True)

        results_dir_name = graph_name + "_" + run_name if len(run_name) > 0 else graph_name
        results_collector = ResultsAnalysis(results_dir_name)

        # --- Creating random intents ---
        all_possible_intents = get_all_intents(path=graph_path)
        intents_lst = [create_intent(all_possible_intents, TIME_HORIZON, TIME_DELTA) for _ in range(num_intents)]

        # --- Solve ---
        ip_objective, greedy_objective = main(graph_path, False, intents_lst, results_collector)

        # --- Storing results ---
        results_collector.save()

    else:
        greedy_objective, ip_objective = main(graph_name, True)

# =============================================== END OF FILE ===============================================

