"""
File description:
-----------------
Testing functionality of the two implemented versions of the Dijkstra's algorithm.
To run all tests, execute this command from project root:
    `python3 -m unittest discover -s tests`
To run a test locally, the path to the example data must be updated. Example:
    path = '../test_examples/test1.json'
"""

import unittest

import checks
import main
import optimization


def setUp(path):
    start, time_horizon, time_delta, speed, nodes, edges, intents, add_reverse_edges = main.read_example(path)
    nodes, edges, intents = main.create_dicts(nodes, edges, intents, time_horizon, time_delta, speed, add_reverse_edges)

    return start, time_horizon, time_delta, nodes, edges, intents


def run_test(self):
    index = 0
    for intent_name, operation_intent in self.intents.items():
        main.uncertainty_reservation_handling('increment', intent_name, operation_intent, self.nodes,
                                              self.intents, self.time_delta)

        goal_node = optimization.find_shortest_path_extended(operation_intent, self.time_delta, self.nodes)
        optimization.find_shortest_path(operation_intent, self.nodes)

        # assert goal node is reached
        self.assertEqual(goal_node.name_original, operation_intent.destination.name)

        # assert goal node is reached at correct layer
        self.assertEqual(goal_node.layer, self.destination_layers[index])

        # assert actual time is correct
        self.assertEqual(operation_intent.actual_greedy_time, self.actual_times[index])

        # assert ideal time is correct
        self.assertEqual(operation_intent.ideal_time, self.ideal_times[index])

        main.uncertainty_reservation_handling('decrement', intent_name, operation_intent, self.nodes,
                                              self.intents, self.time_delta)

        main.adjust_capacities(goal_node, self.nodes)

        index += 1

    # assert solution is valid
    greedy_valid_solution, _ = (
        checks.sanity_check(self.intents, self.nodes, self.edges, self.time_delta, self.time_horizon))
    self.assertTrue(greedy_valid_solution.time_correct)
    self.assertTrue(greedy_valid_solution.capacity_correct)
    self.assertTrue(not greedy_valid_solution.cycles_exists)


class Example1(unittest.TestCase):
    """A small graph of two edges and one intent."""
    def setUp(self) -> None:
        self.start, self.time_horizon, self.time_delta, self.nodes, self.edges, self.intents = (
            setUp('tests/test_examples/test1.json'))

        self.ideal_times = [11]
        self.actual_times = [20]
        self.destination_layers = [2]

    def test_dijkstra(self):
        run_test(self)


class Example2(unittest.TestCase):
    """A small graph of two edges and one intent, where layer jump is needed."""
    def setUp(self) -> None:
        self.start, self.time_horizon, self.time_delta, self.nodes, self.edges, self.intents = (
            setUp('tests/test_examples/test2.json'))

        self.ideal_times = [10]
        self.actual_times = [15]
        self.destination_layers = [3]

    def test_dijkstra(self):
        run_test(self)


class Example4(unittest.TestCase):
    """A small graph where intent travel time is beyond time horizon, hence no solution."""
    def setUp(self) -> None:
        self.start, self.time_horizon, self.time_delta, self.nodes, self.edges, self.intents = (
            setUp('tests/test_examples/test4.json'))

    def test_dijkstra(self):
        index = 0

        intent_name, operation_intent = list(self.intents.items())[index]
        goal_node = optimization.find_shortest_path_extended(operation_intent, self.time_delta, self.nodes)
        optimization.find_shortest_path(operation_intent, self.nodes)

        # assert goal node is reached
        self.assertIs(goal_node, None)
        self.assertEqual(operation_intent.actual_greedy_time, 0)
        self.assertEqual(operation_intent.ideal_time, 36)


class Example5(unittest.TestCase):
    """A graph with two identical paths between `src` and `dest`."""
    def setUp(self) -> None:
        self.start, self.time_horizon, self.time_delta, self.nodes, self.edges, self.intents = (
            setUp('tests/test_examples/test5.json'))

        self.ideal_times = [9]
        self.actual_times = [10]
        self.destination_layers = [2]

    def test_dijkstra(self):
        run_test(self)


class Example6(unittest.TestCase):
    """start time for the intent is at first layer instead of time 0."""
    def setUp(self) -> None:
        self.start, self.time_horizon, self.time_delta, self.nodes, self.edges, self.intents = (
            setUp('tests/test_examples/test6.json'))

        self.ideal_times = [8]
        self.actual_times = [10]
        self.destination_layers = [3]

    def test_dijkstra(self):
        run_test(self)


class Example7(unittest.TestCase):
    """Test contains two intents starting at different nodes."""
    def setUp(self) -> None:
        self.start, self.time_horizon, self.time_delta, self.nodes, self.edges, self.intents = (
            setUp('tests/test_examples/test7.json'))

        self.ideal_times = [7, 7]
        self.actual_times = [10, 10]
        self.destination_layers = [2, 2]

    def test_dijkstra(self):
        run_test(self)


class Example8(unittest.TestCase):
    """Test contains two intents where the second one has to wait at `src` due to limited capacity at `dest`."""
    def setUp(self) -> None:
        self.start, self.time_horizon, self.time_delta, self.nodes, self.edges, self.intents = (
            setUp('tests/test_examples/test8.json'))

        self.ideal_times = [7, 7]
        self.actual_times = [10, 15]
        self.destination_layers = [2, 3]

    def test_dijkstra(self):
        run_test(self)


class Example9(unittest.TestCase):
    """Test contains two intents where the second one has to follow a longer path due to limited capacity at `dest`."""
    def setUp(self) -> None:
        self.start, self.time_horizon, self.time_delta, self.nodes, self.edges, self.intents = (
            setUp('tests/test_examples/test9.json'))

        self.ideal_times = [7, 7]
        self.actual_times = [10, 15]
        self.destination_layers = [2, 3]

    def test_dijkstra(self):
        run_test(self)


class Example10(unittest.TestCase):
    """A large graph with several intents."""
    def setUp(self) -> None:
        self.start, self.time_horizon, self.time_delta, self.nodes, self.edges, self.intents = (
            setUp('tests/test_examples/test10.json'))

        self.ideal_times = [12, 3, 10]
        self.actual_times = [20, 5, 20]
        self.destination_layers = [4, 2, 4]

    def test_dijkstra(self):
        run_test(self)


class Example11(unittest.TestCase):
    """Test contains two intents with time uncertainty where the second intent cannot be operated."""
    def setUp(self) -> None:
        self.start, self.time_horizon, self.time_delta, self.nodes, self.edges, self.intents = (
            setUp('tests/test_examples/test11.json'))

        self.ideal_times = [12, 0]
        self.actual_times = [14, 0]
        self.destination_layers = [7, None]

    def test_dijkstra(self):
        _, operation_intent = list(self.intents.items())[-1]
        self.intents = {key: val for key, val in list(self.intents.items())[:-1]}
        run_test(self)

        intent_name, operation_intent = list(self.intents.items())[-1]
        goal_node = optimization.find_shortest_path_extended(operation_intent, self.time_delta, self.nodes)
        self.assertIs(goal_node, None)


class Example12(unittest.TestCase):
    """Test contains two intents with time uncertainty where the second intent can be operated as capacity exists."""
    def setUp(self) -> None:
        self.start, self.time_horizon, self.time_delta, self.nodes, self.edges, self.intents = (
            setUp('tests/test_examples/test12.json'))

        self.ideal_times = [12, 10]
        self.actual_times = [14, 12]
        self.destination_layers = [7, 6]

    def test_dijkstra(self):
        run_test(self)


class Example13(unittest.TestCase):
    """Test contains two intents where no vertiports are shared."""
    def setUp(self) -> None:
        self.start, self.time_horizon, self.time_delta, self.nodes, self.edges, self.intents = (
            setUp('tests/test_examples/test13.json'))

        self.ideal_times = [5, 5]
        self.actual_times = [6, 6]
        self.destination_layers = [3, 3]

    def test_dijkstra(self):
        run_test(self)


class Example14(unittest.TestCase):
    """Same as test 11 but where the second intent starts later hence it can be operated."""
    def setUp(self) -> None:
        self.start, self.time_horizon, self.time_delta, self.nodes, self.edges, self.intents = (
            setUp('tests/test_examples/test14.json'))

        self.ideal_times = [12, 10]
        self.actual_times = [14, 14]
        self.destination_layers = [7, 10]

    def test_dijkstra(self):
        run_test(self)


if __name__ == "__main__":
    unittest.main()

# =============================================== END OF FILE ===============================================
