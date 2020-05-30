import argparse
import collections
import copy
import datetime
import math
import numpy as np
import os
import pickle
import random
import sys
import tensorflow as tf
from tensorflow.contrib import rnn
from sequence_encoder import encode_seq, embed_seq


EPSILON = 1e-6

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_config(args=None):
    parser = argparse.ArgumentParser(description="Meta optimization")
    parser.add_argument('--epoch_size', type=int, default=5120000, help='Epoch size')

    parser.add_argument('--num_lstm_units', type=int, default=128, help="number of LSTM units")
    parser.add_argument('--num_feedforward_units', type=int, default=128, help="number of feedforward units")
    parser.add_argument('--problem', default='vrp', help="the problem to be solved, {tsp, vrp}")
    parser.add_argument('--train_operators', type=str2bool, nargs='?', const=True, default=False, help="")
    parser.add_argument('--depot_positioning', default='R', help="{R, C, E}")
    parser.add_argument('--customer_positioning', default='R', help="{R, C, RC}")

    parser.add_argument('--num_training_points', type=int, default=100, help="size of the problem for training")
    parser.add_argument('--num_test_points', type=int, default=100, help="size of the problem for testing")
    parser.add_argument('--num_episode', type=int, default=40000, help="number of training episode")
    parser.add_argument('--max_num_rows', type=int, default=2000000, help="")
    parser.add_argument('--num_paths_to_ruin', type=int, default=2, help="")
    parser.add_argument('--batch_size', type=int, default=1000, help='batch size')
    parser.add_argument('--max_rollout_steps', type=int, default=20000, help="maximum rollout steps")
    parser.add_argument('--max_rollout_seconds', type=int, default=1000, help="maximum rollout time in seconds")
    parser.add_argument('--use_cyclic_rollout', type=str2bool, nargs='?', const=True, default=False, help="use cyclic rollout")
    parser.add_argument('--use_random_rollout',type=str2bool, nargs='?', const=True, default=False, help="use random rollout")
    parser.add_argument('--detect_negative_cycle', type=str2bool, nargs='?', const=True, default=False, help="")
    parser.add_argument('--use_rl_loss', type=str2bool, nargs='?', const=True, default=True, help="")
    parser.add_argument('--use_attention_embedding', type=str2bool, nargs='?', const=True, default=True, help="")
    parser.add_argument('--epsilon_greedy', type=float, default=0.05, help="")
    parser.add_argument('--sample_actions_in_rollout', type=str2bool, nargs='?', const=True, default=True, help="")
    parser.add_argument('--num_active_learning_iterations', type=int, default=1, help="")
    parser.add_argument('--max_no_improvement', type=int, default=6, help="")
    parser.add_argument('--debug_mode', type=str2bool, nargs='?', const=True, default=False, help="")
    parser.add_argument('--debug_steps', type=int, default=1, help="")
    parser.add_argument('--num_actions', type=int, default=27, help="dimension of action space")
    parser.add_argument('--max_num_customers_to_shuffle', type=int, default=20, help="")
    parser.add_argument('--problem_seed', type=int, default=1, help="problem generating seed")
    parser.add_argument('--input_embedded_trip_dim', type=int, default=9, help="")
    parser.add_argument('--input_embedded_trip_dim_2', type=int, default=11, help="")
    parser.add_argument('--num_embedded_dim_1', type=int, default=64, help="")
    parser.add_argument('--num_embedded_dim_2', type=int, default=64, help="dim")
    parser.add_argument('--discount_factor', type=float, default=1.0, help="discount factor of policy network")
    parser.add_argument('--policy_learning_rate', type=float, default=0.001, help="learning rate of policy network")
    parser.add_argument('--hidden_layer_dim', type=int, default=64, help="dimension of hidden layer in policy network")
    parser.add_argument('--num_history_action_use', type=int, default=0, help="number of history actions used in the representation of current state")
    parser.add_argument('--use_history_action_distribution', type=str2bool, nargs='?', const=True, default=False, help="")
    parser.add_argument('--step_interval', type=int, default=500)

    # './rollout_model_1850.ckpt'
    parser.add_argument('--model_to_restore', type=str, default=None, help="")
    parser.add_argument('--max_num_training_epsisodes', type=int, default=10000000, help="")

    parser.add_argument('--max_points_per_trip', type=int, default=15, help="upper bound of number of point in one trip")
    parser.add_argument('--max_trips_per_solution', type=int, default=15, help="upper bound of number of trip in one solution")

    config = parser.parse_args(args)
    return config


config = get_config()
if config.max_no_improvement is None:
    config.max_no_improvement = config.num_actions


def calculate_distance(point0, point1):
    dx = point1[0] - point0[0]
    dy = point1[1] - point0[1]
    return math.sqrt(dx * dx + dy * dy)


class Problem:
    def __init__(self, locations, capacities):
        self.locations = copy.deepcopy(locations)
        self.capacities = copy.deepcopy(capacities)
        self.distance_matrix = []
        for from_index in range(len(self.locations)):
            distance_vector = []
            for to_index in range(len(self.locations)):
                distance_vector.append(calculate_distance(locations[from_index], locations[to_index]))
            self.distance_matrix.append(distance_vector)
        self.total_customer_capacities = 0
        for capacity in capacities[1:]:
            self.total_customer_capacities += capacity
        self.change_at = [0] * (len(self.locations) + 1)
        self.no_improvement_at = {}
        self.num_solutions = 0
        self.num_traversed = np.zeros((len(locations), len(locations)))
        self.distance_hashes = set()

    def record_solution(self, solution, distance):
        self.num_solutions += 1.0 / distance
        for path in solution:
            if len(path) > 2:
                for to_index in range(1, len(path)):
                    #TODO: change is needed for asymmetric cases.
                    self.num_traversed[path[to_index - 1]][path[to_index]] += 1.0 / distance
                    self.num_traversed[path[to_index]][path[to_index - 1]] += 1.0 / distance
                    # for index_in_the_same_path in range(to_index + 1, len(path)):
                    #     self.num_traversed[path[index_in_the_same_path]][path[to_index]] += 1
                    #     self.num_traversed[path[to_index]][path[index_in_the_same_path]] += 1

    def add_distance_hash(self, distance_hash):
        self.distance_hashes.add(distance_hash)

    def get_location(self, index):
        return self.locations[index]

    def get_capacity(self, index):
        return self.capacities[index]

    def get_capacity_ratio(self):
        return self.total_customer_capacities / float(self.get_capacity(0))

    def get_num_customers(self):
        return len(self.locations) - 1

    def get_distance(self, from_index, to_index):
        return self.distance_matrix[from_index][to_index]

    def get_frequency(self, from_index, to_index):
        return self.num_traversed[from_index][to_index] / (1.0 + self.num_solutions)

    def reset_change_at_and_no_improvement_at(self):
        self.change_at = [0] * (len(self.locations) + 1)
        self.no_improvement_at = {}

    def mark_change_at(self, step, path_indices):
        for path_index in path_indices:
            self.change_at[path_index] = step

    def mark_no_improvement(self, step, action, index_first, index_second=-1, index_third=-1):
        key = '{}_{}_{}_{}'.format(action, index_first, index_second, index_third)
        self.no_improvement_at[key] = step

    def should_try(self, action, index_first, index_second=-1, index_third=-1):
        key = '{}_{}_{}_{}'.format(action, index_first, index_second, index_third)
        no_improvement_at = self.no_improvement_at.get(key, -1)
        return self.change_at[index_first] >= no_improvement_at or \
               self.change_at[index_second] >= no_improvement_at or \
               self.change_at[index_third] >= no_improvement_at


def calculate_distance_between_indices(problem, from_index, to_index):
    return problem.get_distance(from_index, to_index)


def calculate_adjusted_distance_between_indices(problem, from_index, to_index):
    distance = problem.get_distance(from_index, to_index)
    frequency = problem.get_frequency(from_index, to_index)
    # return (1.0 - frequency)
    return distance * (1.0 - frequency)
    # return distance * frequency


def calculate_trip_distance(trip):
    sum = 0.0
    for i in range(len(trip)):
        sum += calculate_distance(trip[i - 1], trip[i])
    return sum


def calculate_path_distance(problem, path):
    sum = 0.0
    for i in range(1, len(path)):
        sum += calculate_distance_between_indices(problem, path[i - 1], path[i])
    return sum


def calculate_solution_distance(problem, solution):
    total_distance = 0.0
    for path in solution:
        total_distance += calculate_path_distance(problem, path)
    return total_distance


def validate_solution(problem, solution, distance=None):
    if config.problem == 'tsp':
        if len(solution) != 1:
            return False
    visited = [0] * (problem.get_num_customers() + 1)
    for path in solution:
        if path[0] != 0 or path[-1] != 0:
            return False
        consumption = calculate_consumption(problem, path)
        if consumption[-2] > problem.get_capacity(path[0]):
            return False
        for customer in path[1:-1]:
            visited[customer] += 1
    for customer in range(1, len(visited)):
        if visited[customer] != 1:
            return False
    if config.problem == 'tsp':
        if visited[0] != 0:
            return False
    if distance is not None and math.fabs(distance - calculate_solution_distance(problem, solution)) > EPSILON:
        return False
    return True


def two_opt(trip, first, second):
    new_trip = copy.deepcopy(trip)
    if first > second:
        first, second = second, first
    first = first + 1
    while first < second:
        temp = copy.copy(new_trip[first])
        new_trip[first] = copy.copy(new_trip[second])
        new_trip[second] = temp
        first = first + 1
        second = second - 1
    return new_trip


def apply_two_opt(trip, distance, top_indices_eval, offset=0):
    #TODO(xingwen): this implementation is very inefficient.
    n = len(trip)
    top_indices = top_indices_eval[0]
    num_indices = len(top_indices)
    min_distance = float('inf')
    min_trip = None
    for i in range(num_indices - 1):
        first = top_indices[i]
        for j in range(i + 1, num_indices):
            second = top_indices[j]
            new_trip = two_opt(trip, (first + offset) % n, (second + offset) % n)
            new_distance = calculate_trip_distance(new_trip)
            if new_distance < min_distance:
                min_distance = new_distance
                min_trip = new_trip
            # print('distance={}, new_distance={}'.format(distance, new_distance))
    if min_distance < distance:
        return min_trip, min_distance
    else:
        return trip, distance


def two_exchange(trip):
    n = len(trip)
    min_delta = -1e-6
    min_first, min_second = None, None
    for first in range(n - 1):
        for second in range(first + 2, min(first + 11, n)):
            if first == 0 and second == n - 1:
                continue
            before = calculate_distance(trip[first - 1], trip[first]) \
                    + calculate_distance(trip[first], trip[first + 1]) \
                    + calculate_distance(trip[second - 1], trip[second]) \
                    + calculate_distance(trip[second], trip[(second + 1) % n])
            after = calculate_distance(trip[first - 1], trip[second]) \
                    + calculate_distance(trip[second], trip[first + 1]) \
                    + calculate_distance(trip[second - 1], trip[first]) \
                    + calculate_distance(trip[first], trip[(second + 1) % n])
            delta = after - before
            if delta < min_delta:
                min_delta = delta
                min_first = first
                min_second = second
    if min_first is None:
        return trip, calculate_trip_distance(trip)
    else:
        new_trip = copy.deepcopy(trip)
        temp = copy.copy(new_trip[min_first])
        new_trip[min_first] = copy.copy(new_trip[min_second])
        new_trip[min_second] = temp
        return new_trip, calculate_trip_distance(new_trip)


def relocate(trip):
    n = len(trip)
    min_delta = -1e-6
    min_first, min_second = None, None
    for first in range(n):
        # for second in range(n):
        for away in range(-10, 10, 1):
            second = (first + away + n) % n
            if second == (first - 1 + n) % n or second == first:
                continue
            before = calculate_distance(trip[first - 1], trip[first]) \
                    + calculate_distance(trip[first], trip[(first + 1) % n]) \
                    + calculate_distance(trip[second], trip[(second + 1) % n])
            after = calculate_distance(trip[first - 1], trip[(first + 1) % n]) \
                    + calculate_distance(trip[second], trip[first]) \
                    + calculate_distance(trip[first], trip[(second + 1) % n])
            delta = after - before
            if delta < min_delta:
                min_delta = delta
                min_first = first
                min_second = second
    if min_first is None:
        return trip, calculate_trip_distance(trip)
    else:
        new_trip = copy.deepcopy(trip)
        temp = copy.copy(new_trip[min_first])
        to_index = min_first
        while to_index != min_second:
            next_index = (to_index + 1) % n
            new_trip[to_index] = copy.copy(new_trip[next_index])
            to_index = next_index
        new_trip[min_second] = temp
        return new_trip, calculate_trip_distance(new_trip)


def mutate(trip):
    n = len(trip)
    min = -1e-6
    label = None
    for first in range(n - 1):
        for second in range(first + 2, n):
            before = calculate_distance(trip[first], trip[first + 1]) \
                     + calculate_distance(trip[second], trip[(second + 1) % n])
            after = calculate_distance(trip[first], trip[second]) \
                    + calculate_distance(trip[first + 1], trip[(second + 1) % n])
            delta = after - before
            if delta < min:
                min = delta
                label = first, second
    if label is None:
        return None, None, None
    else:
        return two_opt(trip, label[0], label[1]), min, label


def do_two_opt_path(path, first, second):
    improved_path = copy.deepcopy(path)
    first = first + 1
    while first < second:
        improved_path[first], improved_path[second] = improved_path[second], improved_path[first]
        first = first + 1
        second = second - 1
    return improved_path


def two_opt_path(problem, path):
    n = len(path) - 1
    min_delta = -EPSILON
    label = None
    for first in range(n - 1):
        for second in range(first + 2, n):
            before = calculate_distance_between_indices(problem, path[first], path[first + 1]) \
                     + calculate_distance_between_indices(problem, path[second], path[second + 1])
            after = calculate_distance_between_indices(problem, path[first], path[second]) \
                    + calculate_distance_between_indices(problem, path[first + 1], path[second + 1])
            delta = after - before
            if delta < min_delta:
                min_delta = delta
                label = first, second
    if label is None:
        return None, None, None
    else:
        return do_two_opt_path(path, label[0], label[1]), min_delta, label


def do_exchange_path(path, first, second):
    improved_path = copy.deepcopy(path)
    improved_path[first], improved_path[second] = improved_path[second], improved_path[first]
    return improved_path


def exchange_path(problem, path):
    n = len(path) - 1
    min_delta = -EPSILON
    label = None
    for first in range(1, n - 1):
        for second in range(first + 1, n):
            if second == first + 1:
                before = calculate_distance_between_indices(problem, path[first - 1], path[first]) \
                     + calculate_distance_between_indices(problem, path[second], path[second + 1])
                after = calculate_distance_between_indices(problem, path[first - 1], path[second]) \
                     + calculate_distance_between_indices(problem, path[first], path[second + 1])
            else:
                before = calculate_distance_between_indices(problem, path[first - 1], path[first]) \
                     + calculate_distance_between_indices(problem, path[first], path[first + 1]) \
                     + calculate_distance_between_indices(problem, path[second - 1], path[second]) \
                     + calculate_distance_between_indices(problem, path[second], path[second + 1])
                after = calculate_distance_between_indices(problem, path[first - 1], path[second]) \
                     + calculate_distance_between_indices(problem, path[second], path[first + 1]) \
                     + calculate_distance_between_indices(problem, path[second - 1], path[first]) \
                     + calculate_distance_between_indices(problem, path[first], path[second + 1])
            delta = after - before
            if delta < min_delta:
                min_delta = delta
                label = first, second
    if label is None:
        return None, None, None
    else:
        return do_exchange_path(path, label[0], label[1]), min_delta, label


def do_relocate_path(path, first, first_tail, second):
    segment = path[first:(first_tail + 1)]
    improved_path = path[:first] + path[(first_tail + 1):]
    if second > first_tail:
        second -= (first_tail - first + 1)
    return improved_path[:(second + 1)] + segment + improved_path[(second + 1):]


def relocate_path(problem, path, exact_length=1):
    n = len(path) - 1
    min_delta = -EPSILON
    label = None
    for first in range(1, n - exact_length + 1):
        first_tail = first + exact_length - 1
        for second in range(n):
            if second >= first - 1 and second <= first_tail:
                continue
            before = calculate_distance_between_indices(problem, path[first - 1], path[first]) \
                    + calculate_distance_between_indices(problem, path[first_tail], path[first_tail + 1]) \
                    + calculate_distance_between_indices(problem, path[second], path[second + 1])
            after = calculate_distance_between_indices(problem, path[first - 1], path[first_tail + 1]) \
                    + calculate_distance_between_indices(problem, path[second], path[first]) \
                    + calculate_distance_between_indices(problem, path[first_tail], path[second + 1])
            delta = after - before
            if delta < min_delta:
                min_delta = delta
                label = first, first_tail, second
    if label is None:
        return None, None, None
    else:
        return do_relocate_path(path, label[0], label[1], label[2]), min_delta, label


def calculate_consumption(problem, path):
    n = len(path)
    consumption = [0] * n
    consumption[0] = 0
    for i in range(1, n - 1):
        consumption[i] = consumption[i - 1] + problem.get_capacity(path[i])
    consumption[n - 1] = consumption[n - 2]
    return consumption


def do_cross_two_paths(path_first, path_second, first, second):
    return path_first[:(first + 1)] + path_second[(second + 1):], path_second[:(second + 1)] + path_first[(first + 1):]


def cross_two_paths(problem, path_first, path_second):
    n_first = len(path_first) - 1
    n_second = len(path_second) - 1
    min_delta = -EPSILON
    label = None
    consumed_capacities_first = calculate_consumption(problem, path_first)
    consumed_capacities_second = calculate_consumption(problem, path_second)

    start_of_second_index = 0
    for first in range(n_first):
        capacity_from_first_to_second = consumed_capacities_first[n_first - 1] - consumed_capacities_first[first]
        for second in range(start_of_second_index, n_second):
            if consumed_capacities_second[second] + capacity_from_first_to_second > problem.get_capacity(path_second[0]):
                break
            if consumed_capacities_first[first] + (consumed_capacities_second[n_second - 1] - consumed_capacities_second[second]) > problem.get_capacity(path_first[0]):
                start_of_second_index = second + 1
                continue
            before = calculate_distance_between_indices(problem, path_first[first], path_first[first + 1]) \
                     + calculate_distance_between_indices(problem, path_second[second], path_second[second + 1])
            after = calculate_distance_between_indices(problem, path_first[first], path_second[second + 1]) \
                    + calculate_distance_between_indices(problem, path_second[second], path_first[first + 1])
            delta = after - before
            if delta < min_delta:
                min_delta = delta
                label = first, second
    if label is None:
        return None, None, None, None
    else:
        improved_path_first, improved_path_second = do_cross_two_paths(path_first, path_second, label[0], label[1])
        return improved_path_first, improved_path_second, min_delta, label


def do_relocate_two_paths(path_first, path_second, first, first_tail, second):
    return path_first[:first] + path_first[(first_tail + 1):], \
           path_second[:(second + 1)] + path_first[first:(first_tail + 1)] + path_second[(second + 1):]


def relocate_two_paths(problem, path_first, path_second, exact_length=None):
    n_first = len(path_first) - 1
    n_second = len(path_second) - 1
    min_delta = -EPSILON
    label = None
    consumed_capacities_first = calculate_consumption(problem, path_first)
    consumed_capacities_second = calculate_consumption(problem, path_second)

    max_length = 1
    min_length = 1
    if exact_length:
        max_length = exact_length
        min_length = exact_length
    for first in range(1, n_first):
        for first_tail in range((first + min_length - 1), min(first + max_length, n_first)):
            capacity_difference = (consumed_capacities_first[first_tail] - consumed_capacities_first[first - 1])
            if consumed_capacities_second[n_second - 1] + capacity_difference > problem.get_capacity(path_second[0]):
                break
            for second in range(0, n_second):
                before = calculate_distance_between_indices(problem, path_first[first - 1], path_first[first]) \
                     + calculate_distance_between_indices(problem, path_first[first_tail], path_first[first_tail + 1])\
                     + calculate_distance_between_indices(problem, path_second[second], path_second[second + 1])
                after = calculate_distance_between_indices(problem, path_first[first - 1], path_first[first_tail + 1])\
                     + calculate_distance_between_indices(problem, path_second[second], path_first[first])\
                     + calculate_distance_between_indices(problem, path_first[first_tail], path_second[second + 1])
                delta = after - before
                if delta < min_delta:
                    min_delta = delta
                    label = first, first_tail, second
    if label is None:
        return None, None, None, None
    else:
        improved_path_first, improved_path_second = do_relocate_two_paths(path_first, path_second, label[0], label[1], label[2])
        return improved_path_first, improved_path_second, min_delta, label


def do_exchange_two_paths(path_first, path_second, first, first_tail, second, second_tail):
    return path_first[:first] + path_second[second:(second_tail + 1)] + path_first[(first_tail + 1):], \
           path_second[:second] + path_first[first:(first_tail + 1)] + path_second[(second_tail + 1):]


def exchange_two_paths(problem, path_first, path_second, exact_lengths=None):
    n_first = len(path_first) - 1
    n_second = len(path_second) - 1
    consumed_capacities_first = calculate_consumption(problem, path_first)
    consumed_capacities_second = calculate_consumption(problem, path_second)
    if exact_lengths:
        min_length_first, max_length_first = exact_lengths[0], exact_lengths[0]
        min_length_second, max_length_second = exact_lengths[1], exact_lengths[1]
    else:
        min_length_first, max_length_first = 1, 1
        min_length_second, max_length_second = 1, 1

    min_delta = -EPSILON
    label = None
    all_delta = 0.0
    for first in range(1, n_first):
        for first_tail in range((first + min_length_first - 1), min(first + max_length_first, n_first)):
            if first_tail >= n_first:
                break
            for second in range(1, n_second):
                if first_tail >= n_first:
                    break
                for second_tail in range((second + min_length_second - 1), min(second + max_length_second, n_second)):
                    if first_tail >= n_first:
                        break
                    if second_tail >= n_second:
                        break
                    capacity_difference = (consumed_capacities_first[first_tail] - consumed_capacities_first[first - 1]) - \
                                          (consumed_capacities_second[second_tail] - consumed_capacities_second[second - 1])
                    if consumed_capacities_first[n_first - 1] - capacity_difference <= problem.get_capacity(path_first[0]) and \
                            consumed_capacities_second[n_second - 1] + capacity_difference <= problem.get_capacity(path_second[0]):
                        pass
                    else:
                        continue
                    before = calculate_distance_between_indices(problem, path_first[first - 1], path_first[first]) \
                     + calculate_distance_between_indices(problem, path_first[first_tail], path_first[first_tail + 1])\
                     + calculate_distance_between_indices(problem, path_second[second - 1], path_second[second])\
                     + calculate_distance_between_indices(problem, path_second[second_tail], path_second[second_tail + 1])
                    after = calculate_distance_between_indices(problem, path_first[first - 1], path_second[second]) \
                     + calculate_distance_between_indices(problem, path_second[second_tail], path_first[first_tail + 1])\
                     + calculate_distance_between_indices(problem, path_second[second - 1], path_first[first])\
                     + calculate_distance_between_indices(problem, path_first[first_tail], path_second[second_tail + 1])
                    delta = after - before
                    if delta < -EPSILON:
                        all_delta += delta
                        label = first, first_tail, second, second_tail
                        path_first, path_second = do_exchange_two_paths(path_first, path_second, label[0], label[1], label[2], label[3])
                        #TODO(xingwen): speedup
                        n_first = len(path_first) - 1
                        n_second = len(path_second) - 1
                        consumed_capacities_first = calculate_consumption(problem, path_first)
                        consumed_capacities_second = calculate_consumption(problem, path_second)
    if label is None:
        return None, None, None, None
    else:
        return path_first, path_second, all_delta, label


def do_eject_two_paths(path_first, path_second, first, second):
    return path_first[:first] + path_first[(first + 1):], \
           path_second[:second] + path_first[first:(first + 1)] + path_second[(second + 1):]


def eject_two_paths(problem, path_first, path_second):
    n_first = len(path_first) - 1
    n_second = len(path_second) - 1
    min_delta = float("inf")
    label = None
    consumed_capacities_second = calculate_consumption(problem, path_second)

    for first in range(1, n_first):
        for second in range(1, n_second):
            capacity_difference = problem.get_capacity(path_first[first]) - problem.get_capacity(path_second[second])
            if consumed_capacities_second[n_second - 1] + capacity_difference > problem.get_capacity(path_second[0]):
                continue
            before = calculate_distance_between_indices(problem, path_first[first - 1], path_first[first]) \
                     + calculate_distance_between_indices(problem, path_first[first], path_first[first + 1]) \
                     + calculate_distance_between_indices(problem, path_second[second - 1], path_second[second]) \
                     + calculate_distance_between_indices(problem, path_second[second], path_second[second + 1])
            after = calculate_distance_between_indices(problem, path_first[first - 1], path_first[first + 1]) \
                    + calculate_distance_between_indices(problem, path_second[second - 1], path_first[first]) \
                    + calculate_distance_between_indices(problem, path_first[first], path_second[second + 1])
            delta = after - before
            if delta < min_delta:
                min_delta = delta
                label = first, second, path_second[second]
    if label is None:
        return None, None, None, None
    else:
        improved_path_first, improved_path_second = do_eject_two_paths(path_first, path_second, label[0], label[1])
        return improved_path_first, improved_path_second, min_delta, label[2]


def insert_into_path(path, first):
    n = len(path) - 1
    min_delta = float("inf")
    label = None
    consumed_capacities = calculate_consumption(problem, path)

    if consumed_capacities[n - 1] + problem.get_capacity(first) > problem.get_capacity(path[0]):
        return None, None, None
    for second in range(0, n):
        before = calculate_distance_between_indices(problem, path[second], path[second + 1])
        after = calculate_distance_between_indices(problem, path[second], first) \
                + calculate_distance_between_indices(problem, first, path[second + 1])
        delta = after - before
        if delta < min_delta:
            min_delta = delta
            label = second

    improved_path_third = path[:(label + 1)] + [first] + path[(label + 1):]
    return improved_path_third, min_delta, label


def do_eject_three_paths(path_first, path_second, path_third, first, second, third):
    return path_first[:first] + [path_third[third]] + path_first[(first + 1):], \
           path_second[:second] + [path_first[first]] + path_second[(second + 1):], \
           path_third[:third] + [path_second[second]] + path_third[(third + 1):]


def eject_three_paths(problem, path_first, path_second, path_third):
    n_first = len(path_first) - 1
    n_second = len(path_second) - 1
    n_third = len(path_third) - 1
    min_delta = -EPSILON
    label = None
    consumed_capacities_first = calculate_consumption(problem, path_first)
    consumed_capacities_second = calculate_consumption(problem, path_second)
    consumed_capacities_third = calculate_consumption(problem, path_third)

    for first in range(1, n_first):
        for second in range(1, n_second):
            if consumed_capacities_second[n_second - 1] + problem.get_capacity(path_first[first]) - problem.get_capacity(path_second[second]) > problem.get_capacity(path_second[0]):
                continue
            for third in range(1, n_third):
                if consumed_capacities_third[n_third - 1] + problem.get_capacity(path_second[second]) - problem.get_capacity(path_third[third]) > problem.get_capacity(path_third[0]):
                    continue
                if consumed_capacities_first[n_first - 1] + problem.get_capacity(path_third[third]) - problem.get_capacity(path_first[first]) > problem.get_capacity(path_first[0]):
                    continue
                before = calculate_distance_between_indices(problem, path_first[first - 1], path_first[first]) \
                    + calculate_distance_between_indices(problem, path_first[first], path_first[first + 1]) \
                    + calculate_distance_between_indices(problem, path_second[second - 1], path_second[second]) \
                    + calculate_distance_between_indices(problem, path_second[second], path_second[second + 1]) \
                    + calculate_distance_between_indices(problem, path_third[third - 1], path_third[third]) \
                    + calculate_distance_between_indices(problem, path_third[third], path_third[third + 1])
                after = calculate_distance_between_indices(problem, path_first[first - 1], path_third[third]) \
                    + calculate_distance_between_indices(problem, path_third[third], path_first[first + 1]) \
                    + calculate_distance_between_indices(problem, path_second[second - 1], path_first[first]) \
                    + calculate_distance_between_indices(problem, path_first[first], path_second[second + 1]) \
                    + calculate_distance_between_indices(problem, path_third[third - 1], path_second[second]) \
                    + calculate_distance_between_indices(problem, path_second[second], path_third[third + 1])
                delta = after - before
                if delta < min_delta:
                    min_delta = delta
                    label = first, second, third
                    improved_path_first, improved_path_second, improved_path_third = do_eject_three_paths(
                        path_first, path_second, path_third, label[0], label[1], label[2])
                    return improved_path_first, improved_path_second, improved_path_third, min_delta, label
    if label is None:
        return None, None, None, None, None
    else:
        improved_path_first, improved_path_second, improved_path_third = do_eject_three_paths(
            path_first, path_second, path_third, label[0], label[1], label[2])
        return improved_path_first, improved_path_second, improved_path_third, min_delta, label


def improve_solution(problem, solution):
    improved_solution = copy.deepcopy(solution)
    all_delta = 0.0
    num_paths = len(improved_solution)

    for path_index in range(num_paths):
        improved_path, delta, label = two_opt_path(problem, improved_solution[path_index])
        if label:
            improved_solution[path_index] = improved_path
            all_delta += delta

        improved_path, delta, label = exchange_path(problem, improved_solution[path_index])
        if label:
            improved_solution[path_index] = improved_path
            all_delta += delta

        improved_path, delta, label = relocate_path(problem, improved_solution[path_index])
        if label:
            improved_solution[path_index] = improved_path
            all_delta += delta

    for path_index_first in range(num_paths - 1):
        for path_index_second in range(path_index_first + 1, num_paths):
            improved_path_first, improved_path_second, delta, label = cross_two_paths(
                problem, improved_solution[path_index_first], improved_solution[path_index_second])
            if label:
                improved_solution[path_index_first] = improved_path_first
                improved_solution[path_index_second] = improved_path_second
                all_delta += delta

            improved_path_first, improved_path_second, delta, label = exchange_two_paths(
                problem, improved_solution[path_index_first], improved_solution[path_index_second])
            if label:
                improved_solution[path_index_first] = improved_path_first
                improved_solution[path_index_second] = improved_path_second
                all_delta += delta

            improved_path_first, improved_path_second, delta, label = relocate_two_paths(
                problem, improved_solution[path_index_first], improved_solution[path_index_second])
            if label:
                improved_solution[path_index_first] = improved_path_first
                improved_solution[path_index_second] = improved_path_second
                all_delta += delta
            improved_path_first, improved_path_second, delta, label = relocate_two_paths(
                problem, improved_solution[path_index_second], improved_solution[path_index_first])
            if label:
                improved_solution[path_index_first] = improved_path_second
                improved_solution[path_index_second] = improved_path_first
                all_delta += delta
    return improved_solution, all_delta


def  get_exact_lengths_for_exchange_two_paths(action):
    if action in [5, 6, 7]:
        return [action - 4, action - 4]
    elif action in range(12, 25):
        exact_lengths = [
            [1, 2],
            [1, 3],
            [2, 1],
            [2, 3],
            [3, 1],
            [3, 2],
            [1, 4],
            [4, 1],
            [2, 4],
            [4, 2],
            [3, 4],
            [4, 3],
            [4, 4],
        ]
        return exact_lengths[action - 12]
    else:
        return None


def improve_solution_by_action(step, problem, solution, action):
    improved_solution = copy.deepcopy(solution)
    all_delta = 0.0
    num_paths = len(improved_solution)

    if action in ([1, 2, 3] + range(25, 28)) or config.problem == 'tsp':
        for path_index in range(num_paths):
            modified = problem.should_try(action, path_index)
            while modified:
                if action == 1:
                    improved_path, delta, label = two_opt_path(problem, improved_solution[path_index])
                elif action == 2:
                    improved_path, delta, label = exchange_path(problem, improved_solution[path_index])
                else:
                    exact_lengths = {
                        3: 1,
                        4: 2,
                        5: 3,
                        6: 4,
                        25: 2,
                        26: 3,
                        27: 4
                    }
                    improved_path, delta, label = relocate_path(problem, improved_solution[path_index], exact_length=exact_lengths[action])
                if label:
                    modified = True
                    problem.mark_change_at(step, [path_index])
                    improved_solution[path_index] = improved_path
                    all_delta += delta
                else:
                    modified = False
                    problem.mark_no_improvement(step, action, path_index)
        return improved_solution, all_delta

    for path_index_first in range(num_paths - 1):
        for path_index_second in range(path_index_first + 1, num_paths):
            modified = problem.should_try(action, path_index_first, path_index_second)
            if action in ([4, 5, 6, 7] + range(12, 25)):
                while modified:
                    if action == 4:
                        improved_path_first, improved_path_second, delta, label = cross_two_paths(
                            problem, improved_solution[path_index_first], improved_solution[path_index_second])
                        if not label:
                            improved_path_first, improved_path_second, delta, label = cross_two_paths(
                                problem, improved_solution[path_index_first], improved_solution[path_index_second][::-1])
                    else:
                        improved_path_first, improved_path_second, delta, label = exchange_two_paths(
                            problem, improved_solution[path_index_first], improved_solution[path_index_second],
                            get_exact_lengths_for_exchange_two_paths(action))
                    if label:
                        modified = True
                        problem.mark_change_at(step, [path_index_first, path_index_second])
                        improved_solution[path_index_first] = improved_path_first
                        improved_solution[path_index_second] = improved_path_second
                        all_delta += delta
                    else:
                        modified = False
                        problem.mark_no_improvement(step, action, path_index_first, path_index_second)

            while action in [8, 9, 10] and modified:
                modified = False
                improved_path_first, improved_path_second, delta, label = relocate_two_paths(
                    problem, improved_solution[path_index_first], improved_solution[path_index_second], action - 7)
                if label:
                    modified = True
                    problem.mark_change_at(step, [path_index_first, path_index_second])
                    improved_solution[path_index_first] = improved_path_first
                    improved_solution[path_index_second] = improved_path_second
                    all_delta += delta
                improved_path_first, improved_path_second, delta, label = relocate_two_paths(
                    problem, improved_solution[path_index_second], improved_solution[path_index_first], action - 7)
                if label:
                    modified = True
                    problem.mark_change_at(step, [path_index_first, path_index_second])
                    improved_solution[path_index_first] = improved_path_second
                    improved_solution[path_index_second] = improved_path_first
                    all_delta += delta
                if not modified:
                    problem.mark_no_improvement(step, action, path_index_first, path_index_second)

            while action == 11 and modified:
                # return improved_solution, all_delta
                modified = False
                improved_path_first, improved_path_second, delta, customer_index = eject_two_paths(
                    problem, improved_solution[path_index_first], improved_solution[path_index_second])
                if customer_index:
                    for path_index_third in range(num_paths):
                        if path_index_third == path_index_first or path_index_third == path_index_second:
                            continue
                        improved_path_third, delta_insert, label = insert_into_path(improved_solution[path_index_third], customer_index)
                        if label is not None and (delta + delta_insert) < -EPSILON:
                            modified = True
                            problem.mark_change_at(step, [path_index_first, path_index_second, path_index_third])
                            improved_solution[path_index_first] = improved_path_first
                            improved_solution[path_index_second] = improved_path_second
                            improved_solution[path_index_third] = improved_path_third
                            all_delta += delta + delta_insert
                            break
                improved_path_first, improved_path_second, delta, customer_index = eject_two_paths(
                    problem, improved_solution[path_index_second], improved_solution[path_index_first])
                if customer_index:
                    for path_index_third in range(num_paths):
                        if path_index_third == path_index_first or path_index_third == path_index_second:
                            continue
                        improved_path_third, delta_insert, label = insert_into_path(improved_solution[path_index_third], customer_index)
                        if label is not None and (delta + delta_insert) < -EPSILON:
                            modified = True
                            problem.mark_change_at(step, [path_index_first, path_index_second, path_index_third])
                            improved_solution[path_index_first] = improved_path_second
                            improved_solution[path_index_second] = improved_path_first
                            improved_solution[path_index_third] = improved_path_third
                            all_delta += delta + delta_insert
                            break
                if not modified:
                    problem.mark_no_improvement(step, action, path_index_first, path_index_second)
    return improved_solution, all_delta


def dense_to_one_hot(labels_dense, num_training_points):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * config.num_training_points
  labels_one_hot = np.zeros((num_labels, config.num_training_points))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


def reshape_input(input, x, y, z):
    return np.reshape(input, (x, y, z))


is_training = tf.placeholder(tf.bool)
keep_prob = tf.placeholder(tf.float32)


def build_multi_operator_model(raw_input):
    input_sequence = tf.unstack(raw_input, config.num_training_points, 1)
    lstm_fw_cell = rnn.BasicLSTMCell(config.num_lstm_units, forget_bias=1.0)
    lstm_bw_cell = rnn.BasicLSTMCell(config.num_lstm_units, forget_bias=1.0)
    lstm_output, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, input_sequence, dtype=tf.float32)
    layer_1 = lstm_output[-1]

    layer_2 = tf.contrib.layers.fully_connected(layer_1, config.num_feedforward_units, activation_fn=tf.nn.relu)
    layer_2 = tf.contrib.layers.fully_connected(layer_2, config.num_lstm_units, activation_fn=None)
    layer_2 += layer_1
    layer_2 = tf.contrib.layers.batch_norm(layer_2, is_training=is_training)
    layer_2 = tf.nn.relu(layer_2)
    # layer_2 = tf.nn.dropout(layer_2, keep_prob)

    output_layer = tf.contrib.layers.fully_connected(layer_2, config.num_training_points, activation_fn=None)
    return output_layer


def get_random_capacities(n):
    capacities = [0] * n
    if config.problem == 'vrp':
        depot_capacity_map = {
            10: 20,
            20: 30,
            50: 40,
            100: 50
        }
        capacities[0] = depot_capacity_map.get(n - 1, 50)
        for i in range(1, n):
            capacities[i] = np.random.randint(9) + 1
    return capacities


def sample_next_index(to_indices, adjusted_distances):
    if len(to_indices) == 0:
        return 0
    adjusted_probabilities = np.asarray([1.0 / max(d, EPSILON) for d in adjusted_distances])
    adjusted_probabilities /= np.sum(adjusted_probabilities)
    return np.random.choice(to_indices, p=adjusted_probabilities)
    # return to_indices[np.argmax(adjusted_probabilities)]


def calculate_replacement_cost(problem, from_index, to_indices):
    return problem.get_distance(from_index, to_indices[0]) + problem.get_distance(from_index, to_indices[2]) \
        - problem.get_distance(to_indices[1], to_indices[0]) - problem.get_distance(to_indices[1], to_indices[2])


class Graph:
    def __init__(self, problem, nodes):
        self.nodes = nodes
        self.num_nodes = len(nodes)
        self.distance_matrix = np.zeros((self.num_nodes, self.num_nodes))
        for from_index in range(self.num_nodes):
            for to_index in range(from_index + 1, self.num_nodes):
                self.distance_matrix[from_index][to_index] = calculate_replacement_cost(problem, nodes[from_index][1], nodes[to_index])
                self.distance_matrix[to_index][from_index] = calculate_replacement_cost(problem, nodes[to_index][1], nodes[from_index])

    def find_negative_cycle(self):
        distance = [float('inf')] * self.num_nodes
        predecessor = [None] * self.num_nodes
        source = 0
        distance[source] = 0.0

        for i in range(1, self.num_nodes):
            improved = False
            for u in range(self.num_nodes):
                for v in range(self.num_nodes):
                    w = self.distance_matrix[u][v]
                    if distance[u] + w < distance[v]:
                        distance[v] = distance[u] + w
                        predecessor[v] = u
                        improved = True
            if not improved:
                break

        for u in range(self.num_nodes):
            for v in range(self.num_nodes):
                w = self.distance_matrix[u][v]
                if distance[u] + w + EPSILON < distance[v]:
                    visited = [0] * self.num_nodes
                    negative_cycle = []
                    negative_cycle.append(self.nodes[v][-2:])
                    count = 1
                    while (u != v) and (not visited[u]):
                        negative_cycle.append(self.nodes[u][-2:])
                        visited[u] = count
                        count += 1
                        u = predecessor[u]
                    if u != v:
                        negative_cycle = negative_cycle[visited[u]:]
                    return negative_cycle[::-1], -1.0

        num_cyclic_perturb = 4
        cutoff = 0.3
        if self.num_nodes >= num_cyclic_perturb:
            candidate_cycles = []
            for index in range(self.num_nodes):
                candidate_cycles.append(([index], 0.0))
            for index_to_choose in range(1, num_cyclic_perturb):
                next_candidate_cycles = []
                for cycle in candidate_cycles:
                    nodes = cycle[0]
                    total_distance = cycle[1]
                    for index in range(self.num_nodes):
                        if index not in nodes:
                            if index_to_choose == num_cyclic_perturb - 1:
                                new_total_distance = total_distance + self.distance_matrix[nodes[-1]][index] + self.distance_matrix[index][nodes[0]]
                            else:
                                new_total_distance = total_distance + self.distance_matrix[nodes[-1]][index]
                            if new_total_distance < cutoff:
                                next_candidate_cycles.append((nodes + [index], new_total_distance))
                candidate_cycles = next_candidate_cycles
            # if len(candidate_cycles):
            #     print('count={}'.format(candidate_cycles))
            if len(candidate_cycles) > 0:
                random_indices = np.random.choice(range(len(candidate_cycles)), 1)[0]
                random_indices = candidate_cycles[random_indices][0]
                negative_cycle = []
                for u in random_indices:
                    negative_cycle.append(self.nodes[u][-2:])
                return negative_cycle, 1.0
        return None, None


def construct_graph(problem, solution, capacity):
    nodes = []
    for path_index in range(len(solution)):
        path = solution[path_index]
        if len(path) > 2:
            node_index = 1
            while node_index < len(path) - 1:
                node_index_end = node_index + 1
                if problem.get_capacity(path[node_index]) == capacity:
                    while problem.get_capacity(path[node_index_end]) == capacity:
                        node_index_end += 1
                    sampled_node_index = np.random.choice(range(node_index, node_index_end))
                    nodes.append([path[sampled_node_index - 1], path[sampled_node_index], path[sampled_node_index + 1],
                                  path_index, sampled_node_index])
                node_index = node_index_end
    graph = Graph(problem, nodes)
    return graph


def get_path_from_cycle(cycle):
    index_of_0 = 0
    while True:
        if cycle[index_of_0] == 0:
            break
        else:
            index_of_0 += 1
    path = cycle[index_of_0:] + cycle[:(index_of_0 + 1)]
    return path


def get_cycle_from_path(path):
    return path[1:]


def construct_solution(problem, existing_solution=None, step=0):
    solution = []
    n = problem.get_num_customers()
    customer_indices = range(n + 1)
    if config.problem == 'tsp':
        num_customers = n + 1
        if existing_solution is None:
            cycle = np.random.permutation(num_customers).tolist()
            path = get_path_from_cycle(cycle)
        else:
            num_customers_to_shuffle = min(config.max_num_customers_to_shuffle, num_customers)
            start_index = np.random.randint(num_customers)
            indices_permuted = np.random.permutation(num_customers_to_shuffle)
            cycle = get_cycle_from_path(existing_solution[0])
            cycle_perturbed = copy.copy(cycle)
            for index in range(num_customers_to_shuffle):
                to_index = start_index + index
                if to_index >= num_customers:
                    to_index -= num_customers
                from_index = start_index + indices_permuted[index]
                if from_index >= num_customers:
                    from_index -= num_customers
                cycle_perturbed[to_index] = cycle[from_index]
            path = get_path_from_cycle(cycle_perturbed)
        solution.append(path)
        problem.reset_change_at_and_no_improvement_at()
        return solution

    if (existing_solution is not None) and (config.num_paths_to_ruin != -1):
        distance = calculate_solution_distance(problem, existing_solution)
        min_reconstructed_distance = float('inf')
        solution_to_return = None
        # for _ in range(10):
        for _ in range(1):
            reconstructed_solution = reconstruct_solution(problem, existing_solution, step)
            reconstructed_distance = calculate_solution_distance(problem, reconstructed_solution)
            if reconstructed_distance / distance <= 1.05:
                solution_to_return = reconstructed_solution
                break
            else:
                if reconstructed_distance < min_reconstructed_distance:
                    min_reconstructed_distance = reconstructed_distance
                    solution_to_return = reconstructed_solution
        return solution_to_return
    else:
        start_customer_index = 1

    trip = [0]
    capacity_left = problem.get_capacity(0)
    i = start_customer_index
    while i <= n:
        random_index = np.random.randint(low=i, high=n+1)

        # if len(trip) > 1:
        #     min_index, min_distance = random_index, float('inf')
        #     for j in range(i, n + 1):
        #         if problem.get_capacity(customer_indices[j]) > capacity_left:
        #             continue
        #         distance = calculate_distance_between_indices(problem, trip[-1], customer_indices[j])
        #         if distance < min_distance:
        #             min_index, min_distance = j, distance
        #     random_index = min_index

        # if len(trip) > 1:
        #     min_index, min_distance = 0, calculate_adjusted_distance_between_indices(problem, trip[-1], 0)
        # else:
        #     min_index, min_distance = random_index, float('inf')
        # for j in range(i, n + 1):
        #     if problem.get_capacity(customer_indices[j]) > capacity_left:
        #         continue
        #     distance = calculate_adjusted_distance_between_indices(problem, trip[-1], customer_indices[j])
        #     if distance < min_distance:
        #         min_index, min_distance = j, distance
        # random_index = min_index

        to_indices = []
        adjusted_distances = []
        # if len(trip) > 1:
        #     to_indices.append(0)
        #     adjusted_distances.append(calculate_adjusted_distance_between_indices(problem, trip[-1], 0))
        for j in range(i, n + 1):
            if problem.get_capacity(customer_indices[j]) > capacity_left:
                continue
            to_indices.append(j)
            adjusted_distances.append(calculate_adjusted_distance_between_indices(problem, trip[-1], customer_indices[j]))
        random_index = sample_next_index(to_indices, adjusted_distances)

        if random_index == 0 or capacity_left < problem.get_capacity(customer_indices[random_index]):
            trip.append(0)
            solution.append(trip)
            trip = [0]
            capacity_left = problem.get_capacity(0)
            continue
        customer_indices[i], customer_indices[random_index] = customer_indices[random_index], customer_indices[i]
        trip.append(customer_indices[i])
        capacity_left -= problem.get_capacity(customer_indices[i])
        i += 1
    if len(trip) > 1:
        trip.append(0)
        solution.append(trip)
    solution.append([0, 0])

    problem.reset_change_at_and_no_improvement_at()
    return solution

def reconstruct_solution_by_exchange(problem, existing_solution, paths_ruined):
    path0 = copy.deepcopy(existing_solution[paths_ruined[0]])
    path1 = copy.deepcopy(existing_solution[paths_ruined[1]])
    num_exchanged = 0
    for i in range(1, len(path0) - 1):
        for j in range(1, len(path1) - 1):
            if problem.get_capacity(path0[i]) == problem.get_capacity(path1[j]):
                #TODO
                if problem.get_distance(path0[i], path1[j]) < 0.2:
                    path0[i], path1[j] = path1[j], path0[i]
                    num_exchanged += 1
                    break
    if num_exchanged >= 0:
        return [path0, path1]
    else:
        return []


def reconstruct_solution(problem, existing_solution, step):
    distance_hash = round(calculate_solution_distance(problem, existing_solution) * 1e6)
    if config.detect_negative_cycle and distance_hash not in problem.distance_hashes:
        problem.add_distance_hash(distance_hash)
        positive_cycles = []
        cycle_selected = None
        for capacity in range(1, 10):
            # TODO: relax the requirement of ==capacity
            # TODO: caching, sparsify
            graph = construct_graph(problem, existing_solution, capacity)
            negative_cycle, flag = graph.find_negative_cycle()
            if negative_cycle:
                if flag == -1.0:
                    cycle_selected = negative_cycle
                    break
                else:
                    positive_cycles.append(negative_cycle)
        if cycle_selected is None and len(positive_cycles) > 0:
            index = np.random.choice(range(len(positive_cycles)), 1)[0]
            cycle_selected = positive_cycles[index]
        if cycle_selected is not None:
                negative_cycle = cycle_selected
                improved_solution = copy.deepcopy(existing_solution)
                customers = []
                for pair in negative_cycle:
                    path_index, node_index = pair[0], pair[1]
                    customers.append(improved_solution[path_index][node_index])
                customers = [customers[-1]] + customers[:-1]
                for index in range(len(negative_cycle)):
                    pair = negative_cycle[index]
                    path_index, node_index = pair[0], pair[1]
                    improved_solution[path_index][node_index] = customers[index]
                    problem.mark_change_at(step, [path_index])
                # if not validate_solution(problem, improved_solution):
                #     print('existing_solution={}, invalid improved_solution={}, negative_cycle={}'.format(
                #         existing_solution, improved_solution, negative_cycle))
                # else:
                #     print('cost={}, negative_cycle={}'.format(
                #         calculate_solution_distance(problem, improved_solution) - calculate_solution_distance(problem, existing_solution),
                #         negative_cycle)
                #     )
                return improved_solution

    solution = []
    n = problem.get_num_customers()
    customer_indices = range(n + 1)

    candidate_indices = []
    for path_index in range(len(existing_solution)):
        if len(existing_solution[path_index]) > 2:
            candidate_indices.append(path_index)
    paths_ruined = np.random.choice(candidate_indices, config.num_paths_to_ruin, replace=False)
    start_customer_index = n + 1
    for path_index in paths_ruined:
        path = existing_solution[path_index]
        for customer_index in path:
            if customer_index == 0:
                continue
            start_customer_index -= 1
            customer_indices[start_customer_index] = customer_index

    if np.random.uniform() < 0.5:
        while len(solution) == 0:
            paths_ruined = np.random.choice(candidate_indices, config.num_paths_to_ruin, replace=False)
            solution = reconstruct_solution_by_exchange(problem, existing_solution, paths_ruined)
    else:
        trip = [0]
        capacity_left = problem.get_capacity(0)
        i = start_customer_index
        while i <= n:
            to_indices = []
            adjusted_distances = []
            # if len(trip) > 1:
            #     to_indices.append(0)
            #     adjusted_distances.append(calculate_adjusted_distance_between_indices(problem, trip[-1], 0))
            for j in range(i, n + 1):
                if problem.get_capacity(customer_indices[j]) > capacity_left:
                    continue
                to_indices.append(j)
                adjusted_distances.append(
                    calculate_adjusted_distance_between_indices(problem, trip[-1], customer_indices[j]))
            random_index = sample_next_index(to_indices, adjusted_distances)

            if random_index == 0:
                trip.append(0)
                solution.append(trip)
                trip = [0]
                capacity_left = problem.get_capacity(0)
                continue
            customer_indices[i], customer_indices[random_index] = customer_indices[random_index], customer_indices[i]
            trip.append(customer_indices[i])
            capacity_left -= problem.get_capacity(customer_indices[i])
            i += 1
        if len(trip) > 1:
            trip.append(0)
            solution.append(trip)

    while len(solution) < len(paths_ruined):
        solution.append([0, 0])
    improved_solution = copy.deepcopy(existing_solution)
    solution_index = 0
    for path_index in sorted(paths_ruined):
        improved_solution[path_index] = copy.deepcopy(solution[solution_index])
        solution_index += 1
    problem.mark_change_at(step, paths_ruined)
    for solution_index in range(len(paths_ruined), len(solution)):
        improved_solution.append(copy.deepcopy(solution[solution_index]))
        problem.mark_change_at(step, [len(improved_solution) - 1])

    has_seen_empty_path = False
    for path_index in range(len(improved_solution)):
        if len(improved_solution[path_index]) == 2:
            if has_seen_empty_path:
                empty_slot_index = path_index
                for next_path_index in range(path_index + 1, len(improved_solution)):
                    if len(improved_solution[next_path_index]) > 2:
                        improved_solution[empty_slot_index] = copy.deepcopy(improved_solution[next_path_index])
                        empty_slot_index += 1
                improved_solution = improved_solution[:empty_slot_index]
                problem.mark_change_at(step, range(path_index, empty_slot_index))
                break
            else:
                has_seen_empty_path = True
    return improved_solution


def get_num_points(config):
    if config.model_to_restore is None:
        return config.num_training_points
    else:
        return config.num_test_points


def generate_problem():
    np.random.seed(config.problem_seed)
    random.seed(config.problem_seed)
    config.problem_seed += 1

    num_sample_points = get_num_points(config)
    if config.problem == 'vrp':
        num_sample_points += 1
    locations = np.random.uniform(size=(num_sample_points, 2))
    if config.problem == 'vrp':
        if config.depot_positioning == 'C':
            locations[0][0] = 0.5
            locations[0][1] = 0.5
        elif config.depot_positioning == 'E':
            locations[0][0] = 0.0
            locations[0][1] = 0.0
        if config.customer_positioning in {'C', 'RC'}:
            S = np.random.randint(6) + 3
            centers = locations[1 : (S + 1)]
            grid_centers, probabilities = [], []
            for x in range(0, 1000):
                for y in range(0, 1000):
                    grid_center = [(x + 0.5) / 1000.0, (y + 0.5) / 1000.0]
                    p = 0.0
                    for center in centers:
                        distance = calculate_distance(grid_center, center)
                        p += math.exp(-distance * 1000.0 / 40.0)
                    grid_centers.append(grid_center)
                    probabilities.append(p)
            probabilities = np.asarray(probabilities) / np.sum(probabilities)
            if config.customer_positioning in 'C':
                num_clustered_locations = get_num_points(config) - S
            else:
                num_clustered_locations = get_num_points(config) // 2 - S
            grid_indices = np.random.choice(range(len(grid_centers)), num_clustered_locations, p=probabilities)
            for index in range(num_clustered_locations):
                grid_index = grid_indices[index]
                locations[index + S + 1][0] = grid_centers[grid_index][0] + (np.random.uniform() - 0.5) / 1000.0
                locations[index + S + 1][1] = grid_centers[grid_index][1] + (np.random.uniform() - 0.5) / 1000.0

    capacities = get_random_capacities(len(locations))
    problem = Problem(locations, capacities)
    np.random.seed(config.problem_seed * 10)
    random.seed(config.problem_seed * 10)
    return problem


ATTENTION_ROLLOUT, LSTM_ROLLOUT = False, True


def embedding_net_nothing(input_):
    return input_


def embedding_net_2(input_):
    with tf.variable_scope("embedding_net"):
        architecture_type = 0
        if architecture_type == 0:
            x = embed_seq(input_seq=input_, from_=config.input_embedded_trip_dim_2, to_=config.num_embedded_dim_1, is_training=True,
                          BN=True, initializer=tf.contrib.layers.xavier_initializer())

            layer_attention = encode_seq(input_seq=x, input_dim=config.num_embedded_dim_1, num_stacks=1, num_heads=8,
                                         num_neurons=64, is_training=True, dropout_rate=0.1)
            # layer_attention = tf.reshape(layer_attention, [-1, (config.num_training_points) * config.num_embedded_dim_1])
            # layer_2 = tf.contrib.layers.fully_connected(layer_attention, config.num_embedded_dim_2, activation_fn=tf.nn.relu)
            # layer_2 = tf.nn.dropout(layer_2, keep_prob)
            layer_2 = tf.reduce_sum(layer_attention, axis=1)
        else:
            #TODO:
            x = embed_seq(input_seq=input_, from_=config.input_embedded_trip_dim_2, to_=config.num_embedded_dim_1, is_training=True, BN=False, initializer=tf.contrib.layers.xavier_initializer())
            x = tf.reduce_sum(x, axis=1)
            layer_2 = tf.nn.relu(x)
    return layer_2


def embedding_net(input_):
    with tf.variable_scope("embedding_net"):
        first_trip = input_[:, 0, :, :]
        first_trip = tf.reshape(first_trip, [-1, config.max_points_per_trip, config.input_embedded_trip_dim])
        trip_embedding = embedding_net_lstm(first_trip)
        for trip_index in range(1, config.max_trips_per_solution):
            with tf.variable_scope("lstm", reuse=True):
                trip = input_[:, trip_index, :, :]
                trip = tf.reshape(trip, [-1, config.max_points_per_trip, config.input_embedded_trip_dim])
                current_trip_embedding = embedding_net_lstm(trip)
                trip_embedding = tf.concat([trip_embedding, current_trip_embedding], axis=1)
        attention_embedding = embedding_net_attention(trip_embedding)
    return attention_embedding


def embed_trip(trip, points_in_trip):
    trip_prev = np.vstack((trip[-1], trip[:-1]))
    trip_next = np.vstack((trip[1:], trip[0]))
    distance_from_prev = np.reshape(np.linalg.norm(trip_prev - trip, axis=1), (points_in_trip, 1))
    distance_to_next = np.reshape(np.linalg.norm(trip - trip_next, axis=1), (points_in_trip, 1))
    distance_from_to_next = np.reshape(np.linalg.norm(trip_prev - trip_next, axis=1), (points_in_trip, 1))
    trip_with_additional_information = np.hstack((trip_prev, trip, trip_next, distance_from_prev, distance_to_next, distance_from_to_next))
    return trip_with_additional_information


def embedding_net_lstm(input_):
    seq = tf.unstack(input_, config.max_points_per_trip, 1)
    num_hidden = 128
    with tf.variable_scope("lstm_embeding", reuse=tf.AUTO_REUSE):
        lstm_fw_cell1 = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
        lstm_bw_cell1 = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
        try:
            outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell1, lstm_bw_cell1, seq, dtype=tf.float32)
        except Exception:  # Old TensorFlow version only returns outputs not states
            outputs = rnn.static_bidirectional_rnn(lstm_fw_cell1, lstm_bw_cell1, seq, dtype=tf.float32)
        layer_lstm = outputs[-1]
        layer_2 = tf.contrib.layers.fully_connected(layer_lstm, config.num_embedded_dim, activation_fn=tf.nn.relu)
        layer_2 = tf.nn.dropout(layer_2, keep_prob)
        layer_2 = tf.reshape(layer_2, [-1, 1, config.num_embedded_dim])
    return layer_2


def embedding_net_attention(input_):
    with tf.variable_scope("attention_embedding"):
        x = embed_seq(input_seq=input_, from_=config.num_embedded_dim, to_=128, is_training=True, BN=True, initializer=tf.contrib.layers.xavier_initializer())
        layer_attention = encode_seq(input_seq=x, input_dim=128, num_stacks=3, num_heads=16, num_neurons=512, is_training=True, dropout_rate=0.1)
        layer_attention = tf.reshape(layer_attention, [-1, config.max_trips_per_solution * 128])
        layer_2 = tf.contrib.layers.fully_connected(layer_attention, config.num_embedded_dim, activation_fn=tf.nn.relu)
        layer_2 = tf.nn.dropout(layer_2, keep_prob)
    return layer_2


def embed_solution(problem, solution):
    embedded_solution = np.zeros((config.max_trips_per_solution, config.max_points_per_trip, config.input_embedded_trip_dim))
    n_trip = len(solution)
    for trip_index in range(min(config.max_trips_per_solution, n_trip)):
        trip = solution[trip_index]
        truncated_trip_length = np.minimum(config.max_points_per_trip, len(trip) - 1)
        if truncated_trip_length > 1:
            points_with_coordinate = np.zeros((truncated_trip_length, 2))
            for point_index in range(truncated_trip_length):
                points_with_coordinate[point_index] = problem.get_location(trip[point_index])
            embedded_solution[trip_index, :truncated_trip_length] = copy.deepcopy(embed_trip(points_with_coordinate, truncated_trip_length))
    return embedded_solution


def embed_solution_with_nothing(problem, solution):
    embedded_solution = np.zeros((config.max_trips_per_solution, config.max_points_per_trip, config.input_embedded_trip_dim))
    return embedded_solution


def embed_solution_with_attention(problem, solution):
    embedded_solution = np.zeros((config.num_training_points, config.input_embedded_trip_dim_2))

    for path in solution:
        if len(path) == 2:
            continue
        n = len(path) - 1
        consumption = calculate_consumption(problem, path)
        for index in range(1, n):
            customer = path[index]
            embedded_input = []
            embedded_input.append(problem.get_capacity(customer))
            embedded_input.extend(problem.get_location(customer))
            embedded_input.append(problem.get_capacity(0) - consumption[-1])
            embedded_input.extend(problem.get_location(path[index - 1]))
            embedded_input.extend(problem.get_location(path[index + 1]))
            embedded_input.append(problem.get_distance(path[index - 1], customer))
            embedded_input.append(problem.get_distance(customer, path[index + 1]))
            embedded_input.append(problem.get_distance(path[index - 1], path[index + 1]))
            for embedded_input_index in range(len(embedded_input)):
                embedded_solution[customer - 1, embedded_input_index] = embedded_input[embedded_input_index]
    return embedded_solution


TEST_X = tf.placeholder(tf.float32, [None, config.num_training_points, config.input_embedded_trip_dim_2])
embedded_x = embedding_net_2(TEST_X)
env_observation_space_n = config.num_history_action_use * 2 + 5
action_labels_placeholder = tf.placeholder("float", [None, config.num_actions - 1])


class PolicyEstimator():
    """
    Policy Function approximator.
    """

    def __init__(self, learning_rate=config.policy_learning_rate, scope="policy_estimator"):
        with tf.variable_scope(scope):
            self.states = tf.placeholder(tf.float32, [None, env_observation_space_n], "states")
            if config.use_attention_embedding:
                full_states = tf.concat([self.states, embedded_x], axis=1)
            else:
                full_states = self.states

            self.hidden1 = tf.contrib.layers.fully_connected(
                inputs=full_states,
                num_outputs=config.hidden_layer_dim,
                activation_fn=tf.nn.relu)
            self.logits = tf.contrib.layers.fully_connected(
                inputs=self.hidden1,
                num_outputs=config.num_actions - 1,
                activation_fn=None)
            #https://stackoverflow.com/questions/33712178/tensorflow-nan-bug?newreg=c7e31a867765444280ba3ca50b657a07
            self.action_probs = tf.clip_by_value(tf.nn.softmax(self.logits), 1e-10, 1.0)

            # training part of graph
            self.actions = tf.placeholder(tf.int32, [None], name="actions")
            self.advantages = tf.placeholder(tf.float32, [None], name="advantages")
            log_prob = tf.log(self.action_probs)
            indices = tf.range(0, tf.shape(log_prob)[0]) * tf.shape(log_prob)[1] + self.actions
            act_prob = tf.gather(tf.reshape(log_prob, [-1]), indices)
            self.loss = -tf.reduce_sum(act_prob * self.advantages)
            # self.loss = -tf.reduce_mean(act_prob * self.advantages)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())

            # Define loss and optimizer
            self.sl_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=action_labels_placeholder))
            self.sl_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.sl_train_op = self.sl_optimizer.minimize(self.sl_loss)
            # Training accuracy
            correct_pred = tf.equal(tf.argmax(self.action_probs, 1), tf.argmax(action_labels_placeholder, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def predict(self, states, test_x, sess=None):
        sess = sess or tf.get_default_session()
        if config.use_attention_embedding:
            feed_dict = {self.states: states, TEST_X: test_x, keep_prob: 1.0}
        else:
            feed_dict = {self.states: states, keep_prob:1.0}
        return sess.run(self.action_probs, feed_dict)

    def update(self, states, test_x, advantages, actions, sess=None):
        sess = sess or tf.get_default_session()
        if config.use_attention_embedding:
            feed_dict = {self.states: states, self.advantages: advantages, self.actions: actions, TEST_X:test_x, keep_prob:1.0}
        else:
            feed_dict = {self.states: states, self.advantages: advantages, self.actions: actions, keep_prob: 1.0}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss

    def train(self, states, test_x, action_labels, sess=None):
        sess = sess or tf.get_default_session()
        if config.use_attention_embedding:
            feed_dict = {self.states: states, TEST_X:test_x, action_labels_placeholder: action_labels, keep_prob:1.0}
        else:
            feed_dict = {self.states: states, action_labels_placeholder: action_labels, keep_prob:1.0}
        _, loss, accuracy = sess.run([self.sl_train_op, self.sl_loss, self.accuracy], feed_dict)
        return loss, accuracy


previous_solution = None
initial_solution = None
best_solution = None


def env_act(step, problem, min_distance, solution, distance, action):
    global initial_solution
    global previous_solution
    if action > 0:
        next_solution, delta = improve_solution_by_action(step, problem, solution, action)
        next_distance = distance + delta
        if config.debug_mode:
            if not validate_solution(problem, next_solution, next_distance):
                print('Invalid solution!')
    else:
        problem.record_solution(solution, distance)
        if distance / min_distance < 1.01:
            previous_solution = solution
            next_solution = construct_solution(problem, solution, step)
        else:
            previous_solution = best_solution
            next_solution = construct_solution(problem, best_solution, step)
            # problem.reset_change_at_and_no_improvement_at()
        next_distance = calculate_solution_distance(problem, next_solution)
        initial_solution = next_solution
    return next_solution, next_distance


action_timers = [0.0] * (config.num_actions * 2)


def env_generate_state(min_distance=None, state=None, action=None, distance=None, next_distance=None):
    if state is None or action == 0:
        next_state = [0.0, 0.0, 0]
        for _ in range(config.num_history_action_use):
            next_state.append(0.0)
            next_state.append(0)
        next_state.append(0.0)
        next_state.append(0)
    else:
        delta = next_distance - distance
        if delta < -EPSILON:
            delta_sign = -1.0
        else:
            delta_sign = 1.0
        next_state = [0.0, next_distance - min_distance, delta]
        if config.num_history_action_use != 0:
            next_state.extend(state[(-config.num_history_action_use * 2):])
        next_state.append(delta_sign)
        next_state.append(action)
    return next_state


def env_step(step, state, problem, min_distance, solution, distance, action, record_time=True):
    start_timer = datetime.datetime.now()
    next_trip, next_distance = env_act(step, problem, min_distance, solution, distance, action)
    next_state = env_generate_state(min_distance, state, action, distance, next_distance)
    reward = distance - next_distance
    end_timer = datetime.datetime.now()
    if record_time:
        action_timers[action * 2] += 1
        action_timers[action * 2 + 1] += (end_timer - start_timer).total_seconds()
    done = (datetime.datetime.now() - env_start_time).total_seconds() >= config.max_rollout_seconds
    return next_state, reward, done, next_trip, next_distance


def format_print(value):
    return round(float(value), 2)


def format_print_array(values):
    results = []
    for value in values:
        results.append(format_print(value))
    return results


def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    # print ([str(i.name) for i in not_initialized_vars])
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))


def should_restart(min_distance, distance, no_improvement):
    if no_improvement >= config.max_no_improvement:
        return True
    if no_improvement <= config.max_no_improvement - 1:
        return False
    percentage_over = round((distance / min_distance - 1.0) * 100)
    upper_limits = [20, 10, 5, 2]
    return percentage_over >= upper_limits[no_improvement - 2]


def get_edge_set(solution):
    edge_set = set()
    for path in solution:
        if len(path) > 2:
            for path_index in range(1, len(path)):
                node_before = path[path_index - 1]
                node_current = path[path_index]
                value = '{}_{}'.format(min(node_before, node_current), max(node_before, node_current))
                edge_set.add(value)
    return edge_set


def calculate_solution_similarity(solutions):
    edge_set = get_edge_set(solutions[0])
    for solution in solutions[1:]:
        edge_set = edge_set.intersection(get_edge_set(solution))
    return len(edge_set)


gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
with tf.Session(config=gpu_config) as sess:
    policy_estimator = PolicyEstimator()
    initialize_uninitialized(sess)
    print(sess.run(tf.report_uninitialized_variables()))
    variables_names = [v.name for v in tf.trainable_variables()]
    values = sess.run(variables_names)
    for k, v in zip(variables_names, values):
        print("Variable={}, Shape={}".format(k, v.shape))
    sys.stdout.flush()
    saver = tf.train.Saver()
    if config.model_to_restore is not None:
        saver.restore(sess, config.model_to_restore)

    distances = []
    steps = []
    consolidated_distances, consolidated_steps = [], []
    timers = []
    num_checkpoint = int(config.max_rollout_steps/config.step_interval)
    step_record = np.zeros((2, num_checkpoint))
    distance_record = np.zeros((2, num_checkpoint))
    start = datetime.datetime.now()
    seed = config.problem_seed
    tf.set_random_seed(seed)

    Transition = collections.namedtuple("Transition", ["state", "trip", "next_distance", "action", "reward", "next_state", "done"])
    for index_sample in range(config.num_episode):
        states = []
        trips = []
        actions = []
        advantages = []
        action_labels = []
        if index_sample > 0 and index_sample % config.debug_steps == 0:
            if not config.use_random_rollout:
                formatted_timers = format_print_array(np.mean(np.asarray(timers), axis=0))
                count_timers = formatted_timers[4:][::2]
                time_timers = formatted_timers[4:][1::2]
                print('time ={}'.format('\t\t'.join([str(x) for x in time_timers])))
                print('count={}'.format('\t\t'.join([str(x) for x in count_timers])))
                start_active = ((len(distances) // config.num_active_learning_iterations) * config.num_active_learning_iterations)
                if start_active == len(distances):
                    start_active -= config.num_active_learning_iterations
                tail_distances = distances[start_active:]
                tail_steps = steps[start_active:]
                min_index = np.argmin(tail_distances)
                if config.num_active_learning_iterations == 1 or len(distances) % config.num_active_learning_iterations == 1:
                    consolidated_distances.append(tail_distances[min_index])
                    consolidated_steps.append(tail_steps[min_index] + min_index * config.max_rollout_steps)
                else:
                    consolidated_distances[-1] = tail_distances[min_index]
                    consolidated_steps[-1] = tail_steps[min_index] + min_index * config.max_rollout_steps
                print('index_sample={}, mean_distance={}, mean_step={}, tail_distance={}, last_distance={}, last_step={}, timers={}'.format(
                    index_sample,
                    format_print(np.mean(consolidated_distances)), format_print(np.mean(consolidated_steps)),
                    format_print(np.mean(consolidated_distances[max(0, len(consolidated_distances) - 1000):])),
                    format_print(consolidated_distances[-1]), consolidated_steps[-1],
                    formatted_timers[:4]
                ))
                sys.stdout.flush()
            else:
                formatted_timers = format_print_array(np.mean(np.asarray(timers), axis=0))
                for index in range(num_checkpoint):
                    print('rollout_num={}, index_sample={}, mean_distance={}, mean_step={}, last_distance={}, last_step={}, timers={}'.format(
                        (index + 1) * config.step_interval, index_sample, ((index_sample - 1) * distance_record[0, index] + distance_record[1, index]) / index_sample,
                        ((index_sample - 1) * step_record[0, index] + step_record[1, index]) / index_sample, distance_record[1, index],
                        step_record[1, index], formatted_timers[:4]
                    ))
                    step_record[0, index] = ((index_sample - 1) * step_record[0, index] + step_record[1, index]) / index_sample
                    distance_record[0, index] = ((index_sample - 1) * distance_record[0, index] + distance_record[1, index]) / index_sample
                sys.stdout.flush()

        problem = generate_problem()
        solution = construct_solution(problem)
        best_solution = copy.deepcopy(solution)

        if config.use_attention_embedding:
            embedded_trip = embed_solution_with_attention(problem, solution)
        else:
            embedded_trip = [0]
        min_distance = calculate_solution_distance(problem, solution)
        min_step = 0
        distance = min_distance

        state = env_generate_state()
        env_start_time = datetime.datetime.now()
        episode = []
        current_best_distances = []
        start_distance = distance
        current_distances = []
        start_distances = []

        inference_time = 0
        gpu_inference_time = 0
        env_act_time = 0
        no_improvement = 0
        loop_action = 0
        num_random_actions = 0
        for action_index in range(len(action_timers)):
            action_timers[action_index] = 0.0
        for step in range(config.max_rollout_steps):
            start_timer = datetime.datetime.now()
            if config.use_cyclic_rollout:
                choices = [1, 3, 4, 5, 8]
                if no_improvement == len(choices) + 1:
                    action = 0
                    no_improvement = 0
                else:
                    action = choices[loop_action]
                    loop_action += 1
                    if loop_action == len(choices):
                        loop_action = 0
            elif config.use_random_rollout:
                action = random.randint(0, config.num_actions - 1)
            else:
                gpu_start_time = datetime.datetime.now()
                action_probs = policy_estimator.predict([state], [embedded_trip], sess)
                gpu_inference_time += (datetime.datetime.now() - gpu_start_time).total_seconds()
                action_probs = action_probs[0]
                history_action_probs = np.zeros(len(action_probs))
                action_prob_sum = 0.0
                for action_prob_index in range(len(action_probs)):
                    action_prob_sum += action_probs[action_prob_index]
                for action_prob_index in range(len(action_probs)):
                    action_probs[action_prob_index] /= action_prob_sum
                if config.use_history_action_distribution and (index_sample > 0):
                    history_action_count_sum = 0
                    for action_count_index in range(len(action_probs)):
                        history_action_count_sum += count_timers[action_count_index + 1]
                    for action_count_index in range(len(action_probs)):
                        history_action_probs[action_count_index] = count_timers[action_count_index + 1]/history_action_count_sum
                        action_probs[action_count_index] = action_probs[action_count_index]/2 + history_action_probs[action_count_index]/2


                if config.use_rl_loss:
                    states.append(state)
                    trips.append(embedded_trip)
                elif random.uniform(0, 1) < 0.05:
                    action_label = [0] * config.num_actions
                    action_index = 0
                    min_action_time = sys.maxint
                    rewards = []
                    action_times = []
                    for action_to_label in range(1, config.num_actions):
                        action_start_time = datetime.datetime.now()
                        _, reward, _, _, _ = env_step(step, state, problem, min_distance, solution, distance, action_to_label, False)
                        action_time = (datetime.datetime.now() - action_start_time).total_seconds()
                        rewards.append(reward)
                        action_times.append(action_time)
                        if reward > EPSILON and action_time < min_action_time:
                            action_index = action_to_label
                            min_action_time = action_time
                            break
                    action_label[action_index] = 1
                    states.append(state)
                    trips.append(embedded_trip)
                    action_labels.append(action_label)

                if (config.model_to_restore is not None and should_restart(min_distance, distance, no_improvement)) or no_improvement >= config.max_no_improvement:
                    action = 0
                    no_improvement = 0
                else:
                    if np.random.uniform() < config.epsilon_greedy:
                        action = np.random.randint(config.num_actions - 1) + 1
                        num_random_actions += 1
                    else:
                        if config.sample_actions_in_rollout:
                            action = np.random.choice(np.arange(len(action_probs)), p=action_probs) + 1
                        else:
                            action = np.argmax(action_probs) + 1
            end_timer = datetime.datetime.now()
            inference_time += (end_timer - start_timer).total_seconds()
            start_timer = end_timer

            next_state, reward, done, next_solution, next_distance = env_step(step, state, problem, min_distance, solution, distance, action)
            if next_distance >= distance - EPSILON:
                no_improvement += 1
            else:
                #TODO
                no_improvement = 0

            current_distances.append(distance)
            start_distances.append(start_distance)
            if action == 0:
                start_distance = next_distance
            current_best_distances.append(min_distance)
            if next_distance < min_distance - EPSILON:
                min_distance = next_distance
                min_step = step
                best_solution = copy.deepcopy(next_solution)
            if (step + 1) % config.step_interval == 0:
                print('rollout_num={}, index_sample={}, min_distance={}, min_step={}'.format(
                    step + 1, index_sample, min_distance, min_step
                ))
                temp_timers = np.asarray(action_timers)
                temp_count_timers = temp_timers[::2]
                temp_time_timers = temp_timers[1::2]
                print('time ={}'.format('\t\t'.join([str(x) for x in temp_time_timers])))
                print('count={}'.format('\t\t'.join([str(x) for x in temp_count_timers])))
            if done:
                break

            episode.append(Transition(
                state=state, trip=copy.deepcopy(embedded_trip), next_distance=next_distance,
                action=action, reward=reward, next_state=next_state, done=done))
            state = next_state
            solution = next_solution
            if config.use_attention_embedding:
                embedded_trip = embed_solution_with_attention(problem, solution)
            else:
                embedded_trip = [0]
            distance = next_distance
            end_timer = datetime.datetime.now()
            env_act_time += (end_timer - start_timer).total_seconds()
            start_timer = end_timer

        if config.use_random_rollout:
            temp = np.inf
            for rollout_step in range(num_checkpoint):
                current_region_min_step = np.argmin(current_distances[(rollout_step * config.step_interval):((rollout_step + 1) * config.step_interval)]) + rollout_step * config.step_interval
                current_region_min_distance = min(current_distances[(rollout_step * config.step_interval):((rollout_step + 1) * config.step_interval)])
                if temp > current_region_min_distance:
                    distance_record[1, rollout_step] = current_region_min_distance
                    step_record[1, rollout_step] = current_region_min_step
                    temp = current_region_min_distance
                else:
                    distance_record[1, rollout_step] = distance_record[1, rollout_step - 1]
                    step_record[1, rollout_step] = step_record[1, rollout_step - 1]

        start_timer = datetime.datetime.now()
        distances.append(min_distance)
        steps.append(min_step)
        if validate_solution(problem, best_solution, min_distance):
            print('solution={}'.format(best_solution))
        else:
            print('invalid solution')
        if not (config.use_cyclic_rollout or config.use_random_rollout):
            future_best_distances = [0.0] * len(episode)
            future_best_distances[-1] = episode[len(episode) - 1].next_distance
            step = len(episode) - 2
            while step >= 0:
                if episode[step].action != 0:
                    future_best_distances[step] = future_best_distances[step + 1] * config.discount_factor
                else:
                    future_best_distances[step] = current_distances[step]
                step = step - 1

            historical_baseline = None
            for t, transition in enumerate(episode):
                # total_return = sum(config.discount_factor**i * future_transition.reward for i, future_transition in enumerate(episode[t:]))
                if historical_baseline is None:
                    if transition.action == 0:
                        #TODO: dynamic updating of historical baseline, and state definition
                        historical_baseline = -current_best_distances[t]
                        # historical_baseline = 1/(current_best_distances[t] - 10)
                    actions.append(0)
                    advantages.append(0)
                    continue
                # if transition.action == 0:
                #     historical_baseline = -current_distances[t]
                if transition.action > 0:
                    # total_return = transition.reward
                    if transition.reward < EPSILON:
                        total_return = -1.0
                    else:
                        total_return = 1.0
                    #     total_return = min(transition.reward, 2.0)
                    # total_return = start_distances[t] - future_best_distances[t]
                    # total_return = min(total_return, 1.0)
                    # total_return = max(total_return, -1.0)
                    total_return = -future_best_distances[t]
                    # total_return = 1/(future_best_distances[t] - 10)
                else:
                    if transition.state[-1] != 0 and transition.state[-2] < 1e-6:
                        # if future_best_distances[t] < current_best_distances[t] - 1e-6:
                        total_return = 1.0
                    else:
                        total_return = -1.0
                    total_return = 0
                    actions.append(0)
                    advantages.append(0)
                    continue
                # baseline_value = value_estimator.predict(states)
                # baseline_value = 0.0
                baseline_value = historical_baseline
                advantage = total_return - baseline_value
                actions.append(transition.action)
                advantages.append(advantage)
                # value_estimator.update(states, [[total_return]], sess)

            states = np.reshape(np.asarray(states), (-1, env_observation_space_n)).astype("float32")
            if config.use_attention_embedding:
                trips = np.reshape(np.asarray(trips), (-1, config.num_training_points, config.input_embedded_trip_dim_2)).astype("float32")
            actions = np.reshape(np.asarray(actions), (-1))
            advantages = np.reshape(np.asarray(advantages), (-1)).astype("float32")
            if config.use_rl_loss:
                print('num_random_actions={}'.format(num_random_actions))
                print('actions={}'.format(actions[:100]).replace('\n', ''))
                print('advantages={}'.format(advantages[:100]).replace('\n', ''))
                if config.model_to_restore is None and index_sample <= config.max_num_training_epsisodes:
                    filtered_states = []
                    filtered_trips = []
                    filtered_advantages = []
                    filtered_actions = []
                    end = 0
                    for action_index in range(len(actions)):
                        if actions[action_index] > 0:
                            filtered_states.append(states[action_index])
                            filtered_trips.append(trips[action_index])
                            filtered_advantages.append(advantages[action_index])
                            filtered_actions.append(actions[action_index] - 1)
                        else:
                            num_bad_steps = config.max_no_improvement
                            end = max(end, len(filtered_states) - num_bad_steps)
                            filtered_states = filtered_states[:end]
                            filtered_trips = filtered_trips[:end]
                            filtered_advantages = filtered_advantages[:end]
                            filtered_actions = filtered_actions[:end]
                    filtered_states = filtered_states[:end]
                    filtered_trips = filtered_trips[:end]
                    filtered_advantages = filtered_advantages[:end]
                    filtered_actions = filtered_actions[:end]
                    num_states = len(filtered_states)
                    if config.use_attention_embedding and num_states > config.batch_size:
                        downsampled_indices = np.random.choice(range(num_states), config.batch_size, replace=False)
                        filtered_states = np.asarray(filtered_states)[downsampled_indices]
                        filtered_trips = np.asarray(filtered_trips)[downsampled_indices]
                        filtered_advantages = np.asarray(filtered_advantages)[downsampled_indices]
                        filtered_actions = np.asarray(filtered_actions)[downsampled_indices]
                    loss = policy_estimator.update(filtered_states, filtered_trips, filtered_advantages, filtered_actions, sess)
                    print('loss={}'.format(loss))
            else:
                #TODO: filter and reshape
                action_labels = np.reshape(np.asarray(action_labels), (-1, config.num_actions))
                loss, accuracy = policy_estimator.train(states, trips, action_labels, sess)
                print('loss={}, accuracy={}'.format(loss, accuracy))
        timers_epoch = [inference_time, gpu_inference_time, env_act_time, (datetime.datetime.now() - start_timer).total_seconds()]
        timers_epoch.extend(action_timers)
        timers.append(timers_epoch)
        if config.model_to_restore is None and index_sample > 0 and index_sample % 500 == 0:
            save_path = saver.save(sess, "./rollout_model_{}_{}_{}.ckpt".format(index_sample, config.num_history_action_use, config.max_rollout_steps))
            print("Model saved in path: %s" % save_path)

    # save_path = saver.save(sess, "./rollout_model.ckpt")
    # print("Model saved in path: %s" % save_path)
    print('solving time = {}'.format(datetime.datetime.now() - start))
