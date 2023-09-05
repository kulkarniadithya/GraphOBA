import numpy as np
from collections import namedtuple
import pickle
import random
import sys
import copy

# I followed the message passing implementation from https://jessicastringham.net/2019/01/09/sum-product-message-passing/

LabeledArray = namedtuple('LabeledArray', [
    'array',
    'axes_labels',
])


def name_to_axis_mapping(labeled_array):
    return {
        name: axis
        for axis, name in enumerate(labeled_array.axes_labels)
    }


def other_axes_from_labeled_axes(labeled_array, axis_label):
    return tuple(
        axis
        for axis, name in enumerate(labeled_array.axes_labels)
        if name != axis_label
    )


def is_conditional_prob(labeled_array, var_name):
    return np.all(np.isclose(np.sum(np.array(
        labeled_array.array),
        axis=name_to_axis_mapping(labeled_array)[var_name]
    ), np.array(1.0)))


def is_joint_prob(labeled_array):
    return np.all(np.isclose(np.sum(labeled_array.array), np.array(1.0)))


def tile_to_shape_along_axis(arr, target_shape, target_axis):
    raw_axes = list(range(len(target_shape)))
    tile_dimensions = [target_shape[a] for a in raw_axes if a != target_axis]
    if len(arr.shape) == 0:
        tile_dimensions += [target_shape[target_axis]]
    elif len(arr.shape) == 1:
        assert arr.shape[0] == target_shape[target_axis]
        tile_dimensions += [1]
    else:
        raise NotImplementedError()
    tiled = np.tile(arr, tile_dimensions)

    shifted_axes = raw_axes[:target_axis] + [raw_axes[-1]] + raw_axes[target_axis:-1]
    transposed = np.transpose(tiled, shifted_axes)

    assert transposed.shape == target_shape
    return transposed


def tile_to_other_dist_along_axis_name(tiling_labeled_array, target_array):
    assert len(tiling_labeled_array.axes_labels) == 1
    target_axis_label = tiling_labeled_array.axes_labels[0]

    return LabeledArray(
        tile_to_shape_along_axis(
            tiling_labeled_array.array,
            target_array.array.shape,
            name_to_axis_mapping(target_array)[target_axis_label]
        ),
        axes_labels=target_array.axes_labels
    )


class Node(object):
    def __init__(self, name):
        self.name = name
        self.neighbors = []

    def __repr__(self):
        return "{classname}({name}, [{neighbors}])".format(
            classname=type(self).__name__,
            name=self.name,
            neighbors=', '.join([n.name for n in self.neighbors])
        )

    def is_valid_neighbor(self, neighbor):
        raise NotImplemented()

    def add_neighbor(self, neighbor):
        assert self.is_valid_neighbor(neighbor)
        self.neighbors.append(neighbor)


class Variable(Node):
    def is_valid_neighbor(self, factor):
        return isinstance(factor, Factor)

    def variable_name(self):
        return self.name


class Factor(Node):
    def is_valid_neighbor(self, variable):
        return isinstance(variable, Variable)

    def __init__(self, name):
        super(Factor, self).__init__(name)
        self.data = None


ParsedTerm = namedtuple('ParsedTerm', [
    'term',
    'var_name',
    'given',
])


def _parse_term(term):
    assert term[0] == '(' and term[-1] == ')'
    term_variables = term[1:-1]

    if '|' in term_variables:
        var, given = term_variables.split('|')
        given = given.split(',')
    else:
        var = term_variables
        given = []
    return var, given


def _parse_model_string_into_terms(model_string):
    return [
        ParsedTerm('p' + term, *_parse_term(term))
        for term in model_string.split('p')
        if term
    ]


def parse_model_into_variables_and_factors(model_string):
    parsed_terms = _parse_model_string_into_terms(model_string)

    variables = {}
    for parsed_term in parsed_terms:
        if parsed_term.var_name not in variables:
            variables[parsed_term.var_name] = Variable(parsed_term.var_name)
        given = parsed_term.given
        for i in range(0, len(given)):
            if given[i] not in variables:
                variables[given[i]] = Variable(given[i])
    factors = []
    for parsed_term in parsed_terms:
        new_factor = Factor(parsed_term.term)
        all_var_names = [parsed_term.var_name] + parsed_term.given
        for var_name in all_var_names:
            new_factor.add_neighbor(variables[var_name])
            variables[var_name].add_neighbor(new_factor)
        factors.append(new_factor)

    return factors, variables


class PGM(object):
    def __init__(self, factors, variables):
        self._factors = factors
        self._variables = variables

    @classmethod
    def from_string(cls, model_string):
        factors, variables = parse_model_into_variables_and_factors(model_string)
        return PGM(factors, variables)

    def set_data(self, data):
        var_dims = {}
        for factor in self._factors:
            factor_data = data[factor.name]

            if set(factor_data.axes_labels) != set(v.name for v in factor.neighbors):
                missing_axes = set(v.name for v in factor.neighbors) - set(data[factor.name].axes_labels)
                raise ValueError("data[{}] is missing axes: {}".format(factor.name, missing_axes))

            for var_name, dim in zip(factor_data.axes_labels, factor_data.array.shape):
                if var_name not in var_dims:
                    var_dims[var_name] = dim

                if var_dims[var_name] != dim:
                    raise ValueError(
                        "data[{}] axes is wrong size, {}. Expected {}".format(factor.name, dim, var_dims[var_name]))

            factor.data = data[factor.name]

    def variable_from_name(self, var_name):
        return self._variables[var_name]


class Messages(object):
    def __init__(self, variable_prior, variable_labels, worker_labels, leaf_nodes, messages=None, message_counter=None, iteration_counter=0):
        if messages is None:
            messages = {}
        self.messages = messages
        if message_counter is None:
            message_counter = {}
        self.message_counter = message_counter
        self.variable_prior = variable_prior
        self.variable_labels = variable_labels
        self.worker_labels = worker_labels
        self.iteration_counter = iteration_counter
        self.leaf_nodes = leaf_nodes

    def _variable_to_factor_messages(self, variable, factor):
        incoming_messages = [
            self.factor_to_variable_message(neighbor_factor, variable)
            for neighbor_factor in variable.neighbors
            if neighbor_factor.name != factor.name
        ]

        return self.variable_prior[variable]*np.prod(np.array(incoming_messages), axis=0)

    def _factor_to_variable_messages(self, factor, variable):
        factor_dist = np.copy(factor.data.array)
        for neighbor_variable in factor.neighbors:
            if neighbor_variable.name == variable.name:
                continue
            incoming_message = self.variable_to_factor_messages(neighbor_variable, factor)
            factor_dist *= tile_to_other_dist_along_axis_name(
                LabeledArray(incoming_message, [neighbor_variable.name]),
                factor.data
            ).array
        other_axes = other_axes_from_labeled_axes(factor.data, variable.name)
        return np.squeeze(np.sum(factor_dist, axis=other_axes))

    def _variable_to_factor_messages_update(self, variable, factor, counter):
        incoming_messages = []
        for neighbor_factor in variable.neighbors:
            if neighbor_factor.name != factor.name:
                counter = counter + 1
                if counter < 200:
                    incoming_messages.append(self.factor_to_variable_message_update(neighbor_factor, variable, counter))
                else:
                    break

        if len(incoming_messages) > 0:
            output = np.array(self.variable_prior[variable.name]) * np.exp(np.sum(np.log(np.array(incoming_messages)), axis=0))
        else:
            output = self.variable_prior[variable.name]
        return np.array(output) / np.sum(np.array(output))

    def _factor_to_variable_messages_update(self, factor, variable, counter):
        factor_dist = np.copy(factor.data.array)
        for neighbor_variable in factor.neighbors:
            if neighbor_variable.name == variable.name:
                continue
            counter = counter + 1
            incoming_message = self.variable_to_factor_message_update(neighbor_variable, factor, counter)
            factor_dist *= tile_to_other_dist_along_axis_name(
                LabeledArray(incoming_message, [neighbor_variable.name]),
                factor.data
            ).array
        other_axes = other_axes_from_labeled_axes(factor.data, variable.name)
        return np.squeeze(np.sum(factor_dist, axis=other_axes))

    def marginal(self, variable):
        unnorm_p = np.prod(np.array([
            self.factor_to_variable_message_update(neighbor_factor, variable, 0)
            for neighbor_factor in variable.neighbors
        ]), axis=0)
        unnorm_p = np.array(self.variable_prior[variable.name])*unnorm_p
        return np.array(unnorm_p) / np.sum(np.array(unnorm_p))

    def forward_propagation(self, variable, label=None):
        for neighbor_factor in variable.neighbors:
            self.factor_to_variable_message_update(neighbor_factor, variable, 0)
        self.update_variable_prior(variable, label)
        self.iteration_counter = self.iteration_counter + 1

    def backward_propagation(self):
        for i in range(0, len(self.leaf_nodes)):
            variable = pgm.variable_from_name(self.leaf_nodes[i])
            for neighbor_factor in variable.neighbors:
                self.factor_to_variable_message_update(neighbor_factor, variable, 0)

    def compute_iteration(self, variable, label=None):
        self.forward_propagation(variable, label)
        self.backward_propagation()

    def compute_iteration_iterative(self, variable):
        for neighbor_factor in variable.neighbors:
            self.factor_to_variable_message_update(neighbor_factor, variable, 0)
        self.backward_propagation()

    def update_variable_prior(self, variable, label):
        if label is None:
            length_variable_labels = len(self.variable_labels[variable.name])
            label = self.worker_labels[variable.name][length_variable_labels]
            self.variable_labels[variable.name].append(label)
            self.variable_prior[variable.name] = self.recompute_prior(self.variable_labels[variable.name])
        else:
            self.variable_labels[variable.name].append(label)
            self.variable_prior[variable.name] = self.recompute_prior(self.variable_labels[variable.name])

    @staticmethod
    def recompute_prior(array):
        count_positive = 0.1
        count_negative = 0.1
        for i in range(0, len(array)):
            if array[i] == 1:
                count_positive = count_positive + 1
            else:
                count_negative = count_negative + 1
        output = [count_negative/(count_positive + count_negative), count_positive/(count_positive + count_negative)]
        return output

    def variable_to_factor_messages(self, variable, factor):
        message_name = (variable.name, factor.name)
        if message_name not in self.messages:
            self.messages[message_name] = self._variable_to_factor_messages(variable, factor)
            self.message_counter[message_name] = 0
        return self.messages[message_name]

    def factor_to_variable_message(self, factor, variable):
        message_name = (factor.name, variable.name)
        if message_name not in self.messages:
            self.messages[message_name] = self._factor_to_variable_messages(factor, variable)
            self.message_counter[message_name] = 0
        return self.messages[message_name]

    def variable_to_factor_message_update(self, variable, factor, counter):
        message_name = (variable.name, factor.name)
        if message_name in self.messages:
            if self.message_counter[message_name] < self.iteration_counter:
                self.messages[message_name] = self._variable_to_factor_messages_update(variable, factor, counter)
                self.message_counter[message_name] = self.message_counter[message_name] + 1
        elif message_name not in self.messages:
            self.messages[message_name] = self._variable_to_factor_messages_update(variable, factor, counter)
            self.message_counter[message_name] = self.iteration_counter
        return self.messages[message_name]

    def factor_to_variable_message_update(self, factor, variable, counter):
        message_name = (factor.name, variable.name)
        if message_name in self.messages:
            if self.message_counter[message_name] < self.iteration_counter:
                self.messages[message_name] = self._factor_to_variable_messages_update(factor, variable, counter)
                self.message_counter[message_name] = self.message_counter[message_name] + 1
        elif message_name not in self.messages:
            self.messages[message_name] = self._factor_to_variable_messages_update(factor, variable, counter)
            self.message_counter[message_name] = self.iteration_counter
        return self.messages[message_name]


if __name__ == "__main__":
    print(sys.getrecursionlimit())
    sys.setrecursionlimit(2500)
    print(sys.getrecursionlimit())
    np.random.seed(111)
    random_state = 111
    random.seed(random_state)
    factor_setting = "55_45"
    with open("../processed_data/cora/factor_graph_initialization_setting_" + str(factor_setting) + ".pickle", "rb") as handle:
        data_dictionary = pickle.load(handle)

    degree_dict = data_dictionary['degree_dict']
    leaf_nodes = data_dictionary['leaf_nodes']
    variable_prior = data_dictionary['variable_prior']
    variable_labels = data_dictionary['variable_labels']
    worker_labels = data_dictionary['worker_labels']
    edge_list = data_dictionary['updated_edge_list']
    factor_probability = data_dictionary['factor_probability']
    probability_string = data_dictionary['probability_string']
    graph_nodes = data_dictionary['graph_nodes']
    graph_out_degree = data_dictionary['graph_out_degree']

    print("Loading done")

    pgm = PGM.from_string("".join(probability_string))
    data = {}
    for i in range(0, len(probability_string)):
        data[probability_string[i]] = LabeledArray(np.array(factor_probability[i]), edge_list[i])
    pgm.set_data(data)

    m_max_10 = Messages(variable_prior=copy.deepcopy(variable_prior), variable_labels=copy.deepcopy(variable_labels), worker_labels=copy.deepcopy(worker_labels),
                 leaf_nodes=copy.deepcopy(leaf_nodes))
    m_sum_10 = Messages(variable_prior=copy.deepcopy(variable_prior), variable_labels=copy.deepcopy(variable_labels),
                  worker_labels=copy.deepcopy(worker_labels),
                  leaf_nodes=copy.deepcopy(leaf_nodes))
    m_random = Messages(variable_prior=copy.deepcopy(variable_prior), variable_labels=copy.deepcopy(variable_labels), worker_labels=copy.deepcopy(worker_labels),
                        leaf_nodes=copy.deepcopy(leaf_nodes))
    print(edge_list[0][1])

    to_choose = list(graph_nodes)
    nodes = list(variable_prior.keys())
    print(to_choose)
    print(nodes)
    to_test_labels = [-1, 1]
    belief_choices_max_10 = {}
    belief_choices_sum_10 = {}
    belief_choices_random = {}
    sample_dict = {}
    for z in range(0, 3001):
        print(z)
        belief_choices_max_10[z] = {}
        belief_choices_sum_10[z] = {}
        belief_choices_random[z] = {}
        sample = random.sample(to_choose, 10)
        sample_dict[z] = sample
        random_choice = random.choice(sample)
        marginal_dict_current_max_10 = []
        marginal_dict_current_sum_10 = []
        for i in range(0, len(nodes)):
            marginal_dict_current_max_10.append(max(m_max_10.marginal(pgm.variable_from_name(nodes[i]))))
            marginal_dict_current_sum_10.append(max(m_sum_10.marginal(pgm.variable_from_name(nodes[i]))))
        marginals_max_10 = []
        marginals_sum_10 = []
        for i in range(0, len(sample)):
            sample_marginals_max_10 = []
            sample_marginals_sum_10 = []
            for k in range(0, len(to_test_labels)):
                m1_max_10 = Messages(variable_prior=copy.deepcopy(m_max_10.variable_prior), variable_labels=copy.deepcopy(m_max_10.variable_labels), worker_labels=copy.deepcopy(m_max_10.worker_labels),
                         leaf_nodes=copy.deepcopy(m_max_10.leaf_nodes), messages=copy.deepcopy(m_max_10.messages), message_counter=copy.deepcopy(m_max_10.message_counter), iteration_counter=copy.deepcopy(m_max_10.iteration_counter))
                m1_max_10.compute_iteration(pgm.variable_from_name(str(sample[i])), label=to_test_labels[k])

                m1_sum_10 = Messages(variable_prior=copy.deepcopy(m_sum_10.variable_prior),
                                     variable_labels=copy.deepcopy(m_sum_10.variable_labels),
                                     worker_labels=copy.deepcopy(m_sum_10.worker_labels),
                                     leaf_nodes=copy.deepcopy(m_sum_10.leaf_nodes),
                                     messages=copy.deepcopy(m_sum_10.messages),
                                     message_counter=copy.deepcopy(m_sum_10.message_counter),
                                     iteration_counter=copy.deepcopy(m_sum_10.iteration_counter))
                m1_sum_10.compute_iteration(pgm.variable_from_name(str(sample[i])), label=to_test_labels[k])

                sample_marginal_next_max_10 = []
                sample_marginal_next_sum_10 = []
                for j in range(0, len(nodes)):
                    sample_marginal_next_max_10.append(max(m1_max_10.marginal(pgm.variable_from_name(nodes[j]))))
                    sample_marginal_next_sum_10.append(max(m1_sum_10.marginal(pgm.variable_from_name(nodes[j]))))
                diff_max_10 = np.sum(np.subtract(np.array(sample_marginal_next_max_10), np.array(marginal_dict_current_max_10)), axis=0)
                diff_sum_10 = np.sum(
                    np.subtract(np.array(sample_marginal_next_sum_10), np.array(marginal_dict_current_sum_10)), axis=0)
                # print(diff)
                sample_marginals_max_10.append(diff_max_10)
                sample_marginals_sum_10.append(diff_sum_10)
                del m1_max_10
                del m1_sum_10
            marginals_max_10.append(max(sample_marginals_max_10))
            marginals_sum_10.append(sample_marginals_sum_10[0]*m_sum_10.variable_prior[str(sample[i])][0] + sample_marginals_sum_10[1]*m_sum_10.variable_prior[str(sample[i])][1])
        # print(marginals)
        maximum_max_10 = max(marginals_max_10)
        index_max_10 = marginals_max_10.index(maximum_max_10)

        maximum_sum_10 = max(marginals_sum_10)
        index_sum_10 = marginals_sum_10.index(maximum_sum_10)

        # print(index)
        belief_choices_max_10[z]['selected_node'] = sample[index_max_10]
        belief_choices_sum_10[z]['selected_node'] = sample[index_sum_10]
        belief_choices_random[z]['selected_node'] = random_choice
        m_max_10.compute_iteration(pgm.variable_from_name(str(sample[index_max_10])))
        for y in range(0, 4):
            m_max_10.compute_iteration_iterative(pgm.variable_from_name(str(sample[y])))

        m_sum_10.compute_iteration(pgm.variable_from_name(str(sample[index_sum_10])))
        for y in range(0, 4):
            m_sum_10.compute_iteration_iterative(pgm.variable_from_name(str(sample[y])))

        m_random.compute_iteration(pgm.variable_from_name(str(random_choice)))
        for y in range(0, 4):
            m_random.compute_iteration_iterative(pgm.variable_from_name(str(random_choice)))

        output_dict_max_10 = {'messages': m_max_10.messages, 'message_counter': m_max_10.message_counter, 'variable_prior': m_max_10.variable_prior,
                       'variable_labels': m_max_10.variable_labels,
                       'iteration_counter': m_max_10.iteration_counter, 'belief_choices': belief_choices_max_10,
                              'sample_dict': sample_dict}

        output_dict_sum_10 = {'messages': m_sum_10.messages, 'message_counter': m_sum_10.message_counter,
                              'variable_prior': m_sum_10.variable_prior,
                              'variable_labels': m_sum_10.variable_labels,
                              'iteration_counter': m_sum_10.iteration_counter,
                              'belief_choices': belief_choices_sum_10,
                              'sample_dict': sample_dict}

        output_dict_random = {'messages': m_random.messages, 'message_counter': m_random.message_counter,
                              'variable_prior': m_random.variable_prior,
                              'variable_labels': m_random.variable_labels,
                              'iteration_counter': m_random.iteration_counter,
                              'belief_choices': belief_choices_random,
                              'sample_dict': sample_dict}

        marginal_dict_max_10 = {}
        marginal_dict_sum_10 = {}
        marginal_dict_random = {}
        # nodes = list(variable_prior.keys())
        for a in range(0, len(nodes)):
            marginal_dict_max_10[nodes[a]] = m_max_10.marginal(pgm.variable_from_name(nodes[a]))
            marginal_dict_sum_10[nodes[a]] = m_sum_10.marginal(pgm.variable_from_name(nodes[a]))
            marginal_dict_random[nodes[a]] = m_random.marginal(pgm.variable_from_name(nodes[a]))

        output_dict_max_10['marginal_dict'] = marginal_dict_max_10
        output_dict_sum_10['marginal_dict'] = marginal_dict_sum_10
        output_dict_random['marginal_dict'] = marginal_dict_random

        with open("../processed_data/cora_" + str(factor_setting) + "/alpha_beta_0_1_factor_" + str(factor_setting) + "_max_10_" + str(z) + ".pickle", "wb") as handle:
            pickle.dump(output_dict_max_10, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open("../processed_data/cora_" + str(factor_setting) + "/alpha_beta_0_1_factor_" + str(factor_setting) + "_sum_10_" + str(z) + ".pickle", "wb") as handle:
            pickle.dump(output_dict_sum_10, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open("../processed_data/cora_" + str(factor_setting) + "/alpha_beta_0_1_factor_" + str(factor_setting) + "_random_10_" + str(z) + ".pickle", "wb") as handle:
            pickle.dump(output_dict_random, handle, protocol=pickle.HIGHEST_PROTOCOL)


