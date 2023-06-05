from __future__ import annotations

import functools
import random
from pprint import pformat


def multiply_vector_elements(vector):
    """ return the multiplication of the vector elements """

    def mult(x, y):
        return x * y

    return functools.reduce(mult, vector, 1)


class Variable(object):
    """ Node in the network. Represent a random Variable """

    def __init__(self, name: str, assignments: tuple[str, ...],
                 probability_table: dict[tuple[str, ...], tuple[float, ...]], parents: list[Variable] = None,
                 children: list[Variable] = None):
        """ Node initialization
            params:
            name: name of this random variable.
            assignments: possible values this variable can have.
            probability_table: the causal probability table of this variable.
            parents: list of references to this Node`s parents.
            children: list of references to this Node`s children.
        """

        if parents is None:
            parents = []

        # the name of this random variable
        self.name = name

        # holds the possible assignments of this random variable
        # assume certain order
        self.assignments: dict[str, int] = {}
        for i in range(len(assignments)):
            self.assignments[assignments[i]] = i

        # holds the distribution table of this random variable
        for key, val in probability_table.items():
            if len(val) != len(assignments):
                # self = None
                raise ValueError('data in probability table is inconsistent with possible assignments')

        self.probability_table: dict[tuple[str, ...], tuple[float, ...]] = probability_table

        # list of dependent variables
        self.children: list[Variable] = children if children is not None else []

        # list of variables which this variable depends upon
        self.parents: list[Variable] = parents

        # holds the marginal, pre-calculated probability to obtain each
        # possible value
        self.marginal_probabilities: list[float] = len(assignments) * [0]

        # indicates whether this node is ready to use
        # true when the marginal probabilities were calculated
        self.ready: bool = False

        self.calculate_marginal_probability()

    def get_name(self) -> str:
        """ return the name of this random variable """
        return self.name

    def get_assignments(self) -> dict[str, int]:
        """ return the possible values this variable can have """
        return self.assignments

    def get_assignment_index(self, assignment: str) -> int:
        """ returns the index of a given possible assignment within the assignments list """
        return self.assignments[assignment]

    def get_probability(self, value: str, parents_values: tuple[str, ...]) -> float:
        """ read from the distribution table and return the probability of having a
            certain value (value) given the values of the parents.
        """
        return self.probability_table[parents_values][self.assignments[value]]

    def get_conditional_probability(self, value: str, parents_values: dict[str, str]) -> float:
        """ read from the distribution table and return the probability of having a
            certain value (value) given the values of the parents.
            here the parents assignments can be partial
            parent_vals is a dictionary: { parent: value }
        """
        res: float = 0
        given_parents_index = []
        marginal_parents_index = []
        for i, v in enumerate(self.parents):
            if v.name in parents_values:
                given_parents_index.append((i, parents_values[v.name]))
            else:
                marginal_parents_index.append(i)

        # go over the rows in the distribution table
        for row_key, row_val in self.probability_table.items():
            valid_row = 1

            # check if this row should count for the marginal conditional
            # probability
            for gpi in given_parents_index:
                if row_key[gpi[0]] != gpi[1]:
                    valid_row = 0
                    break

            # if this row is valid, add the corresponding conditional
            # probability
            if valid_row:
                parents_probability = 1
                for mpi in marginal_parents_index:
                    parents_probability *= self.parents[mpi].get_marginal_probability(row_key[mpi])

                res += row_val[self.assignments[value]] * parents_probability
        return res

    def calculate_marginal_probability(self):
        """ calculates and stores the marginal probabilities of this node.
            this function should be call before any other calculation is done.
        """

        # return, if already done
        if self.ready:
            return

        # TODO: COMPLETE THIS FUNCTION
        # Set self.marginal_probabilities

        loops = 0
        for parent_combinations, val in self.probability_table.items():
            loops += 1
            for i, prob in enumerate(val):
                self.marginal_probabilities[i] += prob

        for i, _ in enumerate(self.marginal_probabilities):
            self.marginal_probabilities[i] /= loops

        # set this Node`s state to ready
        self.ready = True

    def get_marginal_probability(self, val: str) -> float:
        """ returns the marginal probability, to have a certain value """
        return self.marginal_probabilities[self.assignments[val]]

    def add_child(self, node):
        """ add dependent Variable to this variable """
        self.children.append(node)

    def add_parent(self, node):
        """ add a parent to this Variable """
        self.parents.append(node)

    def get_parents(self):
        """ returns the parent list """
        return self.parents

    def get_children(self):
        """ returns the children list """
        return self.children

    def is_child_of(self, node):
        """ return boolean, indicating whether this Node is a child of a given
            Node
        """
        for var in self.parents:
            if var.name == node.name:
                return 1
        return 0


class BayesianNetwork(object):
    """ Bayesian Network implementation. This implementation incorporates few
        assumptions (see comments).
    """

    def __init__(self):
        """ Initialize connectivity matrix. """
        self.variables: list[Variable] = []  # list of variables (Nodes)
        self.varsMap: dict[str, Variable] = {}  # a mapping of variable name to the actual node, for easy access
        self.ready: bool = False  # indication of this net state

    def calculate_marginal_probabilities(self) -> None:
        """ pre-calculate and stores the marginal probabilities of all the nodes """

        # iterate over the Nodes, from parents to children
        for var in self.variables:
            var.calculate_marginal_probability()
        self.ready = True

    def get_variables(self) -> list[Variable]:
        """ returns the variables """

        return self.variables

    def get_variable(self, varName: str) -> Variable:
        """ returns the variable with the given name """

        return self.varsMap[varName]

    def add_variable(self, var: Variable, index: int = -1) -> None:  # len(variables)):
        """ add a single Node to the net """

        if index < 0:
            self.variables.append(var)
        else:
            self.variables.insert(index, var)

        self.varsMap[var.name] = var
        self.ready = False  # we need to re-calculate marginals

    def set_variables(self, varList: list[Variable]) -> None:
        """ quick assignment: set the given Node list to be the Nodes of this
            net
        """

        self.variables = varList
        for var in self.variables:
            self.varsMap[var.name] = var
        self.ready = False  # we need to re-calculate marginals

        self.calculate_marginal_probabilities()

    def get_marginal_probability(self, var: Variable, val: str) -> float:
        """ returns the marginal probability of a given node """

        return var.get_marginal_probability(val)

    # values is dictionary
    def get_joint_probability(self, values: dict[str, str]) -> float:
        """ return the joint probability of the Nodes """

        if not self.ready:
            self.calculate_marginal_probabilities()

        prob_out: float = 1
        # Initialized to 1 allows us to *= in all iterations in the loop below since 1 * a = a
        # therefore the first loop still gives correct result

        for variable_string, value in values.items():
            variable = self.varsMap[variable_string]

            # Get parents values as dict[variable_name, Variable value]
            # These will be the evidents
            cond_evidents = {}
            for parent in variable.get_parents():
                if parent.name in values.keys():
                    cond_evidents[parent.name] = values[parent.name]
                else:
                    raise KeyError(f"{parent.name} is not present in values.keys: {values.keys()}")

            cond_input_values = {variable_string: value}

            prob_out *= self.get_conditional_probability(cond_input_values, cond_evidents)

        # Return join probability
        return prob_out

    def get_conditional_probability(self, values: dict[str, str], evidents: dict[str, str]) -> float:
        """ returns the conditional probability.
            Here I do not introduce advanced algorithms for inference (e.g. junctions trees)
            this method implement only simple inference, namely: the joint probability of children given their parents
            or the probability of parents given their children.
            assumption: variables in each level are independent, or independent given their parents
            (i.e vars in values are independent, as well as vars in evidents
        """
        res: float = 1

        # when we want probability of children given their parents
        # if self.varsMap[list(values.keys())[0]].is_child_of(self.varsMap[list(evidents.keys())[0]]):
        first_val = list(values.keys())[0]
        first_variable = self.varsMap[first_val]
        if all(first_variable.is_child_of(self.varsMap[evident]) for evident in evidents.keys()):
            # print('probability of children given their parents')
            for child, c_val in values.items():
                res *= self.varsMap[child].get_conditional_probability(c_val, evidents)

        # when we want probability of parents given their children
        # make use of Bayes rule
        # assumption: nodes in each level are independent, given their parents
        else:
            print('probability of parents given their children')

            joint_marginal_parents = 1
            joint_marginal_children = 1
            joint_conditional_children = 1
            marginal_of_evidents = 1

            # calculating the joint probability of the parents
            for parent, p_val in values.items():
                joint_marginal_parents *= self.varsMap[parent].get_marginal_probability(p_val)

            # calculating the joint probability of the children, and the joint joint probability
            # of the children given their parents
            for child, c_val in evidents.items():
                joint_marginal_children *= self.varsMap[child].get_marginal_probability(c_val)

                # children given their parents. here the values become the
                # evidents!
                joint_conditional_children *= self.varsMap[child].get_conditional_probability(c_val, values)

                k = list(values.keys())[0]
                complementary_conditional_values = values.copy()
                complementary_conditional_values[k] = 'false' if values[k] == 'true' else 'true'
                marginal_of_evidents = marginal_of_evidents * self.varsMap[child].get_conditional_probability(c_val,
                                                                                                              complementary_conditional_values)

                # print("Child: {}".format(child))
                # print("    Given: {}".format(complementary_conditional_values))

            # uses Bayes rule, for calculating the conditional probability
            res = (joint_conditional_children * joint_marginal_parents) / (
                    (joint_conditional_children * joint_marginal_parents) + marginal_of_evidents * (
                    1 - joint_marginal_parents))

        return res

    # helper method
    def sub_vals(self, var: Variable, values: dict[str, str]) -> tuple[str, ...]:
        """ return a tuple, contain all the relevant
            assignments for the given variable (i.e - the assignments
            pertaining to the variable`s parents."""
        sub = []
        for p in var.parents:
            sub.append(values[p.name])
        return tuple(sub)


def create_random_sample(network: BayesianNetwork) -> dict[str, str]:
    """ creates random sample for the given network.
        the distribution of the samples follows the joint probability function.
        assumes binary variables. """
    sample: dict[str, str] = {}
    for var in network.variables:

        samp = random.random()
        assignment1 = list(var.assignments.keys())[0]
        assignment2 = list(var.assignments.keys())[1]

        parents_values = network.sub_vals(var, sample)
        prob = var.get_probability(assignment1, parents_values)

        if samp <= prob:
            sample[var.name] = assignment1
        else:
            sample[var.name] = assignment2
    return sample


def pad(string: str, pad: int = 4) -> str:
    lines = string.split('\n')
    padded_lines = (' ' * pad + line for line in lines)
    return '\n'.join(padded_lines)


def print_conditional_probability(network: BayesianNetwork, conditionals_vars: dict[str, str],
                                  conditionals_evidents: dict[str, str]) -> None:
    print('Given')
    print(pad(pformat(conditionals_evidents)))
    print('conditional probability of')
    print(pad(pformat(conditionals_vars)))
    print("is {:f}".format(
        network.get_conditional_probability(
            conditionals_vars,
            conditionals_evidents
        )))
    print()


def print_joint_probability(network: BayesianNetwork, values: dict[str, str]) -> None:
    print('Joint probability of')
    print(pad(pformat(values)))
    print("is {:f}".format(network.get_joint_probability(values)))


def print_marginal_probabilities(network: BayesianNetwork) -> None:
    print("Marginal probabilities:")
    for variable in network.get_variables():
        print("    {}".format(variable.get_name()))
        for assignment in variable.get_assignments():
            print("        {}: {:f}".format(
                assignment,
                variable.get_marginal_probability(assignment))
            )


def tire():
    # the values kept as dictionary

    # False first!!

    # Damaged tire
    DT = {(): (0.7, 0.3)}
    # Electronics malfunctioning
    EM = {(): (0.7, 0.3)}
    # Fuel tank leaking
    FTL = {(): (0.8, 0.2)}
    # Slow max speed
    SMS = {
        ('true', 'true'): (0.95, 0.05),
        ('true', 'false'): (0.4, 0.6),
        ('false', 'true'): (0.7, 0.3),
        ('false', 'false'): (0.3, 0.7)
    }
    # High fuel consumption
    HC = {
        # DT ,   FTL,     EM
        ('true', 'true', 'true'): (0.1, 0.9),
        ('true', 'true', 'false'): (0.2, 0.8),
        ('true', 'false', 'true'): (0.7, 0.3),
        ('true', 'false', 'false'): (0.8, 0.2),
        ('false', 'true', 'true'): (0.4, 0.6),
        ('false', 'true', 'false'): (0.5, 0.5),
        ('false', 'false', 'true'): (0.9, 0.1),
        ('false', 'false', 'false'): (0.99, 0.01),
    }
    V = {('true',): (0.3, 0.7), ('false',): (0.9, 0.1)}
    # Vibrations

    # creation of Nodes objects
    Damaged_tire = Variable('DT', ('false', 'true'), DT)
    Electronic_malfunction = Variable('EM', ('false', 'true'), EM)
    Fuel_leak = Variable('FTL', ('false', 'true'), FTL)
    Vibrations = Variable('V', ('false', 'true'), V, [Damaged_tire])
    Slow_max_speed = Variable('SMS', ('false', 'true'), SMS, [Damaged_tire, Electronic_malfunction])
    High_consumption = Variable('HC', ('false', 'true'), HC, [Damaged_tire, Fuel_leak, Electronic_malfunction])

    variables = [Damaged_tire, Electronic_malfunction, Fuel_leak, Vibrations, Slow_max_speed, High_consumption]

    # creation of Network
    network = BayesianNetwork()
    network.set_variables(variables)

    # pre-calculate marginals
    # network.calculate_marginal_probabilities()

    print_marginal_probabilities(network)

    print('')

    joint_values = {
        'DT': 'true',
        'EM': 'false',
        'FTL': 'true',
        'V': 'false',
        'SMS': 'false',
        'HC': 'false'
    }
    print_joint_probability(network, joint_values)

    print('')
    # These have been retrieved from the description of the vehicle problems:
    conditionals_evidents = {
        'V': 'true',  # You have noticed that there are vibrations while driving,
        'SMS': 'true',  # and it is difficult to reach the top speed.
        'HC': 'false'  # Your car does not consume more than usual.
    }

    # conditionals_evidents = {
    #     'HC': 'true'  # Your car does not consume more than usual.
    # }

    print_conditional_probability(network, {'DT': 'true', }, conditionals_evidents)
    print_conditional_probability(network, {'EM': 'true', }, conditionals_evidents)
    print_conditional_probability(network, {'FTL': 'true', }, conditionals_evidents)


if __name__ == '__main__':
    # sprinkler()
    tire()
