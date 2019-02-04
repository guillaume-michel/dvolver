import random

import numpy as np

from .search_space import search_space as search_space
from .search_space import get_possible_params_for_index as get_possible_params_for_index
from .search_space import is_extra_concat_param as is_extra_concat_param

def get_list_of_possible_mutations(i, current_value, search_space):
    possible_choices = get_possible_params_for_index(search_space, current_value, i)
    possible_choices.remove(current_value)

    if len(possible_choices) == 0:
        # edge case: no other choice than the current one
        possible_choices.append(current_value)

    return possible_choices


def mutate(individual, indpb):
    for i in np.arange(len(individual)):
        if random.random() < indpb:
            individual[i] = random.choice(get_list_of_possible_mutations(i, individual[i], search_space))

    return individual,


def has_duplicate_values(l):
    return len(l) != len(set(l))


def perform_extra_concat_crossover(extra_concat1, extra_concat2):

    if extra_concat1 == extra_concat2:
        return extra_concat1, extra_concat2

    sizes = [len(extra_concat1), len(extra_concat2)]
    size1 = random.choice(sizes)
    sizes.remove(size1)
    size2 = sizes[0]

    new_extra_concat1 = []
    new_extra_concat2 = []

    possible_connections = extra_concat1 + extra_concat2

    for i in range(size1):
        c = random.choice(possible_connections)
        # make sure c is not already in new_extra_concat1
        while c in new_extra_concat1:
            c = random.choice(possible_connections)

        new_extra_concat1.append(c)
        possible_connections.remove(c)

    new_extra_concat1 = sorted(new_extra_concat1)

    # make sure the number of remaining connections match the expected value of size2
    assert len(possible_connections) == size2
    new_extra_concat2 = sorted(possible_connections)

    if has_duplicate_values(new_extra_concat1) or has_duplicate_values(new_extra_concat2):
        # restart
        return perform_extra_concat_crossover(extra_concat1, extra_concat2)

    if extra_concat1 != new_extra_concat1:
        # crossover did something -> good
        return new_extra_concat1, new_extra_concat2
    else:
        # retry until there is a crossover
        return perform_extra_concat_crossover(extra_concat1, extra_concat2)


def mate(ind1, ind2, indpb):
    """Executes a uniform crossover that modify in place the two
    :term:`sequence` individuals. The attributes are swapped accordingto the
    *indpb* probability.
    for extra_concat special crossover is applied:


    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :param indpb: Independent probabily for each attribute to be exchanged.
    :returns: A tuple of two individuals.
    This function uses the :func:`~random.random` function from the python base
    :mod:`random` module.
    """
    assert len(ind1) == len(ind2)

    for i in range(len(ind1)):
        if random.random() < indpb:
            if not is_extra_concat_param(search_space, i):
                # regular uniform crossover
                ind1[i], ind2[i] = ind2[i], ind1[i]
            else:
                # special crossover for extra_concat
                ind1[i], ind2[i] = perform_extra_concat_crossover(ind1[i], ind2[i])

    return ind1, ind2
