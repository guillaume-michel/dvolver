"""
Defines search space for NASNET-A as decribed in:
Learning Transferable Architectures for Scalable Image Recognition
https://arxiv.org/abs/1707.07012
"""

import random
import copy

search_space = {
    # possible operators used in a block
    'operations': ['separable_7x7_2',
                   'separable_5x5_2',
                   'separable_3x3_2',
                   'avg_pool_3x3',
                   'max_pool_3x3',
                   'none',
    ],
    # number of blocks per cell
    'B': 5,
}


def get_possible_hiddenstate_indices_for_block(search_space, block_index):
    # 2 because 0 and 1 are always possible
    # -1 because last hidden state cannot be an input
    return list(range(search_space['B']+2-1)[:block_index+2])


def get_possible_operations_for_block(search_space):
    return search_space['operations']


def is_hiddenstate_indices_param(search_space, index):
    B = search_space['B']

    if index < 2*2*B:
        return index % 2 == 0
    elif index > 2*2*B and index < 2*2*2*B+1:
        return (index-1) % 2 == 0
    else:
        return False


def is_extra_concat_param(search_space, index):
    B = search_space['B']
    normal_index = 2*2*B # 2 operators * 2 indices per block and there are B blocks in a cell
    reduction_index = normal_index + 2*2*B + 1
    return (index == normal_index) or (index == reduction_index)


def get_block_from_index(search_space, index):
    """
    return the block number in the cell based on the index in the parameters list
    """
    B = search_space['B']

    assert (index < 2*2*B) or (index > 2*2*B and index < 2*2*2*B+1)

    if index < 2*2*B:
        return index // 4
    else:
        return (index-4*B-1)//4


def get_possible_extra_concat(search_space, extra_concat):
    assert isinstance(extra_concat, list)

    B = search_space['B']

    possible_indices = list(range(B+1))

    results = [extra_concat]

    possible_new_connections = list(set(possible_indices)-set(extra_concat))

    # we can mutate the existing connections
    for i in range(len(extra_concat)):
        for j in possible_new_connections:
            l = copy.deepcopy(extra_concat)
            l[i] = j
            results += [sorted(l)]

    # we can add new connections
    for i in possible_new_connections:
        l = copy.deepcopy(extra_concat)
        l.append(i)
        results += [sorted(l)]

    # we can remove existing connections
    for i in range(len(extra_concat)):
        l = copy.deepcopy(extra_concat)
        l.remove(l[i])
        results += [sorted(l)]

    return results


def get_possible_params_for_index(search_space, current_value, index):

    if is_extra_concat_param(search_space, index):
        return get_possible_extra_concat(search_space, current_value)

    block_index = get_block_from_index(search_space, index)

    if is_hiddenstate_indices_param(search_space, index):
        return copy.deepcopy(get_possible_hiddenstate_indices_for_block(search_space, block_index))
    else:
        return copy.deepcopy(get_possible_operations_for_block(search_space))


# B = 5 # number of blocks per cell
# bi = block i
# h0 = hidden state on the left for the block
# h1 = hidden state on the right for the block
# op0 = left operation for the block
# op1 = right operation for the block
# extra_concat: list of additional hidden states indices to concat (possible indices: 0, 1,.., B)
# uh.i = 1 if hiddenstate i is used as input for a following block: i=0 is previous previous cell and i=1 is previous cell
# combination = always ADD
# individual representation: [# normal cell
#                             b0.h0, b0.op0, b0.h1, b0.op1,
#                             b1.h0, b1.op0, b1.h1, b1.op1,
#                             b2.h0, b2.op0, b2.h1, b2.op1,
#                             b3.h0, b3.op0, b3.h1, b3.op1,
#                             b4.h0, b4.op0, b4.h1, b4.op1,
#                             [extra_concat],
#                             # reduction cell
#                             b0.h0, b0.op0, b0.h1, b0.op1,
#                             b1.h0, b1.op0, b1.h1, b1.op1,
#                             b2.h0, b2.op0, b2.h1, b2.op1,
#                             b3.h0, b3.op0, b3.h1, b3.op1,
#                             b4.h0, b4.op0, b4.h1, b4.op1,
#                             [extra_concat],
#                            ]

def create_random_cell(search_space):
    genes = []

    B = search_space['B']

    # construct each block in the cell
    for i in range(B):
        genes += [
            # LEFT
            random.choice(get_possible_hiddenstate_indices_for_block(search_space, i)),
            random.choice(get_possible_operations_for_block(search_space)),
            # RIGHT
            random.choice(get_possible_hiddenstate_indices_for_block(search_space, i)),
            random.choice(get_possible_operations_for_block(search_space))]

    # construct empty extra concat list
    # we favor simple architectures and let mutation find interesting extra concat
    genes += [[]]

    return genes


def create_random_params(search_space):
    """
    Pick a random individual parameters among all possible individuals in the search space.
    returns: a list of parameters
    """
    normal_cell = create_random_cell(search_space)
    reduction_cell = create_random_cell(search_space)

    return normal_cell + reduction_cell


def create_random_individual(icls):
    """
    Create a random individual with class icls
    """
    return icls(create_random_params(search_space))


def parse_arch(arch):
    assert (len(arch)-2) % (2*4) == 0

    B = (len(arch)-2)//(2*4)

    for i in [2*2*B, 2*2*2*B+1]:
        if isinstance(arch[i], str):
            # the string should contains a python list
            arch[i] = eval(arch[i])
            assert isinstance(arch[i], list)

    return arch
