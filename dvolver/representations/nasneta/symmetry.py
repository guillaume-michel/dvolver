def normalize(ind):
    """
    Apply symmetries of representation to reduce ind to its normal form
    """
    assert isinstance(ind, tuple), 'ind should be a tuple in dvolver.nasneta.normalize'

    assert (len(ind)-2) % 8 == 0

    B = (len(ind)-2)//8

    normal_cell = ind[:2*2*B+1]
    reduction_cell = ind[2*2*B+1:]

    return normalize_cell(normal_cell) + normalize_cell(reduction_cell)


def get_block(ind, i):
    return ind[4*i:4*i+4]


def get_pair(block, i):
    return block[2*i:2*i+2]


def normalize_block(block):
    """
    Inside a block, we can exchange left and right because + is commutative
    """
    return sum(sorted(get_pair(block,i) for i in range(len(block)//2)),
               ())


def normalize_intra_blocks(ind):
    """
    normalize each block in the full representation, order of blocks is conserved
    """
    return sum((normalize_block(get_block(ind, i)) for i in range(len(ind)//4)),
               ())


def normalize_cell(cell):
    B = (len(cell)-1)//4
    regular_topology = cell[:2*2*B]
    extra_concat = cell[2*2*B:][0]

    return normalize_intra_blocks(regular_topology) + (tuple(sorted(extra_concat)),)
