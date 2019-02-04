from deap import base
from deap import creator

weights = (1.0, 1.0)

# maximize objective 1 and 2
creator.create("FitnessMax", base.Fitness, weights=weights)

# Individuals are lists
creator.create("Individual", list, fitness=creator.FitnessMax)

def individual_to_str(ind):
    name = '       '
    if hasattr(ind, 'archIndex'):
        name = str(ind.archIndex) + ': '

    return name + str(ind) + ' -> ' + str(ind.fitness.values)

def create_individual(params, values=None, archIndex=None):
    """
    Create individual from params and fitness values
    """
    individual = creator.Individual(params)

    if values != None:
        individual.fitness.values = values

    if archIndex != None:
        individual.archIndex = archIndex

    return individual
