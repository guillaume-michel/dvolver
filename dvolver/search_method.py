from enum import Enum

class SearchMethod(Enum):
    """
    type of search methods: random or genetic
    """
    RANDOM = 'random'
    GENETIC = 'genetic'

    def __str__(self):
        return self.value
