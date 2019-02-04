from enum import Enum

class TrainMode(Enum):
    """
    train mode: search or full
    """
    SEARCH = 'search'
    FULL = 'full'

    def __str__(self):
        return self.value


def get_dir_name_for_train_mode(trainMode):

    if isinstance(trainMode, str) and trainMode in [str(e) for e in TrainMode]:
        trainMode = TrainMode(trainMode)

    if not isinstance(trainMode, TrainMode):
        raise ValueError('get_dir_name_for_train_mode should be given a TrainMode value')

    if trainMode == TrainMode.SEARCH:
        return 'samples'
    elif trainMode == TrainMode.FULL:
        return 'full'

    raise ValueError('Unknown train mode: ' + str(trainMode))
