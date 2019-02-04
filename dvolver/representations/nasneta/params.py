import argparse

from dvolver import *

def add_argument(parser, trainMode):

    parser.add_argument('--model-type', type=str, default='cifar', help="Type of NASNet-A model.")

    args, _ = parser.parse_known_args()

    if args.model_type == 'cifar':
        if trainMode == TrainMode.SEARCH:
            num_cells = 6
            num_epochs_per_decay = 24
            learning_rate_decay_type = 'triangular2'
        elif trainMode == TrainMode.FULL:
            num_cells = 18
            num_epochs_per_decay = 600
            learning_rate_decay_type = 'cosine'
        else:
            raise ValueError('Invalid trainMode: '+str(trainMode))

        num_conv_filters = 32
        stem_multiplier = 3.0
        drop_path_keep_prob = 0.6
        use_aux_head = 1
        dense_dropout_keep_prob = 1.0
        filter_scaling_rate = 2.0
        num_reduction_layers = 2
        skip_reduction_layer_input = 0

        label_smoothing = 0
        aux_weight = 0.4
        clip_gradient_norm = 5.0

        triangular_min_learning_rate = 1e-1
        triangular_max_learning_rate = 1
        final_min_learning_rate = 1e-6
        triangular_min_momentum = 0.85
        triangular_max_momentum = 0.95
        learning_rate_decay_factor = 1

        optimizer = 'momentum'
        rmsprop_decay = 0
        epsilon = 1.0
        moving_average_decay = None

    elif args.model_type == 'mobile':
        if trainMode == TrainMode.SEARCH:
            raise ValueError('You should not use search mode with model type mobile. Think again!')
        elif trainMode == TrainMode.FULL:
            num_cells = 12
            num_epochs_per_decay = 2
            learning_rate_decay_type = 'exponential'
        else:
            raise ValueError('Invalid trainMode: '+str(trainMode))

        num_conv_filters = 44
        stem_multiplier = 1.0
        drop_path_keep_prob = 1.0
        use_aux_head = 1
        dense_dropout_keep_prob = 0.5
        filter_scaling_rate = 2.0
        num_reduction_layers = 2
        skip_reduction_layer_input = 0

        label_smoothing = 0.1
        aux_weight = 0.4

        clip_gradient_norm = 10.0

        triangular_min_learning_rate = 1e-1
        triangular_max_learning_rate = 1
        final_min_learning_rate = 1e-6
        triangular_min_momentum = 0.85
        triangular_max_momentum = 0.95

        learning_rate_decay_factor = 0.97

        optimizer = 'rmsprop'
        rmsprop_decay = 0.9
        epsilon = 1.0
        moving_average_decay = 0.9999

    elif args.model_type == 'large':
        if trainMode == TrainMode.SEARCH:
            raise ValueError('You should not use search mode with model type large. Think again!')
        elif trainMode == TrainMode.FULL:
            num_cells = 18
            num_epochs_per_decay = 2
            learning_rate_decay_type = 'exponential'
        else:
            raise ValueError('Invalid trainMode: '+str(trainMode))

        num_conv_filters = 168
        stem_multiplier = 3.0
        drop_path_keep_prob = 0.7
        use_aux_head = 1
        dense_dropout_keep_prob = 0.5
        filter_scaling_rate = 2.0
        num_reduction_layers = 2
        skip_reduction_layer_input = 1

        label_smoothing = 0.1
        aux_weight = 0.4

        clip_gradient_norm = 2.0

        triangular_min_learning_rate = 1e-1
        triangular_max_learning_rate = 1
        final_min_learning_rate = 1e-6
        triangular_min_momentum = 0.85
        triangular_max_momentum = 0.95

        learning_rate_decay_factor = 0.94

        optimizer = 'rmsprop'
        rmsprop_decay = 0.9
        epsilon = 1.0
        moving_average_decay = 0.9999

    else:
        raise ValueError('Unsupported model type: ' + args.model_type)

    ##############################################
    parser.add_argument('--num-cells', type=int, default=num_cells, help='NASNet-A num of normal cells (parameter N in paper notation N@C).')
    parser.add_argument('--num-epochs-per-decay', type=int, default=num_epochs_per_decay, help='Number of epoches for one period of cosine decay')
    parser.add_argument('--learning-rate-decay-type', type=str, default=learning_rate_decay_type, help="The decay type for learning rate.")

    parser.add_argument('--num-conv-filters', type=int, default=num_conv_filters, help='NASNet-A num conv filters (parameter C/24 in paper notation N@C).')
    parser.add_argument('--stem-multiplier', type=float, default=stem_multiplier, help='NASNet-A stem multiply factor')
    parser.add_argument('--drop-path-keep-prob', type=float, default=drop_path_keep_prob, help='NASNet-A drop path keep probability')
    parser.add_argument('--use-aux-head', type=int, default=use_aux_head, help='NASNet-A use aux loss')
    parser.add_argument('--dense-dropout-keep-prob', type=float, default=dense_dropout_keep_prob, help='NASNet-A dense dropout keep probability')
    parser.add_argument('--filter-scaling-rate', type=float, default=filter_scaling_rate, help='NASNet-A filter scaling rate')
    parser.add_argument('--num-reduction-layers', type=int, default=num_reduction_layers, help='NASNet-A num of reduction cells.')
    parser.add_argument('--skip-reduction-layer-input', type=int, default=skip_reduction_layer_input, help='NASNet-A skip reduction cell as input')

    parser.add_argument('--label-smoothing', type=float, default=label_smoothing, help="Label smoothing value.")
    parser.add_argument('--aux-weight', type=float, default=aux_weight, help="Auxiliary loss weight")
    parser.add_argument('--clip-gradient-norm', type=float, default=clip_gradient_norm, help="Gradient norm clipping value")

    parser.add_argument('--triangular-min-learning-rate',
                        type=float, default=triangular_min_learning_rate, help='Minimal learning rate value for triangular scheduling')
    parser.add_argument('--triangular-max-learning-rate',
                        type=float, default=triangular_max_learning_rate, help='Maximal learning rate value for triangular scheduling')
    parser.add_argument('--final-min-learning-rate',
                        type=float, default=final_min_learning_rate, help='Final Minimal learning rate value after triangular scheduling')
    parser.add_argument('--triangular-min-momentum',
                        type=float, default=triangular_min_momentum, help='Minimal momentum value for triangular scheduling')
    parser.add_argument('--triangular-max-momentum',
                        type=float, default=triangular_max_momentum, help='Maximal momentum value for triangular scheduling')
    parser.add_argument('--learning-rate-decay-factor',
                        type=float, default=learning_rate_decay_factor, help='Learning rate decay factor for exponential decay')

    parser.add_argument('--optimizer',
                        type=str, default=optimizer, help='The name of the optimizer')
    parser.add_argument('--rmsprop-decay',
                        type=float, default=rmsprop_decay, help='Decay term for RMSProp')
    parser.add_argument('--epsilon',
                        type=float, default=epsilon, help='Epsilon term for the optimizer')
    parser.add_argument('--moving-average-decay', default=moving_average_decay, help='The decay to use for the moving average. If left as None, then moving averages are not used.')


def add_worker_args(args, worker_args):
    extra_args = {
        'num_cells': args.num_cells,
        'num_conv_filters': args.num_conv_filters,
        'num_epochs_per_decay': args.num_epochs_per_decay,
        'stem_multiplier': args.stem_multiplier,
        'drop_path_keep_prob': args.drop_path_keep_prob,
        'use_aux_head': args.use_aux_head,
        'dense_dropout_keep_prob': args.dense_dropout_keep_prob,
        'filter_scaling_rate': args.filter_scaling_rate,
        'num_reduction_layers': args.num_reduction_layers,
        'skip_reduction_layer_input': args.skip_reduction_layer_input,
        'label_smoothing': args.label_smoothing,
        'aux_weight': args.aux_weight,
        'clip_gradient_norm': args.clip_gradient_norm,
        'learning_rate_decay_type': args.learning_rate_decay_type,
        'triangular_min_learning_rate': args.triangular_min_learning_rate,
        'triangular_max_learning_rate': args.triangular_max_learning_rate,
        'final_min_learning_rate': args.final_min_learning_rate,
        'triangular_min_momentum': args.triangular_min_momentum,
        'triangular_max_momentum': args.triangular_max_momentum,
        'learning_rate_decay_factor': args.learning_rate_decay_factor,
        'optimizer': args.optimizer,
        'rmsprop_decay': args.rmsprop_decay,
        'epsilon': args.epsilon,
        'moving_average_decay': args.moving_average_decay,
        'model_type': args.model_type,
    }

    return dict(worker_args, **extra_args)
