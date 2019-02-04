#!/usr/bin/python3
"""
Use multi objectives genetic algoritm to search for Pareto optimal neural network architectures
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
import numpy as np

import time
import random
import logging
import threading
import json
import importlib
import itertools

from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import hypervolume
from deap import creator
from deap import tools

from collections import namedtuple

from tqdm import tqdm
tqdm.monitor_interval = 0  # see https://github.com/tqdm/tqdm/issues/481

import pika
import uuid

from dvolver import *

Metrics = namedtuple('Metrics', 'cache_hit, pending_cache_hit, cache_miss')

def computeMetricsPercents(metrics):
    cache_hit = metrics.cache_hit / (metrics.cache_hit+metrics.cache_miss+metrics.pending_cache_hit)*100
    pending_cache_hit = metrics.pending_cache_hit / (metrics.cache_hit+metrics.cache_miss+metrics.pending_cache_hit)*100
    cache_miss = metrics.cache_miss / (metrics.cache_hit+metrics.cache_miss+metrics.pending_cache_hit)*100
    total_cache_hit = (metrics.cache_hit+metrics.pending_cache_hit) / (metrics.cache_hit+metrics.cache_miss+metrics.pending_cache_hit)*100

    return (cache_hit, pending_cache_hit, cache_miss, total_cache_hit)


def printMetrics(metrics):
    cache_hit, pending_cache_hit, cache_miss, total_cache_hit = computeMetricsPercents(metrics)

    print('cache hit        :', metrics.cache_hit, '(', cache_hit, '%)')
    print('pending cache hit:', metrics.pending_cache_hit, '(', pending_cache_hit, '%)')
    print('cache miss       :', metrics.cache_miss, '(', cache_miss, '%)')
    print('total cache hit  :', metrics.cache_hit + metrics.pending_cache_hit, '(', total_cache_hit, '%)')


def writeMetricsInTB(writer, metrics, step):
    cache_hit, pending_cache_hit, cache_miss, total_cache_hit = computeMetricsPercents(metrics)

    writer.add_scalar('metrics/cache_hit', cache_hit, step)
    writer.add_scalar('metrics/pending_cache_hit', pending_cache_hit, step)
    writer.add_scalar('metrics/cache_miss', cache_miss, step)
    writer.add_scalar('metrics/total_cache_hit', total_cache_hit, step)


def normalize_cache(cache):
    normalized_cache = {representation.normalize(k):v for k,v in cache.items()}

    assert len(normalized_cache) == len(cache), 'Cache cannot be automatically normalized: dupplicated entries detected'

    return normalized_cache


def parse(msg):
    """
    example of representation (json format):
    {
        "individual": [1, 2, 3],
        "fitness": [1, 4, 9]
    }
    """
    return json.loads(msg.decode('utf-8'))


def get_integers_gen(start=0):
    """
    Returns a function that will output the next integer on each call
    The returned function is safe to be called from multiple threads
    """
    lock = threading.Lock()

    def generator():
        i = start

        while True:
            yield i
            i += 1

    g = generator()

    def gen():
        with lock:
            return next(g)

    return gen


def get_brief_base_path(path):
    """
    Returns the base path for brief
    """
    return path + '/brief'


def get_pareto_image(front, reference_inds, gen):
    front_array = np.array([ind.fitness.values for ind in front])

    ref_plt_attr = itertools.cycle(itertools.product(['r', 'g', 'c', 'm', 'y', 'k'], ['+', 'x', '^', 'v', 's', '*']))

    # all = front + reference_inds
    # min_accuracy = sorted(all, key=lambda x: x.fitness.values[1], reverse=True)[-1].fitness.values[1]
    # max_accuracy = sorted(all, key=lambda x: x.fitness.values[1], reverse=True)[0].fitness.values[1]
    # min_speed = sorted(all, key=lambda x: x.fitness.values[1], reverse=True)[-1].fitness.values[0]
    # max_speed = sorted(all, key=lambda x: x.fitness.values[1], reverse=True)[0].fitness.values[0]

    fig, ax = plt.subplots()
    for ref_ind in reference_inds:
        ref_array = np.array([ref_ind.fitness.values])
        color, marker = next(ref_plt_attr)
        ax.scatter(ref_array[:,0], ref_array[:,1], label=str(ref_ind.archIndex), marker=marker, color=color)

    ax.scatter(front_array[:,0], front_array[:,1], label=str(gen), color='b')
    ax.axis("tight")
    plt.xlabel('Speed')
    plt.ylabel('Accuracy')
    # plt.xlim(xmin=min_speed*0.9, xmax=max_speed*1.1)
    # plt.ylim(ymin=min_accuracy*0.9, ymax=max_accuracy*1.1)
    plt.grid(True)
    legend = ax.legend(loc='upper right', shadow=False)

    im = fig2data(fig)

    plt.close(fig)

    return im


def samples_count(gen, n):
    return (gen+1)*n


def getEvaluateIndividuals(cache,
                           channel,
                           request_queue,
                           callback_queue,
                           worker_args,
                           start=0):

    archIndex_gen = get_integers_gen(start)

    lock = threading.Lock() # for protecting cache

    def evaluateIndividuals(individuals, gen):

        results_inds = []

        pending_inds = []
        delayed_inds = []
        pending_cache = set()

        cache_hit = 0
        cache_miss = 0
        pending_cache_hit = 0

        for ind in individuals:

            if not hasattr(ind, 'archIndex'):
                ind.archIndex = archIndex_gen()

            lastArchIndex = ind.archIndex

            with lock:
                if representation.normalize(tuple(ind)) in cache:
                    # cache hit
                    ind.fitness.values = cache[representation.normalize(tuple(ind))]
                    results_inds.append(ind)
                    cache_hit += 1
                elif representation.normalize(tuple(ind)) in pending_cache:
                    # pending cache hit
                    # this ind has already been queued for evaluation
                    # don't duplicate the work, we wait for the result to come back and use it
                    delayed_inds.append(ind)
                    pending_cache_hit += 1
                else:
                    # cache miss
                    cache_miss += 1
                    pending_inds.append(ind)
                    pending_cache.add(representation.normalize(tuple(ind)))
                    request = {'individual':ind, 'archIndex':ind.archIndex, 'gen': gen, 'worker_args':worker_args}
                    corr_id = str(uuid.uuid4())
                    channel.basic_publish(exchange='',
                                          routing_key=request_queue,
                                          properties=pika.BasicProperties(
                                              reply_to = callback_queue,
                                              correlation_id = corr_id,
                                              content_type = 'application/json',
                                          ),
                                          body=json.dumps(request))

        pbar = tqdm(total = len(pending_inds))

        while len(results_inds) < len(individuals):
            method_frame, header_frame, body = channel.basic_get(queue = callback_queue)

            if not method_frame:
                time.sleep(1)
                continue
            else:
                #print(method_frame, header_frame, body)

                response = parse(body)

                # handle response here
                #print('response =', response)

                ind = creator.Individual(response['individual'])
                ind.archIndex = response['archIndex']

                # update cache
                with lock:
                    if representation.normalize(tuple(ind)) not in cache:
                        # update cache
                        cache[representation.normalize(tuple(ind))] = response['fitness']

                    ind.fitness.values = cache[representation.normalize(tuple(ind))]

                results_inds.append(ind)

                # now that response is handle we can safely acknowledge
                channel.basic_ack(delivery_tag=method_frame.delivery_tag)

                pbar.update(1)

                # check if we can process some delayed_inds
                with lock:
                    for delayed_ind in delayed_inds:
                        if representation.normalize(tuple(delayed_ind)) in cache:
                            # we finally got an evaluation for this delayed_ind, add it to the final result
                            delayed_ind.fitness.values = cache[representation.normalize(tuple(delayed_ind))]
                            results_inds.append(delayed_ind)

                    # update list of delayed_inds, to keep only the one that did not get a result yet
                    delayed_inds = [delayed_ind for delayed_ind in delayed_inds if not representation.normalize(tuple(delayed_ind)) in cache]

        if len(results_inds) != len(individuals):
            raise ValueError('missing responses')

        assert len(delayed_inds) == 0, 'some delayed_inds did not get a results. Something went wrong...'
        assert len(individuals) == cache_hit+cache_miss+pending_cache_hit, 'Wrong classification of cache hit/miss. Something is wrong...'

        print('')

        for ind in results_inds:
            print(individual_to_str(ind))

        metrics = Metrics(cache_hit, pending_cache_hit, cache_miss)

        printMetrics(metrics)

        return results_inds, lastArchIndex, metrics

    return evaluateIndividuals


def search(search_method,
           n,
           ngen,
           ref_point,
           cxpb,
           cxindpb,
           mutindpb,
           log_dir,
           channel,
           request_queue,
           callback_queue,
           worker_args,
           reference_file,
           initial_cache_file,
           verbose=False):

    if not isinstance(search_method, SearchMethod):
        raise ValueError('search_method should be: ' + str(', '.join(map(str, list(SearchMethod)))))

    reference_inds = read_reference_file(representation, reference_file)

    if reference_inds:
        print("Loaded reference file:", reference_file)

    toolbox = base.Toolbox()

    toolbox.register("individual", representation.create_random_individual, creator.Individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    if search_method == SearchMethod.GENETIC:
        toolbox.register("mate", representation.mate, indpb=cxindpb)
        toolbox.register("mutate", representation.mutate, indpb=mutindpb)

    toolbox.register("select", tools.selNSGA2)

    brief_base_path = get_brief_base_path(log_dir)
    pop_writer = SummaryWriter(brief_base_path + '/population')
    hof_writer = SummaryWriter(brief_base_path + '/hall_of_fame')

    external_cache = read_external_cache_file(initial_cache_file)

    checkpoint_base_path = get_checkpoint_base_path(log_dir)
    latest_checkpoint = get_latest_checkpoint(checkpoint_base_path)

    if latest_checkpoint:
        # A checkpoint file has been found, then load the data from the file
        print('Restoring search from checkpoint:', latest_checkpoint[1], ' ...')
        cp = restore_checkpoint(latest_checkpoint[1])

        if search_method != cp['search_method']:
            raise ValueError('Cannot mix search_method in the same directory')

        random.setstate(cp['rndstate'])
        startArchIndex = cp['lastArchIndex'] + 1
        current_gen = cp['generation']
        start_gen = current_gen + 1

        history = cp['history']

        if search_method == SearchMethod.GENETIC:
            # Decorate the variation operators
            toolbox.decorate("mate", history.decorator)
            toolbox.decorate("mutate", history.decorator)

        samples = cp['samples']
        hypervolumes = cp['hypervolumes']
        hof_hypervolumes = cp['hof_hypervolumes']

        pop = cp['population']

        if len(pop) != cp['population_count']:
            raise ValueError('Cannot change population count for restored search')

        hof = cp['hof']

        # This is just to assign the crowding distance to the individuals
        # no actual selection is done
        pop = toolbox.select(pop, len(pop))

        # Restore cache from checkpoint
        cache = cp['cache']
        print('Restored', len(cache), 'entries in cache')

        if len(external_cache) > 0:
            print('Merging', len(external_cache), 'entries from external cache in actual cache with', len(cache), 'entries in it.')
            cache = merge_caches(external_cache, cache)
            print('cache has now', len(cache), 'entries.')

        print('Normalizing cache...')
        start = time.time()
        cache = normalize_cache(cache)
        end = time.time()
        print('Cache normalization completed in:', end-start, 's')

        evaluateIndividuals = getEvaluateIndividuals(cache=cache,
                                                     channel=channel,
                                                     request_queue=request_queue,
                                                     callback_queue=callback_queue,
                                                     worker_args=worker_args,
                                                     start=startArchIndex)

        # Evaluate reference points
        if reference_inds:
            print('='*80)
            print("Starting evaluation for reference points:")
            print('='*80)
            reference_inds, _, _ = evaluateIndividuals(reference_inds, 'references')

            hof_writer.add_text('References sorted by accuracy',
                                representation.generate_markdown_table(sorted(reference_inds, key=lambda x: x.fitness.values[1], reverse=True)),
                                0)

    else:
        # start a new evolution
        print('Starting new %s search...'%(search_method))
        startArchIndex = 0
        current_gen = 0
        start_gen = 1

        history = tools.History()

        if search_method == SearchMethod.GENETIC:
            # Decorate the variation operators
            toolbox.decorate("mate", history.decorator)
            toolbox.decorate("mutate", history.decorator)

        samples = []
        hypervolumes = []
        hof_hypervolumes = []

        hof = tools.ParetoFront()

        cache = {}

        if len(external_cache) > 0:
            print('Merging', len(external_cache), 'entries from external cache in actual cache with', len(cache), 'entries in it.')
            cache = merge_caches(external_cache, cache)
            print('cache has now', len(cache), 'entries.')

        evaluateIndividuals = getEvaluateIndividuals(cache=cache,
                                                     channel=channel,
                                                     request_queue=request_queue,
                                                     callback_queue=callback_queue,
                                                     worker_args=worker_args,
                                                     start=startArchIndex)

        # Evaluate reference points
        if reference_inds:
            print('='*80)
            print("Starting evaluation for reference points:")
            print('='*80)
            reference_inds, _, _ = evaluateIndividuals(reference_inds, 'references')

            hof_writer.add_text('References sorted by accuracy',
                                representation.generate_markdown_table(sorted(reference_inds, key=lambda x: x.fitness.values[1], reverse=True)),
                                0)


        # Evaluate the initial individuals
        print('='*80)
        print("Starting evaluation for generation: %d/%d"%(current_gen, ngen-1))
        print('='*80)
        pop, lastArchIndex, metrics = evaluateIndividuals(toolbox.population(n=n), current_gen)

        samples.append(samples_count(current_gen, n))

        # This is just to assign the crowding distance to the individuals
        # no actual selection is done
        pop = toolbox.select(pop, len(pop))

        hof.update(pop)
        history.update(pop)

        hypervolumes.append(hypervolume(pop, ref_point))
        hof_hypervolumes.append(hypervolume(hof, ref_point))

        # SUMMARIES
        # Scalar: Hypervolume
        pop_writer.add_scalar('hypervolume', hypervolumes[-1], samples[-1])
        hof_writer.add_scalar('hypervolume', hof_hypervolumes[-1], samples[-1])

        # Scalar: cxindpb
        pop_writer.add_scalar('pb/cxindpb', cxindpb, samples[-1])
        hof_writer.add_scalar('pb/cxindpb', cxindpb, samples[-1])

        # Scalar: mutindpb
        pop_writer.add_scalar('pb/mutindpb', mutindpb, samples[-1])
        hof_writer.add_scalar('pb/mutindpb', mutindpb, samples[-1])

        # Scalar: train_batch_size
        pop_writer.add_scalar('batch_size/train', worker_args['train_batch_size'], samples[-1])
        hof_writer.add_scalar('batch_size/train', worker_args['train_batch_size'], samples[-1])

        # Scalar: eval_batch_size
        pop_writer.add_scalar('batch_size/eval', worker_args['eval_batch_size'], samples[-1])
        hof_writer.add_scalar('batch_size/eval', worker_args['eval_batch_size'], samples[-1])

        # Scalar: metrics
        writeMetricsInTB(pop_writer, metrics, samples[-1])
        writeMetricsInTB(hof_writer, metrics, samples[-1])

        # Text: Paretor front
        pop_pareto_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]

        pop_writer.add_text('Pareto front sorted by accuracy',
                            representation.generate_markdown_table(sorted(pop_pareto_front, key=lambda x: x.fitness.values[1], reverse=True)),
                            samples[-1])
        hof_writer.add_text('Pareto front sorted by accuracy',
                            representation.generate_markdown_table(sorted(hof, key=lambda x: x.fitness.values[1], reverse=True)),
                            samples[-1])

        # Image: Pareto front
        pop_writer.add_image('Pareto_front', get_pareto_image(pop, reference_inds, current_gen), samples[-1])
        hof_writer.add_image('Pareto_front', get_pareto_image(hof, reference_inds, current_gen), samples[-1])

        # save checkpoint for gen = 0
        save_checkpoint(checkpoint_base_path + '/' + checkpoint_name(current_gen),
                        search_method=search_method,
                        rndstate=random.getstate(),
                        lastArchIndex = lastArchIndex,
                        generation=current_gen,
                        history=history,
                        samples=samples,
                        hypervolumes=hypervolumes,
                        hof_hypervolumes=hof_hypervolumes,
                        logbook=None,
                        population=pop,
                        hof=hof,
                        cache=cache)

    # Begin the generational process
    for gen in range(start_gen, ngen):
        # Vary the population
        if search_method == SearchMethod.GENETIC:
            offspring = tools.selTournamentDCD(pop, len(pop))
            offspring = [toolbox.clone(ind) for ind in offspring]

            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if random.random() <= cxpb:
                    toolbox.mate(ind1, ind2)

                toolbox.mutate(ind1)
                toolbox.mutate(ind2)
                del ind1.fitness.values, ind2.fitness.values
                if hasattr(ind1, 'archIndex'):
                    del ind1.archIndex
                if hasattr(ind2, 'archIndex'):
                    del ind2.archIndex


        elif search_method == SearchMethod.RANDOM:
            offspring = toolbox.population(n=n)

            for ind in offspring:
                del ind.fitness.values
        else:
            raise ValueError('Use of undefined SearchMethod: ' + str(search_method))

        print('='*80)
        print("Starting evaluation for generation: %d/%d"%(gen, ngen-1))
        print('='*80)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        valid_ind = [ind for ind in offspring if ind.fitness.valid]

        invalid_ind, lastArchIndex, metrics = evaluateIndividuals(invalid_ind, gen)

        offspring = valid_ind + invalid_ind

        samples.append(samples_count(gen, n))

        # Update the hall of fame with the generated individuals
        hof.update(offspring)

        # Select the next generation population
        pop = toolbox.select(pop + offspring, n)

        hypervolumes.append(hypervolume(pop, ref_point))
        hof_hypervolumes.append(hypervolume(hof, ref_point))

        # SUMMARIES
        # Scalar: Hypervolume
        pop_writer.add_scalar('hypervolume', hypervolumes[-1], samples[-1])
        hof_writer.add_scalar('hypervolume', hof_hypervolumes[-1], samples[-1])

        # Scalar: cxindpb
        pop_writer.add_scalar('pb/cxindpb', cxindpb, samples[-1])
        hof_writer.add_scalar('pb/cxindpb', cxindpb, samples[-1])

        # Scalar: mutindpb
        pop_writer.add_scalar('pb/mutindpb', mutindpb, samples[-1])
        hof_writer.add_scalar('pb/mutindpb', mutindpb, samples[-1])

        # Scalar: train_batch_size
        pop_writer.add_scalar('batch_size/train', worker_args['train_batch_size'], samples[-1])
        hof_writer.add_scalar('batch_size/train', worker_args['train_batch_size'], samples[-1])

        # Scalar: eval_batch_size
        pop_writer.add_scalar('batch_size/eval', worker_args['eval_batch_size'], samples[-1])
        hof_writer.add_scalar('batch_size/eval', worker_args['eval_batch_size'], samples[-1])

        # Scalar: metrics
        writeMetricsInTB(pop_writer, metrics, samples[-1])
        writeMetricsInTB(hof_writer, metrics, samples[-1])

        # Text: Paretor front
        pop_pareto_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]

        pop_writer.add_text('Pareto front sorted by accuracy',
                            representation.generate_markdown_table(sorted(pop_pareto_front, key=lambda x: x.fitness.values[1], reverse=True)),
                            samples[-1])
        hof_writer.add_text('Pareto front sorted by accuracy',
                            representation.generate_markdown_table(sorted(hof, key=lambda x: x.fitness.values[1], reverse=True)),
                            samples[-1])

        # Image: Pareto front
        pop_writer.add_image('Pareto_front', get_pareto_image(pop, reference_inds, gen), samples[-1])
        hof_writer.add_image('Pareto_front', get_pareto_image(hof, reference_inds, gen), samples[-1])

        # save checkpoint for gen
        save_checkpoint(checkpoint_base_path + '/' + checkpoint_name(gen),
                        search_method=search_method,
                        rndstate=random.getstate(),
                        lastArchIndex = lastArchIndex,
                        generation=gen,
                        history=history,
                        samples=samples,
                        hypervolumes=hypervolumes,
                        hof_hypervolumes=hof_hypervolumes,
                        logbook=None,
                        population=pop,
                        hof=hof,
                        cache=cache)

        print("hypervolume is %f" % hypervolumes[-1])

    print(len(cache), 'entries in cache')


def main(args):

    connection = pika.BlockingConnection(pika.ConnectionParameters(
        host=args.broker))

    channel = connection.channel()

    channel.queue_declare(queue=args.request_queue)
    #result = channel.queue_declare(queue=args.response_queue, exclusive=True)
    result = channel.queue_declare(queue=args.response_queue)
    callback_queue = result.method.queue

    representation_name = args.representation
    search_method = args.search_method
    n = args.population_size
    ngen = args.max_generations
    ref_point = [0.0, 0.0]
    cxpb = args.cxpb
    cxindpb = args.cxindpb
    mutindpb = args.mutindpb
    verbose = args.verbose
    log_dir = args.job_dir
    reference_file = args.reference_file
    initial_cache_file = args.initial_cache_file

    worker_args = representation.add_worker_args(args, {
        'data_dir': args.data_dir,
        'job_dir': args.job_dir,
        'nb_classes': input_pipeline.NB_CLASSES,
        'data_format': args.data_format,
        'max_steps': args.max_steps,
        'train_batch_size': args.train_batch_size,
        'eval_batch_size': args.eval_batch_size,
        'learning_rate': args.learning_rate,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'enable_cutout': (not args.disable_cutout),
        'log_device_placement': args.log_device_placement,
        'log_step_count_steps': args.log_step_count_steps,
        'save_checkpoints_steps': args.save_checkpoints_steps,
        'save_summary_steps': args.save_summary_steps,
        'keep_checkpoint_every_n_hours' : args.keep_checkpoint_every_n_hours,
        'keep_checkpoint_max' : args.keep_checkpoint_max,
        'save_checkpoints_secs': args.save_checkpoints_secs,
        'preproc_threads': args.preproc_threads,
        'representation_name': 'dvolver.representations.' + args.representation,
        'train_mode': str(args.train_mode),
        'throttle_secs': args.throttle_secs,
        'input_pipeline_name': 'input_pipeline.' + args.input_pipeline
    })

    try:
        search(search_method=search_method,
               n=n,
               ngen=ngen,
               ref_point=ref_point,
               cxpb=cxpb,
               cxindpb=cxindpb,
               mutindpb=mutindpb,
               log_dir=log_dir,
               channel=channel,
               request_queue=args.request_queue,
               callback_queue=callback_queue,
               worker_args=worker_args,
               reference_file=reference_file,
               initial_cache_file=initial_cache_file,
               verbose=verbose)
    except KeyboardInterrupt:
        print('Keyboard Interrupt called. Wait until it properly stops...')
    finally:
        print('closing connection...')
        connection.close()
        print('connection closed')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--broker', type=str, default='rabbitmq', help='RabbitMQ broker')
    parser.add_argument('--request-queue', type=str, default='dvolver_request', help='Request queue name (from server -> workers)')
    parser.add_argument('--response-queue', type=str, default='dvolver_response', help='response queue (from workers -> server)')

    parser.add_argument('--initial-cache-file', type=str, default=None, help='initial cache file to seed search cache')

    parser.add_argument('--population-size', type=int, default=32, help='Population size')
    parser.add_argument('--max-generations', type=int, default=1000, help='Maximum number of generations')
    parser.add_argument('--cxpb', type=float, default=0.9, help='crossover probability')
    parser.add_argument('--cxindpb', type=float, default=1/10, help='individual component\'s crossover probability')
    parser.add_argument('--mutindpb', type=float, default=1/10, help='individual component\'s mutation probability')

    parser.add_argument('--verbose', action='store_true', default=False, help='Enable verbose mode.')

    parser.add_argument('--search_method', type=SearchMethod, choices=list(SearchMethod), required=True, help='Search method to use')
    parser.add_argument('--train-mode', type=TrainMode, choices=list(TrainMode), required=True, help='Train mode to use')
    parser.add_argument('--data-dir', type=str, required=True, help='base directory where CIFAR-10 tfrecords are.')
    parser.add_argument('--job-dir', type=str, required=True, help='The directory where the models will be stored.')
    parser.add_argument('--data-format', type=str, default='channels_first', help='Image format to use.')
    parser.add_argument('--max-steps', type=int, default=21600, help='The number of steps to use for training.')
    parser.add_argument('--train-batch-size', type=int, default=150, help='Batch size for training.')
    parser.add_argument('--eval-batch-size', type=int, default=125, help='Batch size for validation.')
    parser.add_argument('--learning-rate', type=float, default=0.025, help="This is the inital learning rate value.")
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for MomentumOptimizer.')
    parser.add_argument('--weight-decay', type=float, default=4e-5, help='Weight decay for convolutions.')
    parser.add_argument('--disable-cutout', action='store_true', default=False, help='Whether to disable cutout in data augmentation.')
    parser.add_argument('--log-device-placement', action='store_true', default=False, help='Whether to log device placement.')
    parser.add_argument('--log-step-count-steps', type=int, default=1000, help='The number of steps to wait between each logs.')
    parser.add_argument('--save-checkpoints-steps', type=int, default=21600, help='The number of steps between each checkpoint and evaluation.')
    parser.add_argument('--save-checkpoints-secs', type=int, default=None, help='The number of secs between each checkpoint and evaluation.')
    parser.add_argument('--save-summary-steps', type=int, default=18750, help='The number of steps between each summary.')
    parser.add_argument('--keep-checkpoint-every-n-hours', type=float, default=1., help='frequency of kept checkpoints (not deleted by checkpoint_keep_max)')
    parser.add_argument('--keep-checkpoint-max', type=int, default=5, help="maximum number of checkpoints to keep")
    parser.add_argument('--throttle_secs', type=int, default=36000, help='Minimal duration between sucessive evaluations in seconds')
    parser.add_argument('--preproc-threads', type=int, default=4, help='The number of dedicated threads for preprocessing.')
    parser.add_argument('--representation', type=str, default='nasneta', help='choice of representation and search space.')
    parser.add_argument('--reference-file', type=str, default='reference.csv', help='Reference architectures csv file')
    parser.add_argument('--input-pipeline', type=str, default='cifar10', help='input pipeline to test.')

    args, _ = parser.parse_known_args()

    representation_name = 'dvolver.representations.' + args.representation

    try:
        print('Loading representation:', representation_name)
        representation = importlib.import_module(representation_name)

    except ImportError:
        print('Failed to find representation:', representation_name)
        exit()

    # load specific arguments for current representation
    representation.add_argument(parser, args.train_mode)
    args = parser.parse_args()

    input_pipeline_name = 'input_pipeline.' + args.input_pipeline

    try:
        print('Loading input pipeline:', input_pipeline_name)
        input_pipeline = importlib.import_module(input_pipeline_name)

    except ImportError:
        print('Failed to find input pipeline:', input_pipeline_name)
        exit()

    args.reference_file = find_reference_file(args.representation, args.reference_file)

    print('Command line arguments:')
    for arg in sorted(vars(args)):
        print('\t', arg+':', getattr(args, arg))

    main(args)
