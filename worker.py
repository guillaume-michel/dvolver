#!/usr/bin/python3
"""
Dvolver Worker: get a representation and train the corresponding network
"""
import argparse
import numpy as np

import random
import copy
import glob
import os
import logging
import threading
import time
import json
import importlib

from concurrent import futures

from tqdm import tqdm
tqdm.monitor_interval = 0  # see https://github.com/tqdm/tqdm/issues/481

import tensorflow as tf

from objective import make_objective

from dvolver import *

import pika

class L(list):
    """
    A subclass of list that can accept additional attributes.
    Should be able to be used just like a regular list.

    The problem:
    a = [1, 2, 4, 8]
    a.x = "Hey!" # AttributeError: 'list' object has no attribute 'x'

    The solution:
    a = L(1, 2, 4, 8)
    a.x = "Hey!"
    print a       # [1, 2, 4, 8]
    print a.x     # "Hey!"
    print len(a)  # 4

    You can also do these:
    a = L( 1, 2, 4, 8 , x="Hey!" )                 # [1, 2, 4, 8]
    a = L( 1, 2, 4, 8 )( x="Hey!" )                # [1, 2, 4, 8]
    a = L( [1, 2, 4, 8] , x="Hey!" )               # [1, 2, 4, 8]
    a = L( {1, 2, 4, 8} , x="Hey!" )               # [1, 2, 4, 8]
    a = L( [2 ** b for b in range(4)] , x="Hey!" ) # [1, 2, 4, 8]
    a = L( (2 ** b for b in range(4)) , x="Hey!" ) # [1, 2, 4, 8]
    a = L( 2 ** b for b in range(4) )( x="Hey!" )  # [1, 2, 4, 8]
    a = L( 2 )                                     # [2]
    """
    def __new__(self, *args, **kwargs):
        return super(L, self).__new__(self, args, kwargs)

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and hasattr(args[0], '__iter__'):
            list.__init__(self, args[0])
        else:
            list.__init__(self, args)
        self.__dict__.update(kwargs)

    def __call__(self, **kwargs):
        self.__dict__.update(kwargs)
        return self


class WorkerArgs():
    pass


def getWorkerArgs(d):
    worker_args = WorkerArgs()

    for key,value in d.items():
        setattr(worker_args, key, value)

    setattr(worker_args, 'num_train_samples', 0)
    setattr(worker_args, 'num_test_samples', 0)

    return worker_args


def get_objective_function(worker_args):

    input_pipeline_name = worker_args.input_pipeline_name

    try:
        print('Loading input pipeline:', input_pipeline_name)
        input_pipeline = importlib.import_module(input_pipeline_name)

    except ImportError:
        print('Failed to find input pipeline:', input_pipeline_name)
        exit()

    train_mode = TrainMode(worker_args.train_mode)

    if train_mode == TrainMode.SEARCH:
        TRAIN_LIST, worker_args.num_train_samples, VALIDATION_LIST, worker_args.num_test_samples = input_pipeline.get_search_mode_files(worker_args.data_dir)

    elif train_mode == TrainMode.FULL:
        TRAIN_LIST, worker_args.num_train_samples, VALIDATION_LIST, worker_args.num_test_samples = input_pipeline.get_full_mode_files(worker_args.data_dir)

    else:
        raise ValueError('Unsupported TrainMode' + train_mode)

    return make_objective(worker_args, train_list=TRAIN_LIST, test_list=VALIDATION_LIST)


def parse(msg):
    """
    example of representation (json format):
    {
        "individual": [1, 2, 3]
    }
    """
    return json.loads(msg.decode('utf-8'))


def process_msg(body):
    request = parse(body)

    print('='*80)
    print('Start to train arch: {0}'.format(request['archIndex']))
    print('='*80)

    worker_args = getWorkerArgs(request['worker_args'])

    objectiveFunc = get_objective_function(worker_args)

    x = L(request['individual'])

    x.archIndex = request['archIndex']
    x.gen = request['gen']

    fitness = objectiveFunc(x)
    #print('fitness =', fitness)

    response = request
    response['fitness'] = list(fitness)

    #print('response:', response)

    json_response = json.dumps(response)

    return json_response


def get_on_request(ex):

    def on_request(ch, method, props, body):
        #print('Received body:', body)

        # schedule in background thread so to avoid connection timeout du to heartbeat failure
        f = ex.submit(process_msg, body)

        # active polling for task completion
        while not f.done():
            ch.connection.process_data_events()
            ch.connection.sleep(1)

        json_response = f.result()

        # post results to response queue
        ch.basic_publish(exchange='',
                         routing_key=props.reply_to,
                         properties=pika.BasicProperties(correlation_id = props.correlation_id,
                                                         content_type = 'application/json'),
                         body=json_response)
        ch.basic_ack(delivery_tag = method.delivery_tag)

    return on_request


def main(args):

    # When on_request is called heartbeart is blocked and eventually connection timeout
    # we use a background thread for long running tasks processing, so the heartbeat can work correctly
    ex = futures.ThreadPoolExecutor(max_workers=1)

    connection = pika.BlockingConnection(pika.ConnectionParameters(
        host=args.broker))

    channel = connection.channel()

    channel.queue_declare(queue=args.request_queue)
    channel.queue_declare(queue=args.response_queue)

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(consumer_callback=get_on_request(ex), queue=args.request_queue)

    print(" [x] Awaiting RPC requests on queue:", args.request_queue)
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        print('Keyboard Interrupt called. Wait until it properly stops...')
    finally:
        # Cancel the consumer and return any pending messages
        channel.cancel()
        connection.close()
        print('Shuting down executor')
        ex.shutdown(wait=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--broker', type=str, default='rabbitmq', help='RabbitMQ broker')
    parser.add_argument('--request-queue', type=str, default='dvolver_request', help='request queue (from server -> workers)')
    parser.add_argument('--response-queue', type=str, default='dvolver_response', help='response queue (from workers -> server)')

    args = parser.parse_args()

    main(args)
