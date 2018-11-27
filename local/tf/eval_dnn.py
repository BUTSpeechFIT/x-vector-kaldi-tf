#!/usr/bin/env python

from __future__ import print_function

import argparse
import logging
import os
import sys
import traceback

import mkl
import numexpr

import ze_utils as utils
from models import Model
from train_dnn_one_iteration import TarFileDataLoader

mkl.set_num_threads(1)
numexpr.set_num_threads(1)

MKL_NUM_THREADS = 1
OMP_NUM_THREADS = 1

logger = logging.getLogger('eval_dnn')
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(pathname)s:%(lineno)s - "
                              "%(funcName)s - %(levelname)s ] %(message)s")


def get_args():
    """ Get args from stdin.
    """

    parser = argparse.ArgumentParser(
        description="""Trains a feed forward DNN using frame-level objectives like cross-entropy 
        and mean-squared-error. DNNs include simple DNNs, TDNNs and CNNs.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler='resolve')

    # Parameters for the optimization0
    parser.add_argument("--use-gpu", type=str, dest='use_gpu', choices=["yes", "no"],
                        help="Use GPU for training.", default="yes")

    parser.add_argument("--tar-file", type=str, dest='tar_file', required=True,
                        help="Specifies a tar file which contains the training data. Also, there must "
                             "ans npy file for labels with same name but with npy extension. If tar file "
                             "was given the scp and ranges file didn't used but at least one there two "
                             "must given.")

    parser.add_argument("--input-dir", type=str, dest='input_dir', required=True,
                        help="Specify the input directory. The model will loaded from this directory and "
                             "the new model will wrote to the output directory.")

    parser.add_argument("--log-file", type=str, dest='log_file', required=True,
                        help="Specify the log file for training to be able to separate training logs "
                             "from tensorflow logs.")

    print(' '.join(sys.argv))

    args = parser.parse_args()

    args = process_args(args)

    handler = logging.StreamHandler(open(args.log_file, 'wt'))
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info('Starting DNN evaluation (eval_dnn.py)')

    return args


def process_args(args):
    """ Process the options got from get_args()
    """

    args.input_dir = args.input_dir.strip()
    if args.input_dir == '' or not os.path.exists(os.path.join(args.input_dir, 'model.meta')):
        raise Exception("This scripts expects the input model was exist in '{0}' directory.".format(args.input_dir))

    if args.tar_file == '' or not os.path.exists(args.tar_file):
        raise Exception("The specified tar file '{0}' not exist.".format(args.tar_file))
    if not os.path.exists(args.tar_file.replace('.tar', '.npy')):
        raise Exception("There is no corresponding npy label file for tar file '{0}'.".format(args.tar_file))

    return args


def eval_dnn(args):
    """ The main function for doing evaluation on a trained network.

    Args:
        args: a Namespace object with the required parameters
            obtained from the function process_args()
    """

    input_dir = args.input_dir
    use_gpu = args.use_gpu == 'yes'
    data_loader = TarFileDataLoader(args.tar_file, logger=None, queue_size=16)
    model = Model()
    model.eval(data_loader, input_dir, use_gpu, logger)


def main():
    args = get_args()
    try:
        eval_dnn(args)
        utils.wait_for_background_commands()
    except BaseException as e:
        # look for BaseException so we catch KeyboardInterrupt, which is
        # what we get when a background thread dies.
        if not isinstance(e, KeyboardInterrupt):
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
