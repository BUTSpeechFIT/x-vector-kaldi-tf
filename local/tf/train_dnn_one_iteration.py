#!/usr/bin/env python

import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import argparse
import logging
import os
import pprint
import sys
import traceback

import numpy as np

import ze_utils as utils
from examples_io import DataLoader, TarFileDataLoader, process_range_file, load_ranges_info, load_ranges_data
import models

logger = logging.getLogger('train_dnn_one_iteration')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(pathname)s:%(lineno)s - "
                              "%(funcName)s - %(levelname)s ] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info('Starting DNN trainer to do a training iteration (train_dnn_one_iteration.py)')


def get_args():
    """ Get args from stdin.
    """

    parser = argparse.ArgumentParser(
        description="""Trains a feed forward DNN using frame-level objectives like cross-entropy 
        and mean-squared-error. DNNs include simple DNNs, TDNNs and CNNs.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler='resolve')

    # Parameters for the optimization
    parser.add_argument("--use-gpu", type=str, dest='use_gpu',
                        choices=["yes", "no", "wait"],
                        help="Use GPU for training.",
                        default="yes")

    parser.add_argument("--momentum", type=float,
                        dest='momentum', default=0.0,
                        help="""Momentum used in update computation.
                        Note: we implemented it in such a way that it
                        doesn't increase the effective learning rate.""")

    parser.add_argument("--shuffle", type=bool,
                        dest='shuffle', default=False,
                        help="Randomly shuffle the training examples.")

    parser.add_argument("--max-param-change", type=float,
                        dest='max_param_change', default=2.0,
                        help="""The maximum change in parameters allowed per minibatch, measured in 
                        Frobenius norm over the entire model""")

    parser.add_argument("--l2-regularize-factor", type=float,
                        dest='max_param_change', default=1.0,
                        help="Factor that affects the strength of l2 regularization on model "
                             "parameters.  --l2-regularize-factor will be multiplied by the component-level "
                             "l2-regularize values and can be used to correct for effects "
                             "related to parallelization by model averaging.")

    parser.add_argument("--random-seed", type=int, dest='random_seed', default=0,
                        help="""Sets the random seed for egs shuffling and tensorflow random seed.
                             Warning: This random seed does not control all aspects of this 
                             experiment.  There might be other random seeds used in other stages of the
                             experiment like data preparation (e.g. volume perturbation).""")

    parser.add_argument("--print-interval", type=int, dest='print_interval', default=10,
                        help="The interval for log printing.")

    parser.add_argument("--verbose", type=int, dest='verbose', default=0,
                        help="Shows the verbose level.")

    parser.add_argument("--feature-dim", type=int, dest='feature_dim', required=True,
                        help="Shows the dimensions of the features. It is used to allocate matrices in "
                             "advance and also to check the dimension of read features with this number.")

    parser.add_argument("--minibatch-size",
                        type=int, dest='minibatch_size', required=True,
                        help="Size of the minibatch used in SGD training.")

    parser.add_argument("--minibatch-count",
                        type=int, dest='minibatch_count', required=True,
                        help="""Number of minibatches in the current ranges file. 
                        This is required to be able to allocate the space in the advance.""")

    parser.add_argument("--learning-rate", type=float,
                        dest='learning_rate', default=-1.0,
                        help="If supplied, all the learning rates of updatable components"
                             "are set to this value.")

    parser.add_argument("--scale", type=float,
                        dest='scale', default=1.0,
                        help="The parameter matrices are scaled by the specified value.")

    parser.add_argument("--dropout-proportion", type=float,
                        dest='dropout_proportion', default=0.0,
                        help="Shows the dropout proportions for the current iteration.")

    parser.add_argument("--ranges-file", type=str, dest='ranges_file',
                        help="Specifies a ranges file which used for current training iteration.")

    parser.add_argument("--scp-file", type=str, dest='scp_file',
                        help="Specifies a scp kaldi file which contains the only corresponding ark file to "
                             "specified ranges file. Note that this file is processed sequentially and if it "
                             "did not filter out the other ark files, it just waste the time and read "
                             "unnecessarily files. Before sending the scp file to this file, use "
                             "utils/filter_scp.pl to filer the overall scp feature file.")

    parser.add_argument("--tar-file", type=str, dest='tar_file',
                        help="Specifies a tar file which contains the training data. Also, there must "
                             "ans npy file for labels with same name but with npy extension. If tar file "
                             "was given the scp and ranges file didn't used but at least one there two "
                             "must given.")

    parser.add_argument("--sequential-loading", dest='sequential_loading', type=str,
                        action=utils.StrToBoolAction, choices=["true", "false"], default=True,
                        help="If true, every minibatch is loaded before sending to the GPU."
                             "This makes the starting faster. If false, the whole training archive ")

    parser.add_argument("--input-dir", type=str, dest='input_dir', required=True,
                        help="Specify the input directory. The model will loaded from this directory and "
                             "the new model will wrote to the output directory.")

    parser.add_argument("--output-dir", type=str, dest='output_dir', required=True,
                        help="Specify the output directory. The new model will wrote to the output directory.")

    print(' '.join(sys.argv))

    args = parser.parse_args()

    args = process_args(args)

    return args


def process_args(args):
    """ Process the options got from get_args()
    """

    args.input_dir = args.input_dir.strip()
    if args.input_dir == '' or not os.path.exists(os.path.join(args.input_dir, 'model.meta')):
        raise Exception("This scripts expects the input model was exist in '{0}' directory.".format(args.input_dir))

    if args.tar_file == '':
        if args.ranges_file == '' or not os.path.exists(args.ranges_file):
            raise Exception("The specified range file '{0}' not exist.".format(args.ranges_file))

        if args.scp_file == '' or not os.path.exists(args.scp_file):
            raise Exception("The specified scp file '{0}' not exist.".format(args.scp_file))
    else:
        if not os.path.exists(args.tar_file):
            raise Exception("The specified tar file '{0}' not exist.".format(args.tar_file))
        if not os.path.exists(args.tar_file.replace('.tar', '.npy')):
            raise Exception("There is no corresponding npy label file for tar file '{0}'.".format(args.tar_file))

    if args.dropout_proportion > 1.0 or args.dropout_proportion < 0.0:
        raise Exception("The value of dropout-proportion must be in range [0 - 1].")

    return args


def train(args):
    """ The main function for doing one iteration training on a input.

    Args:
        args: a Namespace object with the required parameters
            obtained from the function process_args()
    """

    arg_string = pprint.pformat(vars(args))
    logger.info("Arguments for the experiment\n{0}".format(arg_string))

    if args.random_seed != 0:
        np.random.seed(args.random_seed)
    minibatch_count = args.minibatch_count
    minibatch_size = args.minibatch_size
    feature_dim = args.feature_dim

    if args.tar_file == '':
        utt_to_chunks, minibatch_info = process_range_file(args.ranges_file, minibatch_count, minibatch_size)
        if args.sequential_loading:
            train_data, train_labels = load_ranges_info(utt_to_chunks, minibatch_info, minibatch_size, args.scp_file,
                                                        feature_dim)
        else:
            train_data, train_labels = load_ranges_data(utt_to_chunks, minibatch_info, minibatch_size, args.scp_file,
                                                        feature_dim)
        if args.shuffle:
            shuffle_indices = np.random.permutation(np.arange(minibatch_count))
            train_data = train_data[shuffle_indices]
            train_labels = train_labels[shuffle_indices]
        data_loader = DataLoader(train_data, train_labels, args.sequential_loading, queue_size=16)
    else:
        data_loader = TarFileDataLoader(args.tar_file, logger=None, queue_size=16)

    if 'ModelWithoutDropoutReluAdversarial' in args.input_dir:
        model = models.ModelWithoutDropoutReluAdversarial()
        model.train_one_iteration(data_loader, args, logger)
    else:
        model = models.Model()
        model.train_one_iteration(data_loader, args, logger)


def main():
    args = get_args()
    try:
        train(args)
        utils.wait_for_background_commands()
    except BaseException as e:
        # look for BaseException so we catch KeyboardInterrupt, which is
        # what we get when a background thread dies.
        if not isinstance(e, KeyboardInterrupt):
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
