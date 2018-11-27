#!/usr/bin/env python

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import argparse
import logging
import os
import sys
import traceback

import numpy as np

import examples_io
import ze_utils as utils

logger = logging.getLogger('create_tar_files')
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(pathname)s:%(lineno)s - "
                              "%(funcName)s - %(levelname)s ] %(message)s")
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info('Starting DNN trainer to do a training iteration (train_dnn_one_iteration.py)')


def get_args():
    """ Get args from stdin.
    """

    parser = argparse.ArgumentParser(
        description="Create a tar file for fast DNN training.  Each of minibatch data will "
                    "saved to a separate numpy file within tar file.  The output file can be "
                    "accessed in sequential mode or in random access mode but the sequential "
                    "more is faster.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler='resolve')

    parser.add_argument("--prefix", type=str, default="",
                        help="The prefix which is used to distinguish between the train "
                             "and diagnostic files.")

    parser.add_argument("--egs-dir", type=str, dest='egs_dir', required=True,
                        help="Directory of training egs.")

    parser.add_argument("--shuffle", type=bool,
                        dest='shuffle', default=True,
                        help="Randomly shuffle the minibatches before writing to file.")

    parser.add_argument("--random-seed", type=int, dest='random_seed', default=0,
                        help="Sets the random seed for minibatch shuffling")

    parser.add_argument("--feature-dim", type=int, dest='feature_dim', required=True,
                        help="Shows the dimensions of the features. It is used to allocate matrices in "
                             "advance and also to check the dimension of read features with this number.")

    parser.add_argument("--minibatch-size",
                        type=int, dest='minibatch_size', required=True,
                        help="Size of the minibatch used in SGD training.")

    parser.add_argument("--outputs-file", type=str, dest='outputs_file', required=True,
                        help="Specifies the egs tar file which should be created.")

    print(' '.join(sys.argv))

    args = parser.parse_args()

    args = process_args(args)

    return args


def process_args(args):
    """ Process the options got from get_args()
    """

    if args.outputs_file == '' or not os.path.exists(args.outputs_file):
        raise Exception("The specified outputs file '{0}' not exist.".format(args.outputs_file))

    return args


def process_files(args):
    if args.random_seed != 0:
        np.random.seed(args.random_seed)
    egs_dir = args.egs_dir
    minibatch_size = args.minibatch_size
    feature_dim = args.feature_dim
    outputs_file = args.outputs_file

    prefix = ""
    if args.prefix != "":
        prefix = args.prefix + "_"
    archives_minibatch_count = {}
    with open('{0}/temp/{1}archive_minibatch_count'.format(egs_dir, prefix), 'rt') as fid:
        for line in fid:
            _line = line.strip()
            if len(_line) == 0:
                continue
            parts = _line.split()
            archives_minibatch_count[int(parts[0])] = int(parts[1])

    with open(outputs_file, 'rt') as fid:
        for line in fid:
            _line = line.strip()
            if len(_line) <= 1:
                continue
            tar_file_path = os.path.join(args.egs_dir, _line)
            idx = int(tar_file_path.split('.')[-2])
            range_file_path = os.path.join(args.egs_dir, 'temp/{0}ranges.{1}'.format(prefix, idx))
            scp_file_path = os.path.join(args.egs_dir, 'temp/{0}feats.scp.{1}'.format(prefix, idx))
            minibatch_count = archives_minibatch_count[idx]
            utt_to_chunks, minibatch_info = examples_io.process_range_file(range_file_path, minibatch_count,
                                                                           minibatch_size)
            all_data_info, labels = examples_io.load_ranges_info(utt_to_chunks, minibatch_info, minibatch_size,
                                                                 scp_file_path, feature_dim)
            if args.shuffle:
                shuffle_indices = np.random.permutation(np.arange(minibatch_count))
                minibatch_info = minibatch_info[shuffle_indices]
                all_data_info = all_data_info[shuffle_indices]
                labels = labels[shuffle_indices]
            if not os.path.exists(tar_file_path):
                logger.info('Processing file {%s}' % tar_file_path)
                temp_file = tar_file_path + '.tmp.tar'
                examples_io.save_data_info_tar(temp_file, minibatch_info, all_data_info, feature_dim, logger)
                os.rename(temp_file, tar_file_path)  # in case of interruption, this ensures atomicity
            else:
                logger.info('Output file {%s} exist from before.' % tar_file_path)
            npy_file_path = tar_file_path[:-4] + '.npy'     # change the extension
            if not os.path.exists(npy_file_path):
                temp_file = npy_file_path + '.tmp.npy'
                np.save(temp_file, labels)
                os.rename(temp_file, npy_file_path)


def main():
    args = get_args()
    try:
        process_files(args)
        utils.wait_for_background_commands()
    except BaseException as e:
        # look for BaseException so we catch KeyboardInterrupt, which is
        # what we get when a background thread dies.
        if not isinstance(e, KeyboardInterrupt):
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
