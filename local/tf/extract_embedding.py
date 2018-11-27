#!/usr/bin/env python

from __future__ import print_function

import argparse
import logging

import os
import sys
import traceback

import mkl
import numexpr

import kaldi_io
import ze_utils as utils
from models import Model

mkl.set_num_threads(1)
numexpr.set_num_threads(1)

MKL_NUM_THREADS = 1
OMP_NUM_THREADS = 1

logger = logging.getLogger('extract_embedding')
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(pathname)s:%(lineno)s - "
                              "%(funcName)s - %(levelname)s ] %(message)s")

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)
host_name = os.uname()[1]
logger.info('Start running on host: %s' % str(host_name))
logger.info('Extract embeddings from features (extract_embedding.py)')


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
                        help="Use GPU for training.", default="no")

    parser.add_argument("--min-chunk-size", type=int, dest='min_chunk_size', default=100,
                        help="Minimum chunk-size allowed when extracting xvectors.")

    parser.add_argument("--chunk-size", type=int, dest='chunk_size', default=-1,
                        help="If set, extracts xvectors from specified chunk-size, and averages.  "
                             "If not set, extracts an xvector from all available features.")

    parser.add_argument("--feature-rspecifier", type=str, dest='feature_rspecifier', required=True,
                        help="Specifies a Kaldi rspecifier which read Kaldi features and write "
                             "them to the pipe.")

    parser.add_argument("--vector-wspecifier", type=str, dest='vector_wspecifier', required=True,
                        help="Specifies a Kaldi wspecifier which convert ark pipe to ark,scp files.  "
                             "See Kaldi copy-vector for more info about this conversion.")

    parser.add_argument("--model-dir", type=str, dest='model_dir', required=True,
                        help="Specify the input model directory. The model will loaded from this "
                             "directory and the new model will wrote to the output directory.")

    # parser.add_argument("--log-file", type=str, dest='log_file',
    #                     help="Specify the log file for training to be able to separate training logs "
    #                          "from tensorflow logs.")

    args = parser.parse_args()

    args = process_args(args)

    return args


def process_args(args):
    """ Process the options got from get_args()
    """

    args.model_dir = args.model_dir.strip()
    if args.model_dir == '' or not os.path.exists(os.path.join(args.model_dir, 'model.meta')):
        raise Exception("This scripts expects the input model was exist in '{0}' directory.".format(args.model_dir))

    return args


def process_wspecifier(wspecifier):
    out_wspecifier = ''
    parts = wspecifier.split()
    for i in range(len(parts) - 1):
        out_wspecifier += parts[i] + ' '
    if parts[-1].startswith('ark,scp:'):
        ark, scp = parts[-1][8:].split(',')
        out_wspecifier += 'ark,scp:%s.tmp.ark,%s.tmp.scp' % (ark, scp)
        return out_wspecifier, ark, scp
    if parts[-1].startswith('scp,ark:'):
        scp, ark = parts[-1][8:].split(',')
        out_wspecifier += 'scp,ark:%s.tmp.scp,%s.tmp.ark' % (scp, ark)
        return out_wspecifier, ark, scp
    else:
        return wspecifier, None, None


def eval_dnn(args):
    """ The main function for doing evaluation on a trained network.

    Args:
        args: a Namespace object with the required parameters
            obtained from the function process_args()
    """

    model_dir = args.model_dir
    use_gpu = args.use_gpu == 'yes'
    min_chunk_size = args.min_chunk_size
    chunk_size = args.chunk_size
    # First change the output files temp ones and at the end rename them
    wspecifier, ark, scp = process_wspecifier(args.vector_wspecifier)

    if ark is not None and os.path.exists(ark) and scp is not None and os.path.exists(scp):
        logger.info('Both output ark and scp files exist. Return from this call.')
        return
    model = Model()
    with kaldi_io.open_or_fd(args.feature_rspecifier) as input_fid:
        with kaldi_io.open_or_fd(wspecifier) as output_fid:
            model.make_embedding(input_fid, output_fid, model_dir, min_chunk_size, chunk_size, use_gpu, logger)

    # rename output files
    if ark is not None:
        os.rename(ark + '.tmp.ark', ark)
    # first load scp and correct them to point to renamed ark file.
    if scp is not None:
        with open(scp + '.tmp.scp', 'rt') as fid_in:
            with open(scp + '.tmp', 'wr') as fid_out:
                text = fid_in.read()
                text = text.replace('ark.tmp.ark', 'ark')
                # Sometimes there is no \n at the end of file ank cause a Kaldi error.
                # For preventing this error juts check the last char and append \n if not exist
                if text[-1] != '\n':
                    text += '\n'
                fid_out.write(text)
        os.rename(scp + '.tmp', scp)
        # after create scp file  now we can delete temp file
        # os.remove(scp + '.tmp.scp')


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
