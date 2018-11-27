#!/usr/bin/env python

import argparse
import os
import sys
import traceback

import numpy as np


def get_args():
    parser = argparse.ArgumentParser(
        description="""Convert Kaldi score file to another one suitable for NIST scoring tool.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler='resolve')

    # Parameters for the optimization0
    parser.add_argument("--vast-score-file", type=str, dest='vast_score_file',
                        help="Path to the vast score file.", default="")

    parser.add_argument("--select-best-test-spk", type=str, dest='select_best_test_spk', choices=["yes", "no"],
                        help="Select maximum scores based on different diarization candidate.", default="no")

    parser.add_argument("input_file", type=str, help="Shows input scores file.")

    parser.add_argument("trials_file", type=str, help="Shows trials file.")

    parser.add_argument("output_file", type=str, help="Shows output scores file.")

    # print(' '.join(sys.argv))

    args = parser.parse_args()

    args = process_args(args)

    return args


def process_args(args):
    """ Process the options got from get_args()
    """
    if not os.path.exists(args.input_file):
        raise Exception("This scripts expects input scores file [ {0} ] exist.".format(args.input_file))

    if not os.path.exists(args.trials_file):
        raise Exception("This scripts expects trials file [ {0} ] exist.".format(args.trials_file))

    args.select_best_test_spk = args.select_best_test_spk == 'yes'

    return args


def make(args):
    trials_file = args.trials_file
    trials = np.genfromtxt(trials_file, dtype=str, delimiter='\t')
    trial2ext = dict()
    for i in range(1, trials.shape[0]):
        seg_id, ext = trials[i, 1].split('.')
        trial2ext[trials[i, 0] + '-' + seg_id] = '.' + ext + '\t' + trials[i, 2]
    scores = np.genfromtxt(args.input_file, dtype=str, delimiter=' ')
    vast_scores = None
    if len(args.vast_score_file) > 0 and os.path.exists(args.vast_score_file):
        vast_scores = {}
        with open(args.vast_score_file, 'rt') as fid:
            for line in fid:
                parts = line[:-1].split()
                parts[1] = os.path.basename(parts[1])
                spk = vast_scores.get(parts[0])
                if spk is None:
                    spk = {}
                    vast_scores[parts[0]] = spk
                if args.select_best_test_spk:
                    name = parts[1][:parts[1].find('-SPK')]
                    if name in spk:
                        m = max(spk[name], float(parts[2]))
                        # print('%s %s Select %.5f between %.5f and %s' % (parts[0], name, m, spk[name], parts[2]))
                        spk[name] = m
                    else:
                        spk[name] = float(parts[2])
                else:
                    spk[parts[1]] = parts[2]
    fid = open(args.output_file, 'wt')
    fid.write("modelid\tsegmentid\tside\tLLR\n")
    cc, cc1 = 0, 0
    for i in range(scores.shape[0]):
        name = os.path.basename(scores[i, 1])
        sco = scores[i, 2]
        if vast_scores is not None:
            spk = vast_scores.get(scores[i, 0])
            if spk is not None:
                if name in spk:
                    sco = str(spk[name])
                    cc += 1
                else:
                    cc1 += 1
                    # print('Warning: ' + name)
                    # print(spk.keys())
        fid.write("%s\t%s%s\t%s\n" % (scores[i, 0], name, trial2ext[scores[i, 0] + '-' + name], sco))
    fid.close()
    # print('Total converted is ' + str(cc) + ", " + str(cc1))


def main():
    args = get_args()
    try:
        make(args)
    except BaseException as e:
        # look for BaseException so we catch KeyboardInterrupt, which is
        # what we get when a background thread dies.
        if not isinstance(e, KeyboardInterrupt):
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
