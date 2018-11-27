#!/usr/bin/env python

import argparse
import logging
import math
import os
import pprint
import shutil
import sys
import traceback

import time

import models
import ze_utils as utils

MAX_TRY_COUNT = 16

logger = logging.getLogger('train_dnn')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(pathname)s:%(lineno)s - "
                              "%(funcName)s - %(levelname)s ] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def get_args():
    """ Get args from stdin.
    """

    parser = argparse.ArgumentParser(
        description="""Trains a feed forward raw DNN (without transition model)
        using frame-level objectives like cross-entropy and mean-squared-error.
        DNNs include simple DNNs, TDNNs and CNNs.""",
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

    parser.add_argument("--targets-scp", type=str, required=False,
                        help="Targets for training neural network.")

    parser.add_argument("--tf-model-class", type=str, dest='tf_model_class', required=True,
                        help="Shows the tensorflow class name which should was defined in models.py file.")

    parser.add_argument("--dir", type=str, required=True,
                        help="Directory to store the models and all other files.")

    parser.add_argument("--egs-dir", type=str, dest='egs_dir', required=True,
                        help="Directory of training egs.")

    parser.add_argument("--num-epochs", type=float, dest='num_epochs', default=6.0,
                        help="Number of epochs to train the model.")

    parser.add_argument("--num-targets", type=int, dest='num_targets', required=True,
                        help="Shows the number of output of the neural network.")

    parser.add_argument("--initial-effective-lrate", type=float,
                        dest='initial_effective_lrate', default=0.0003,
                        help="Learning rate used during the initial iteration.")

    parser.add_argument("--final-effective-lrate", type=float,
                        dest='final_effective_lrate', default=0.00003,
                        help="Learning rate used during the final iteration.")

    parser.add_argument("--num-jobs-initial", type=int, dest='num_jobs_initial', default=1,
                        help="Number of neural net jobs to run in "
                             "parallel at the start of training")

    parser.add_argument("--num-jobs-final", type=int, dest='num_jobs_final', default=8,
                        help="Number of neural net jobs to run in "
                             "parallel at the end of training")

    parser.add_argument("--minibatch-size",
                        type=int, dest='minibatch_size', required=True,
                        help="Size of the minibatch used in SGD training.")

    parser.add_argument("--do-final-combination", dest='do_final_combination', type=str,
                        action=utils.StrToBoolAction,
                        choices=["true", "false"], default=False,
                        help="""Set this to false to disable the final
                             'combine' stage (in this case we just use the
                             last-numbered model as the final.mdl).""")

    parser.add_argument("--random-seed", type=int, dest='random_seed', default=0,
                        help="""Sets the random seed for egs shuffling and tensorflow random seed.
                             Warning: This random seed does not control all aspects of this 
                             experiment.  There might be other random seeds used in other stages of the
                             experiment like data preparation (e.g. volume perturbation).""")

    parser.add_argument("--dropout-schedule", type=str,
                        action=utils.NullStrToNoneAction,
                        dest='dropout_schedule', default=None,
                        help="""Use this to specify the dropout
                             schedule.  You specify a piecewise linear
                             function on the domain [0,1], where 0 is the
                             start and 1 is the end of training; the
                             function-argument (x) rises linearly with the
                             amount of data you have seen, not iteration
                             number (this improves invariance to
                             num-jobs-{initial-final}).  E.g. '0,0.2,0'
                             means 0 at the start; 0.2 after seeing half
                             the data; and 0 at the end.  You may specify
                             the x-value of selected points, e.g.
                             '0,0.2@0.25,0' means that the 0.2
                             dropout-proportion is reached a quarter of the
                             way through the data.   The start/end x-values
                             are at x=0/x=1, and other unspecified x-values
                             are interpolated between known x-values.  You
                             may specify different rules for different
                             component-name patterns using 'pattern1=func1
                             pattern2=func2', e.g. 'relu*=0,0.1,0
                             lstm*=0,0.2,0'.  More general should precede
                             less general patterns, as they are applied
                             sequentially.""")
    parser.add_argument("--max-objective-evaluations",
                        type=int, dest='max_objective_evaluations', default=30,
                        help="""The maximum number of objective evaluations in order to figure out the
                         best number of models to combine. It helps to speedup if the number of models provided to the
                         model combination binary is quite large (e.g. several hundred).""")

    parser.add_argument("--preserve-model-interval", dest="preserve_model_interval",
                        type=int, default=10,
                        help="""Determines iterations for which models
                             will be preserved during cleanup.
                             If mod(iter,preserve_model_interval) == 0
                             model will be preserved.""")

    parser.add_argument("--cleanup", type=str, action=utils.StrToBoolAction,
                        choices=["true", "false"], default=True,
                        help="Clean up models after training")

    parser.add_argument("--max-param-change", type=float, dest='max_param_change',
                        default=2.0, help="""The maximum change in parameters
                             allowed per minibatch, measured in Frobenius
                             norm over the entire model""")

    parser.add_argument("--proportional-shrink", type=float, dest='proportional_shrink',
                        default=0.0, help="""If nonzero, this will set a shrinkage (scaling)
                        factor for the parameters, whose value is set as:
                        shrink-value=(1.0 - proportional-shrink * learning-rate), where
                        'learning-rate' is the learning rate being applied
                        on the current iteration, which will vary from
                        initial-effective-lrate*num-jobs-initial to
                        final-effective-lrate*num-jobs-final.
                        Unlike for train_rnn.py, this is applied unconditionally,
                        it does not depend on saturation of nonlinearities.
                        Can be used to roughly approximate l2 regularization.""")

    parser.add_argument("--stage", type=int, default=-4,
                        help="Specifies the stage of the experiment to execution from")

    parser.add_argument("--cmd", type=str, dest="command",
                        action=utils.NullStrToNoneAction,
                        help="""Specifies the script to launch jobs.
                             e.g. queue.pl for launching on SGE cluster
                                    run.pl for launching on local machine
                             """, default="queue.pl")

    parser.add_argument("--max-models-combine", type=int, dest='max_models_combine',
                        default=20, help="""The maximum number of models used in
                                 the final model combination stage.  These
                                 models will themselves be averages of
                                 iteration-number ranges""")

    print(' '.join(sys.argv))

    args = parser.parse_args()

    [args, run_opts] = process_args(args)

    return [args, run_opts]


def process_args(args):
    """ Process the options got from get_args()
    """

    if not os.path.exists(args.dir):
        raise Exception("This scripts expects {0} to exist.".format(args.dir))

    try:
        eval('models.%s()' % args.tf_model_class)
    except AttributeError:
        raise Exception("The specified class name {0} does not exist.".format(args.tf_model_class))

    # set the options corresponding to args.use_gpu
    run_opts = utils.RunOpts()
    if args.use_gpu in ["yes", "wait"]:
        run_opts.train_queue_opt = "--gpu 1"
        run_opts.parallel_train_opts = "--use-gpu={}".format(args.use_gpu)
        run_opts.combine_gpu_opt = "--use-gpu={}".format(args.use_gpu)
        run_opts.combine_queue_opt = "--gpu 1"
        run_opts.prior_gpu_opt = "--use-gpu={}".format(args.use_gpu)
        run_opts.prior_queue_opt = "--gpu 1"
    else:
        logger.warning("Without using a GPU this will be very slow. "
                       "nnet3 does not yet support multiple threads.")
        run_opts.train_queue_opt = ""
        run_opts.parallel_train_opts = "--use-gpu=no"
        run_opts.combine_gpu_opt = "--use-gpu=no"
        run_opts.combine_queue_opt = ""
        run_opts.prior_gpu_opt = "--use-gpu=no"
        run_opts.prior_queue_opt = ""

    run_opts.command = args.command
    # run_opts.num_jobs_compute_prior = args.num_jobs_compute_prior

    return [args, run_opts]


def train_new_models(model_dir, _iter, random_seed, num_jobs,
                     num_archives_processed, num_archives,
                     learning_rate, shrinkage_value, dropout_proportion, egs_dir,
                     momentum, max_param_change, minibatch_size,
                     run_opts, feature_dim, archives_minibatch_count, try_count=0, train_opts=""):
    """ Called from train_one_iteration(), this model does one iteration of
    training with 'num_jobs' jobs, and writes models in dirs like
    exp/tdnn_a/model_24.{1,2,3,..<num_jobs>}

    We cannot easily use a single parallel SGE job to do the main training,
    because the computation of which archive and which --frame option
    to use for each job is a little complex, so we spawn each one separately.
    """

    threads = []

    # the GPU timing info is only printed if we use the --verbose=1 flag; this
    # slows down the computation slightly, so don't accumulate it on every
    # iteration.  Don't do it on iteration 0 either, because we use a smaller
    # than normal minibatch size, and people may get confused thinking it's
    # slower for iteration 0 because of the verbose option.
    verbose_opt = ("--verbose=1" if _iter % 20 == 0 and _iter > 0 else "")

    for job in range(1, num_jobs + 1):
        # k is a zero-based index that we will derive the other indexes from.
        k = num_archives_processed + job - 1

        # work out the 1-based archive index.
        archive_index = (k % num_archives) + 1
        minibatch_count = archives_minibatch_count[archive_index]

        if try_count > 0 and utils.is_correct_model_dir('{0}/model_{1}.{2}'.format(model_dir, _iter + 1, job)):
            continue

        egs_rspecifier = \
            '--ranges-file="{egs_dir}/temp/ranges.{archive_index}" ' \
            '--scp-file="{egs_dir}/temp/feats.scp.{archive_index}" ' \
            '--shuffle=True --minibatch-size={minibatch_size}'.format(
                egs_dir=egs_dir, archive_index=archive_index,
                minibatch_size=minibatch_size)

        # check whether tar file exist or not. If it was generated, so lets pass it to the script for speedup
        tar_file = '{egs_dir}/egs.{archive_index}.tar'.format(egs_dir=egs_dir, archive_index=archive_index)
        if os.path.exists(tar_file):
            egs_rspecifier = '--tar-file="{0}" {1}'.format(tar_file, egs_rspecifier)

        _command = '{command} {train_queue_opt} {dir}/log/train.{iter}.{job}.log ' \
                   'local/tf/train_dnn_one_iteration.py ' \
                   '{parallel_train_opts} ' \
                   '{verbose_opt} --print-interval=10 ' \
                   '--momentum={momentum} ' \
                   '--max-param-change={max_param_change} ' \
                   '--l2-regularize-factor={l2_regularize_factor} ' \
                   '--random-seed={random_seed} {train_opts} ' \
                   '--learning-rate={learning_rate} ' \
                   '--scale={shrinkage_value} ' \
                   '--minibatch-count={minibatch_count} ' \
                   '--feature-dim={feature_dim} ' \
                   '--dropout-proportion={dropout_proportion} ' \
                   '{egs_rspecifier} ' \
                   '--input-dir={dir}/model_{iter} ' \
                   '--output-dir={dir}/model_{next_iter}.{job}' \
            .format(command=run_opts.command,
                    train_queue_opt=run_opts.train_queue_opt,
                    dir=model_dir, iter=_iter,
                    next_iter=_iter + 1, random_seed=_iter + random_seed,
                    job=job,
                    parallel_train_opts=run_opts.parallel_train_opts,
                    verbose_opt=verbose_opt,
                    momentum=momentum, max_param_change=max_param_change,
                    l2_regularize_factor=1.0 / num_jobs,
                    train_opts=train_opts,
                    learning_rate=learning_rate,
                    shrinkage_value=shrinkage_value,
                    minibatch_count=minibatch_count,
                    feature_dim=feature_dim,
                    dropout_proportion=dropout_proportion,
                    egs_rspecifier=egs_rspecifier)

        thread = utils.background_command(_command, require_zero_status=False)
        threads.append(thread)

    for thread in threads:
        thread.join()


def train_one_iteration(model_dir, _iter, random_seed, egs_dir,
                        num_jobs, num_archives_processed, num_archives,
                        learning_rate, minibatch_size,
                        momentum, max_param_change, run_opts,
                        feature_dim, archives_minibatch_count,
                        shrinkage_value=1.0, current_dropout=0.0):
    """ Called from train for one iteration of neural network training

    Selected args:
        shrinkage_value: If value is 1.0, no shrinkage is done; otherwise
            parameter values are scaled by this value.
    """

    # check if different iterations use the same random seed
    random_seed_file = '{0}/random_seed'.format(model_dir)
    if os.path.exists(random_seed_file):
        try:
            with open(random_seed_file, 'r') as fid:
                saved_random_seed = int(fid.readline().strip())
        except (IOError, ValueError):
            logger.error("Exception while reading the random seed for training")
            raise
        if random_seed != saved_random_seed:
            logger.warning("The random seed provided to this iteration (random_seed={0}) is "
                           "different from the one saved last time (random_seed={1}). "
                           "Using random_seed={0}.".format(random_seed, saved_random_seed))
    else:
        with open(random_seed_file, 'w') as fid:
            fid.write(str(random_seed))

    # Sets off some background jobs to compute train and validation set objectives
    eval_trained_dnn(model_dir, _iter, egs_dir, run_opts)

    new_model = "{0}/model_{1}/model.meta".format(model_dir, _iter + 1)
    if utils.is_correct_model_dir("{0}/model_{1}".format(model_dir, _iter + 1)):
        logger.info('The output model {0} was exist and so I do not continue this iteration.'.format(new_model))
        return

    do_average = (_iter > 0)

    dropout_proportion = current_dropout

    if do_average or True:  # TODO
        cur_minibatch_size = minibatch_size
        cur_max_param_change = max_param_change
    else:
        # on iteration zero, use a smaller minibatch size (and we will later
        # choose the output of just one of the jobs): the model-averaging isn't
        # always helpful when the model is changing too fast (i.e. it can worsen
        # the objective function), and the smaller minibatch size will help to
        # keep the update stable.
        cur_minibatch_size = minibatch_size / 2
        cur_max_param_change = float(max_param_change) / math.sqrt(2)

    try_count = 0
    training_flag = True
    while try_count < MAX_TRY_COUNT and training_flag:
        train_new_models(model_dir=model_dir, _iter=_iter, random_seed=random_seed, num_jobs=num_jobs,
                         num_archives_processed=num_archives_processed,
                         num_archives=num_archives, learning_rate=learning_rate,
                         shrinkage_value=shrinkage_value, dropout_proportion=dropout_proportion,
                         egs_dir=egs_dir, momentum=momentum, max_param_change=cur_max_param_change,
                         minibatch_size=cur_minibatch_size, feature_dim=feature_dim, try_count=try_count,
                         archives_minibatch_count=archives_minibatch_count, run_opts=run_opts)
        try_count += 1
        training_flag = False
        cnt = 0
        for job in range(1, num_jobs + 1):
            if not utils.is_correct_model_dir("{0}/model_{1}.{2}".format(model_dir, _iter + 1, job)):
                # move logs to prevent rewriting.
                log_file = "{0}/log/train.{1}.{2}.log".format(model_dir, _iter, job)
                if os.path.exists(log_file):
                    os.rename(log_file, "{0}.{1}".format(log_file, try_count))
                training_flag = True
                cnt += 1
        if training_flag:
            if try_count < MAX_TRY_COUNT:
                logger.warn("{0}/{1} of jobs failed. Resubmitting them may solved the problem. "
                            "Start resubmitting them after 30 seconds ...".format(cnt, num_jobs))
                # sleep for 30 seconds before resubmitting
                time.sleep(30)
            else:
                logger.error("{0}/{1} of jobs failed and maximum number of retrying is reached. "
                             "Stop the training ...".format(cnt, num_jobs))

    if training_flag:
        raise Exception("Some training jobs failed more than %d times. "
                        "Please check the log files." % MAX_TRY_COUNT)

    [models_to_average, best_model] = utils.get_successful_models(
        num_jobs, '{0}/log/train.{1}.%.log'.format(model_dir, _iter))

    if do_average and len(models_to_average) > 1:
        # average the output of the different jobs.
        # TODO
        nets_dirs = []
        for n in models_to_average:
            nets_dirs.append("{0}/model_{1}.{2}".format(model_dir, _iter + 1, n))
        utils.get_average_nnet_model(
            dir=model_dir, iter=_iter,
            nnets_list=" ".join(nets_dirs),
            run_opts=run_opts)
    else:
        # choose the best model from different jobs
        utils.copy_best_nnet_dir(_dir=model_dir, _iter=_iter, best_model_index=best_model)

    try:
        for i in range(1, num_jobs + 1):
            shutil.rmtree("{0}/model_{1}.{2}".format(model_dir, _iter + 1, i))
    except OSError:
        logger.error("Error while trying to delete the client models.")
        raise

    if not os.path.isfile(new_model):
        raise Exception("Could not find {0}, at the end of iteration {1}".format(new_model, _iter))
    elif os.stat(new_model).st_size == 0:
        raise Exception("{0} has size 0. Something went wrong in iteration {1}".format(new_model, _iter))


def eval_trained_dnn(main_dir, _iter, egs_dir, run_opts):
    input_model_dir = "{dir}/model_{iter}".format(dir=main_dir, iter=_iter)

    # we assume that there are just one tar file for validation
    tar_file = ("{0}/valid_egs.1.tar".format(egs_dir))

    _command = '{command} "{main_dir}/log/compute_prob_valid.{iter}.log" ' \
               'local/tf/eval_dnn.py ' \
               '--tar-file="{tar_file}" --use-gpu=no ' \
               '--log-file="{main_dir}/log/compute_prob_valid.{iter}.log" ' \
               '--input-dir="{input_model_dir}"'.format(command=run_opts.command,
                                                        main_dir=main_dir,
                                                        iter=_iter,
                                                        tar_file=tar_file,
                                                        input_model_dir=input_model_dir)

    utils.background_command(_command)

    # we assume that there are just one tar file for train diagnostics
    tar_file = ("{0}/train_subset_egs.1.tar".format(egs_dir))

    _command = '{command} "{main_dir}/log/compute_prob_train_subset.{iter}.log" ' \
               'local/tf/eval_dnn.py ' \
               '--tar-file="{tar_file}" --use-gpu=no ' \
               '--log-file="{main_dir}/log/compute_prob_train_subset.{iter}.log" ' \
               '--input-dir="{input_model_dir}"'.format(command=run_opts.command,
                                                        main_dir=main_dir,
                                                        iter=_iter,
                                                        tar_file=tar_file,
                                                        input_model_dir=input_model_dir)

    utils.background_command(_command)


def train(args, run_opts):
    """ The main function for training.

    Args:
        args: a Namespace object with the required parameters
            obtained from the function process_args()
        run_opts: RunOpts object obtained from the process_args()
    """

    arg_string = pprint.pformat(vars(args))
    logger.info("Arguments for the experiment\n{0}".format(arg_string))

    default_egs_dir = '{0}/egs'.format(args.dir)
    if args.egs_dir is None:
        egs_dir = default_egs_dir
    else:
        egs_dir = args.egs_dir

    [num_archives, egs_feat_dim, archives_minibatch_count] = utils.verify_egs_dir(egs_dir)

    if args.num_jobs_final > num_archives:
        raise Exception('num_jobs_final cannot exceed the number of archives '
                        'in the egs directory')

    if args.stage <= -1:
        if not os.path.exists('{0}/model_0/done'.format(args.dir)):
            logger.info("Preparing the initial network.")
            num_classes = args.num_targets
            model_name = args.tf_model_class
            model = eval('models.%s()' % model_name)
            logger.info("Start calling build_model to initialize the model %s ..." % model_name)
            model.build_model(num_classes, egs_feat_dim, '{0}/model_0'.format(args.dir), logger=logger)
            with open(os.path.join(args.dir, 'model_name.txt'), 'wt') as fid:
                fid.write(model_name)
        else:
            logger.info("The initial network exist from before.")

    # set num_iters so that as close as possible, we process the data
    # $num_epochs times, i.e. $num_iters * $avg_num_jobs) ==
    # $num_epochs * $num_archives, where
    # avg_num_jobs = (num_jobs_initial + num_jobs_final) / 2.
    num_archives_to_process = int(args.num_epochs * num_archives)
    num_archives_processed = 0
    num_iters = ((num_archives_to_process * 2) / (args.num_jobs_initial + args.num_jobs_final))

    # If do_final_combination is True, compute the set of models_to_combine.
    # Otherwise, models_to_combine will be none.
    if args.do_final_combination:
        models_to_combine = utils.get_model_combine_iters(num_iters, num_archives, args.max_models_combine,
                                                          args.num_jobs_final)
    else:
        models_to_combine = None

    logger.info("Training will run for {0} epochs = {1} iterations".format(args.num_epochs, num_iters))

    for _iter in range(num_iters):
        current_num_jobs = int(0.5 + args.num_jobs_initial
                               + (args.num_jobs_final - args.num_jobs_initial)
                               * float(_iter) / num_iters)

        if args.stage <= _iter:
            lrate = utils.get_learning_rate(_iter, current_num_jobs,
                                            num_iters,
                                            num_archives_processed,
                                            num_archives_to_process,
                                            args.initial_effective_lrate,
                                            args.final_effective_lrate)

            shrinkage_value = 1.0 - (args.proportional_shrink * lrate)
            if shrinkage_value <= 0.5:
                raise Exception("proportional-shrink={0} is too large, it gives "
                                "shrink-value={1}".format(args.proportional_shrink,
                                                          shrinkage_value))

            percent = num_archives_processed * 100.0 / num_archives_to_process
            epoch = (num_archives_processed * args.num_epochs / num_archives_to_process)
            shrink_info_str = ''
            if shrinkage_value != 1.0:
                shrink_info_str = 'shrink: {0:0.5f}'.format(shrinkage_value)
            logger.info("Iter: {0}/{1}    Epoch: {2:0.2f}/{3:0.1f} ({4:0.1f}% complete)    "
                        "lr: {5:0.6f}    {6}".format(_iter, num_iters - 1, epoch, args.num_epochs,
                                                     percent, lrate, shrink_info_str))
            current_dropout = utils.get_dropout_edit_string(
                args.dropout_schedule, float(num_archives_processed) / num_archives_to_process)
            train_one_iteration(
                model_dir=args.dir,
                _iter=_iter,
                random_seed=args.random_seed,
                egs_dir=egs_dir,
                num_jobs=current_num_jobs,
                num_archives_processed=num_archives_processed,
                num_archives=num_archives,
                learning_rate=lrate,
                current_dropout=current_dropout,
                minibatch_size=args.minibatch_size,
                momentum=args.momentum,
                max_param_change=args.max_param_change,
                shrinkage_value=shrinkage_value,
                feature_dim=egs_feat_dim,
                archives_minibatch_count=archives_minibatch_count,
                run_opts=run_opts)

            if args.cleanup:
                # do a clean up everything but the last 2 models, under certain conditions
                utils.remove_model(args.dir, _iter - 2, models_to_combine, args.preserve_model_interval)

        num_archives_processed = num_archives_processed + current_num_jobs

    if args.stage <= num_iters:
        # TODO now do_final_combination is false but the default value was True
        if args.do_final_combination:
            logger.info("Doing final combination to produce final.raw")
            raise Exception('combine model using average not implemented yet.')
            # train_lib.common.combine_models(
            #     dir=args.dir, num_iters=num_iters,
            #     models_to_combine=models_to_combine, egs_dir=egs_dir,
            #     minibatch_size_str=args.minibatch_size, run_opts=run_opts,
            #     get_raw_nnet_from_am=False,
            #     max_objective_evaluations=args.max_objective_evaluations)
        else:
            utils.force_symlink("model_{1}".format(args.dir, num_iters), "{0}/model_final".format(args.dir))

    if args.cleanup:
        logger.info("Cleaning up the experiment directory {0}".format(args.dir))
        for _iter in range(num_iters):
            utils.remove_model(args.dir, _iter, models_to_combine, args.preserve_model_interval)

    # do some reporting
    [report, _, _] = utils.generate_report(args.dir)
    with open("{dir}/accuracy.report".format(dir=args.dir), "wt") as fid:
        fid.write(report)


def main():
    [args, run_opts] = get_args()
    try:
        train(args, run_opts)
        utils.wait_for_background_commands()
    except BaseException as e:
        # look for BaseException so we catch KeyboardInterrupt, which is
        # what we get when a background thread dies.
        if not isinstance(e, KeyboardInterrupt):
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
