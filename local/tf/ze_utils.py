import argparse
import inspect
import logging
import math
import os
import re
import shutil
import subprocess
import threading
import thread
import traceback

import datetime

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

cuda_command = 'nvidia-smi --query-gpu=memory.free,memory.total --format=csv | tail -n+2 | ' \
               'awk \'BEGIN{FS=" "}{if ($1/$3 > 0.98) print NR-1}\''

cuda_command2 = 'nvidia-smi -q | grep "Minor\|Processes" | grep "None" -B1 | tr -d " " | cut -d ":" -f2 | sed -n "1p"'
gpu_used_pid = 'nvidia-smi -q | grep "Process ID" | tr -d " " | cut -d ":" -f2'


def set_cuda_visible_devices(use_gpu=True, logger=None):
    try:
        if use_gpu:
            free_gpu = subprocess.check_output(cuda_command2, shell=True)
            if len(free_gpu) == 0:
                create_log_on_gpu_error()
                if logger is not None:
                    logger.info("No GPU seems to be available and I cannot continue without GPU.")
                raise Exception("No GPU seems to be available and I cannot continue without GPU.")
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = free_gpu.decode().strip()
            if logger is not None:
                logger.info("CUDA_VISIBLE_DEVICES " + os.environ["CUDA_VISIBLE_DEVICES"])
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ''
    except subprocess.CalledProcessError:
        if logger is not None:
            create_log_on_gpu_error()
            logger.info("No GPU seems to be available and I cannot continue without GPU.")
        # os.environ["CUDA_VISIBLE_DEVICES"] = ''
        if use_gpu:
            raise


def print_function_args_values(frame):
    args, _, _, values = inspect.getargvalues(frame)
    print('Function name "%s"' % inspect.getframeinfo(frame)[2])
    for arg in args:
        print("    %s = %s" % (arg, values[arg]))


def verify_egs_dir(egs_dir):
    try:
        egs_feat_dim = int(open('{0}/info/feat_dim'.format(egs_dir)).readline())

        num_archives = int(open('{0}/info/num_archives'.format(egs_dir)).readline())

        archives_minibatch_count = {}
        with open('{0}/temp/archive_minibatch_count'.format(egs_dir), 'rt') as fid:
            for line in fid:
                if len(line.strip()) == 0:
                    continue
                parts = line.split()
                archives_minibatch_count[int(parts[0])] = int(parts[1])

        return [num_archives, egs_feat_dim, archives_minibatch_count]
    except (IOError, ValueError):
        logger.error("The egs dir {0} has missing or malformed files.".format(egs_dir))
        raise


def get_model_combine_iters(num_iters, num_archives, max_models_combine, num_jobs_final):
    """ Figures out the list of iterations for which we'll use those models
        in the final model-averaging phase.  (note: it's a weighted average
        where the weights are worked out from a subset of training data.)"""

    approx_iters_per_epoch_final = num_archives / num_jobs_final
    # Note: it used to be that we would combine over an entire epoch,
    # but in practice we very rarely would use any weights from towards
    # the end of that range, so we are changing it to use not
    # approx_iters_per_epoch_final, but instead:
    # approx_iters_per_epoch_final/2 + 1,
    # dividing by 2 to use half an epoch, and adding 1 just to make sure
    # it's not zero.

    # First work out how many iterations we want to combine over in the final
    # nnet3-combine-fast invocation.
    # The number we use is:
    # min(max(max_models_combine, approx_iters_per_epoch_final/2+1), iters/2)
    # But if this value is > max_models_combine, then the models
    # are sub-sampled to get these many models to combine.

    num_iters_combine_initial = min(approx_iters_per_epoch_final / 2 + 1, num_iters / 2)

    if num_iters_combine_initial > max_models_combine:
        subsample_model_factor = int(float(num_iters_combine_initial) / max_models_combine)
        models_to_combine = set(range(num_iters - num_iters_combine_initial + 1,
                                      num_iters + 1, subsample_model_factor))
        models_to_combine.add(num_iters)
    else:
        num_iters_combine = min(max_models_combine, num_iters / 2)
        models_to_combine = set(range(num_iters - num_iters_combine + 1, num_iters + 1))

    return models_to_combine


def get_learning_rate(_iter, num_jobs, num_iters, num_archives_processed, num_archives_to_process,
                      initial_effective_lrate, final_effective_lrate):
    if _iter + 1 >= num_iters:
        effective_learning_rate = final_effective_lrate
    else:
        effective_learning_rate = (initial_effective_lrate *
                                   math.exp(num_archives_processed *
                                            math.log(final_effective_lrate / initial_effective_lrate)
                                            / num_archives_to_process))
    return num_jobs * effective_learning_rate


def get_successful_models(num_models, log_file_pattern, difference_threshold=1.0):
    assert num_models > 0

    parse_regex = re.compile(
        "INFO .* Overall average objective function is ([0-9e.\-+= ]+) over ([0-9e.\-+]+) segments")
    objectives = []
    for i in range(num_models):
        model_num = i + 1
        logfile = re.sub('%', str(model_num), log_file_pattern)
        lines = open(logfile, 'r').readlines()
        this_objective = -100000.0
        for line_num in range(1, len(lines) + 1):
            # we search from the end as this would result in
            # lesser number of regex searches. Python regex is slow !
            mat_obj = parse_regex.search(lines[-1 * line_num])
            if mat_obj is not None:
                this_objective = float(mat_obj.groups()[0].split()[-1])
                break
        objectives.append(this_objective)
    max_index = objectives.index(max(objectives))
    accepted_models = []
    for i in range(num_models):
        if (objectives[max_index] - objectives[i]) <= difference_threshold:
            accepted_models.append(i + 1)

    if len(accepted_models) != num_models:
        logger.warn("Only {0}/{1} of the models have been accepted "
                    "for averaging, based on log files {2}.".format(
            len(accepted_models),
            num_models, log_file_pattern))

    return [accepted_models, max_index + 1]


def copy_best_nnet_dir(_dir, _iter, best_model_index):
    best_model_dir = "{dir}/model_{next_iter}.{best_model_index}".format(
        dir=_dir, next_iter=_iter + 1, best_model_index=best_model_index)
    out_model_dir = "{dir}/model_{next_iter}".format(dir=_dir, next_iter=_iter + 1)
    shutil.copytree(best_model_dir, out_model_dir)


def get_average_nnet_model(dir, iter, nnets_list, run_opts,
                           get_raw_nnet_from_am=True):
    next_iter = iter + 1
    if get_raw_nnet_from_am:
        out_model = ("""- \| nnet3-am-copy --set-raw-nnet=-  \
                        {dir}/{iter}.mdl {dir}/{next_iter}.mdl""".format(
            dir=dir, iter=iter,
            next_iter=next_iter))
    else:
        out_model = "{dir}/{next_iter}.raw".format(
            dir=dir, next_iter=next_iter)

    # common_lib.execute_command(
    #     """{command} {dir}/log/average.{iter}.log \
    #             nnet3-average {nnets_list} \
    #             {out_model}""".format(command=run_opts.command,
    #                                   dir=dir,
    #                                   iter=iter,
    #                                   nnets_list=nnets_list,
    #                                   out_model=out_model))


def remove_model(nnet_dir, _iter, models_to_combine=None, preserve_model_interval=100):
    if _iter % preserve_model_interval == 0:
        return
    if models_to_combine is not None and _iter in models_to_combine:
        return
    model_dir = '{0}/model_{1}'.format(nnet_dir, _iter)

    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)


def background_command_waiter(command, popen_object, require_zero_status):
    """ This is the function that is called from background_command, in
        a separate thread."""

    popen_object.communicate()
    if popen_object.returncode is not 0:
        _str = "Command exited with status {0}: {1}".format(popen_object.returncode, command)
        if require_zero_status:
            logger.error(_str)
            # thread.interrupt_main() sends a KeyboardInterrupt to the main
            # thread, which will generally terminate the program.
            thread.interrupt_main()
        else:
            logger.warning(_str)


def background_command(command, require_zero_status=False):
    """Executes a command in a separate thread, like running with '&' in the shell.
       If you want the program to die if the command eventually returns with
       nonzero status, then set require_zero_status to True.  'command' will be
       executed in 'shell' mode, so it's OK for it to contain pipes and other
       shell constructs.

       This function returns the Thread object created, just in case you want
       to wait for that specific command to finish.  For example, you could do:
             thread = background_command('foo | bar')
             # do something else while waiting for it to finish
             thread.join()

       See also:
         - wait_for_background_commands(), which can be used
           at the end of the program to wait for all these commands to terminate.
         - execute_command() and get_command_stdout(), which allow you to
           execute commands in the foreground.
    """

    p = subprocess.Popen(command, shell=True)
    thread = threading.Thread(target=background_command_waiter, args=(command, p, require_zero_status))
    thread.daemon = True  # make sure it exits if main thread is terminated abnormally.
    thread.start()
    return thread


def wait_for_background_commands():
    """ This waits for all threads to exit.  You will often want to
        run this at the end of programs that have launched background
        threads, so that the program will wait for its child processes
        to terminate before it dies."""
    for t in threading.enumerate():
        if not t == threading.current_thread():
            t.join()


def force_symlink(file1, file2):
    import errno
    try:
        os.symlink(file1, file2)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.unlink(file2)
            os.symlink(file1, file2)


def str_to_bool(value):
    if value.lower() == "true":
        return True
    elif value.lower() == "false":
        return False
    else:
        raise ValueError


class StrToBoolAction(argparse.Action):
    """ A custom action to convert booleans from shell format i.e., true/false
        to python format i.e., True/False """

    def __call__(self, parser, namespace, values, option_string=None):
        try:
            setattr(namespace, self.dest, str_to_bool(values))
        except ValueError:
            raise Exception("Unknown value {0} for --{1}".format(values, self.dest))


class NullStrToNoneAction(argparse.Action):
    """ A custom action to convert empty strings passed by shell to None in
    python. This is necessary as shell scripts print null strings when a
    variable is not specified. We could use the more apt None in python. """

    def __call__(self, parser, namespace, values, option_string=None):
        if values.strip() == "":
            setattr(namespace, self.dest, None)
        else:
            setattr(namespace, self.dest, values)


class RunOpts(object):
    """A structure to store run options.

    Run options like queue.pl and run.pl, along with their memory
    and parallel training options for various types of commands such
    as the ones for training, parallel-training, running on GPU etc.
    """

    def __init__(self):
        self.command = None
        self.train_queue_opt = None
        self.combine_gpu_opt = None
        self.combine_queue_opt = None
        self.prior_gpu_opt = None
        self.prior_queue_opt = None
        self.parallel_train_opts = None


def _get_component_dropout(dropout_schedule, data_fraction):
    """Retrieve dropout proportion from schedule when data_fraction
    proportion of data is seen. This value is obtained by using a
    piecewise linear function on the dropout schedule.
    This is a module-internal function called by _get_dropout_proportions().

    See help for --trainer.dropout-schedule for how the dropout value
    is obtained from the options.

    Arguments:
        dropout_schedule: A list of (data_fraction, dropout_proportion) values
            sorted in descending order of data_fraction.
        data_fraction: The fraction of data seen until this stage of
            training.
    """
    if data_fraction == 0:
        # Dropout at start of the iteration is in the last index of
        # dropout_schedule
        assert dropout_schedule[-1][0] == 0
        return dropout_schedule[-1][1]
    try:
        # Find lower bound of the data_fraction. This is the
        # lower end of the piecewise linear function.
        (dropout_schedule_index, initial_data_fraction,
         initial_dropout) = next((i, tup[0], tup[1])
                                 for i, tup in enumerate(dropout_schedule)
                                 if tup[0] <= data_fraction)
    except StopIteration:
        raise RuntimeError(
            "Could not find data_fraction in dropout schedule "
            "corresponding to data_fraction {0}.\n"
            "Maybe something wrong with the parsed "
            "dropout schedule {1}.".format(data_fraction, dropout_schedule))

    if dropout_schedule_index == 0:
        assert dropout_schedule[0][0] == 1 and data_fraction == 1
        return dropout_schedule[0][1]

    # The upper bound of data_fraction is at the index before the
    # lower bound.
    final_data_fraction, final_dropout = dropout_schedule[
        dropout_schedule_index - 1]

    if final_data_fraction == initial_data_fraction:
        assert data_fraction == initial_data_fraction
        return initial_dropout

    assert (initial_data_fraction <= data_fraction < final_data_fraction)

    return ((data_fraction - initial_data_fraction)
            * (final_dropout - initial_dropout)
            / (final_data_fraction - initial_data_fraction)
            + initial_dropout)


def _parse_dropout_string(dropout_str):
    """Parses the dropout schedule from the string corresponding to a
    single component in --trainer.dropout-schedule.
    This is a module-internal function called by parse_dropout_function().

    Arguments:
        dropout_str: Specifies dropout schedule for a particular component
            name pattern.
            See help for the option --trainer.dropout-schedule.

    Returns a list of (data_fraction_processed, dropout_proportion) tuples
    sorted in descending order of num_archives_processed.
    A data fraction of 1 corresponds to all data.
    """
    dropout_values = []
    parts = dropout_str.strip().split(',')

    try:
        if len(parts) < 2:
            raise Exception("dropout proportion string must specify "
                            "at least the start and end dropouts")

        # Starting dropout proportion
        dropout_values.append((0, float(parts[0])))
        for i in range(1, len(parts) - 1):
            value_x_pair = parts[i].split('@')
            if len(value_x_pair) == 1:
                # Dropout proportion at half of training
                dropout_proportion = float(value_x_pair[0])
                data_fraction = 0.5
            else:
                assert len(value_x_pair) == 2

                dropout_proportion = float(value_x_pair[0])
                data_fraction = float(value_x_pair[1])

            if (data_fraction < dropout_values[-1][0]
                    or data_fraction > 1.0):
                logger.error(
                    "Failed while parsing value %s in dropout-schedule. "
                    "dropout-schedule must be in incresing "
                    "order of data fractions.", value_x_pair)
                raise ValueError

            dropout_values.append((data_fraction, float(dropout_proportion)))

        dropout_values.append((1.0, float(parts[-1])))
    except Exception:
        logger.error("Unable to parse dropout proportion string %s. "
                     "See help for option "
                     "--trainer.dropout-schedule.", dropout_str)
        raise

    # reverse sort so that its easy to retrieve the dropout proportion
    # for a particular data fraction
    dropout_values.reverse()
    for data_fraction, proportion in dropout_values:
        assert 0.0 <= data_fraction <= 1.0
        assert 0.0 <= proportion <= 1.0

    return dropout_values


def get_dropout_edit_string(dropout_schedule, data_fraction):
    """Returns dropout proportion based on the dropout_schedule for the
    fraction of data seen at this stage of training.
    Returns None if dropout_schedule is None.

    Arguments:
        dropout_schedule: Value for the --dropout-schedule option.
            See help for --dropout-schedule.
        data_fraction: The fraction of data seen until this stage of
            training.
    """
    if dropout_schedule is None:
        return None
    dropout_schedule = _parse_dropout_string(dropout_schedule)
    dropout_proportion = _get_component_dropout(dropout_schedule, data_fraction)
    return dropout_proportion


def get_command_stdout(command, require_zero_status=True):
    """ Executes a command and returns its stdout output as a string.  The
        command is executed with shell=True, so it may contain pipes and
        other shell constructs.

        If require_zero_stats is True, this function will raise an exception if
        the command has nonzero exit status.  If False, it just prints a warning
        if the exit status is nonzero.
    """
    p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)

    stdout = p.communicate()[0]
    if p.returncode is not 0:
        output = "Command exited with status {0}: {1}".format(p.returncode, command)
        if require_zero_status:
            raise Exception(output)
        else:
            logger.warning(output)
    return stdout if type(stdout) is str else stdout.decode()


def get_train_times(exp_dir):
    train_log_files = "%s/log/" % (exp_dir)
    train_log_names = "train.*.log"
    command = 'find {0} -name "{1}" | xargs grep -H -e Accounting'.format(train_log_files,train_log_names)
    train_log_lines = get_command_stdout(command, require_zero_status=False)
    parse_regex = re.compile(".*train\.([0-9]+)\.([0-9]+)\.log:# Accounting: time=([0-9]+) thread.*")

    train_times = dict()
    for line in train_log_lines.split('\n'):
        mat_obj = parse_regex.search(line)
        if mat_obj is not None:
            groups = mat_obj.groups()
            try:
                train_times[int(groups[0])][int(groups[1])] = float(groups[2])
            except KeyError:
                train_times[int(groups[0])] = {}
                train_times[int(groups[0])][int(groups[1])] = float(groups[2])
    iters = train_times.keys()
    for _iter in iters:
        values = train_times[_iter].values()
        train_times[_iter] = max(values)
    return train_times


def parse_prob_logs(exp_dir, key='accuracy'):
    train_prob_files = "%s/log/compute_prob_train_subset.*.log" % exp_dir
    valid_prob_files = "%s/log/compute_prob_valid.*.log" % exp_dir
    train_prob_strings = get_command_stdout('grep -e {0} {1}'.format(key, train_prob_files))
    valid_prob_strings = get_command_stdout('grep -e {0} {1}'.format(key, valid_prob_files))

    # Overall average loss is 0.6923 over 1536 segments. Also, the overall average accuracy is 0.8548.
    parse_regex = re.compile(".*compute_prob_.*\.([0-9]+).log.*Overall average ([a-zA-Z\-]+) is ([0-9.\-e]+) "
                             ".*overall average ([a-zA-Z\-]+) is ([0-9.\-e]+)\.")

    train_objf = {}
    valid_objf = {}
    for line in train_prob_strings.split('\n'):
        mat_obj = parse_regex.search(line)
        if mat_obj is not None:
            groups = mat_obj.groups()
            if groups[3] == key:
                train_objf[int(groups[0])] = (groups[2], groups[4])
    if not train_objf:
        raise Exception("Could not find any lines with {key} in {log}".format(key=key, log=train_prob_files))

    for line in valid_prob_strings.split('\n'):
        mat_obj = parse_regex.search(line)
        if mat_obj is not None:
            groups = mat_obj.groups()
            if groups[3] == key:
                valid_objf[int(groups[0])] = (groups[2], groups[4])

    if not valid_objf:
        raise Exception("Could not find any lines with {key} in {log}".format(key=key, log=valid_prob_files))

    iters = list(set(valid_objf.keys()).intersection(train_objf.keys()))
    if not iters:
        raise Exception("Could not any common iterations with key {k} in both {tl} and {vl}".format(
                    k=key, tl=train_prob_files, vl=valid_prob_files))
    iters.sort()
    return list(map(lambda x: (int(x), float(train_objf[x][0]), float(train_objf[x][1]),
                               float(valid_objf[x][0]), float(valid_objf[x][1])), iters))


def generate_report(exp_dir, key="accuracy"):
    try:
        times = get_train_times(exp_dir)
    except Exception:
        tb = traceback.format_exc()
        logger.warning("Error getting info from logs, exception was: " + tb)
        times = {}

    report = ["%Iter\tduration\ttrain_loss\tvalid_loss\tdifference\ttrain_acc\tvalid_acc\tdifference"]
    try:
        data = list(parse_prob_logs(exp_dir, key))
    except Exception:
        tb = traceback.format_exc()
        logger.warning("Error getting info from logs, exception was: " + tb)
        data = []
    for x in data:
        try:
            report.append("%d\t%s\t%g\t%g\t%g\t%g\t%g\t%g" %
                          (x[0], str(times[x[0]]), x[1], x[3], x[3]-x[1], x[2], x[4], x[2]-x[4]))
        except KeyError:
            continue

    total_time = 0
    for _iter in times.keys():
        total_time += times[_iter]
    report.append("Total training time is {0}\n".format(
                    str(datetime.timedelta(seconds=total_time))))
    return ["\n".join(report), times, data]


def is_correct_model_dir(model_dir):
    model_file = "{0}/model.meta".format(model_dir)
    done_file = "{0}/done".format(model_dir)
    if os.path.isfile(model_file) and os.stat(model_file).st_size > 0 and \
            os.path.isfile(done_file) and os.stat(done_file).st_size > 0:
        return True
    return False


def create_log_on_gpu_error():
    try:
        import xml.etree.ElementTree as ET
        from xml.etree.ElementTree import tostring
        import numpy as np
        print(os.uname()[1])
        uname = os.uname()[1]
        used_pid = subprocess.check_output(gpu_used_pid, shell=True).decode().strip().split('\n')
        command = 'ps -o user='
        for pid in used_pid:
            command += ' -p ' + pid
        users = subprocess.check_output(command, shell=True).decode().strip().split('\n')
        pid_user = {}
        for user, pid in zip(users, used_pid):
            pid_user[pid] = user

        s = os.popen("qstat -r -s r -xml").read()
        root = ET.fromstring(s)

        # users = [x.text for x in root.findall('.//JB_owner')]
        users = np.unique(users)

        user_gpu = {}
        for u in users:
            q = [jl for jl in root.findall(".//*[JB_owner='%s']" % u)]

            p = filter(lambda x: x.find('master').text == 'MASTER', q)

            p1 = filter(lambda x: x.find('queue_name').text.endswith(uname), p)

            hrs = [n.findall('hard_request') for n in p1]

            a = [x for subl in hrs for x in subl]
            v = [(x.get('name'), x.text) for x in a]

            for (n, h) in v:
                if n == 'gpu':
                    if u in user_gpu:
                        user_gpu[u] += int(h)
                    else:
                        user_gpu[u] = int(h)

        for pid, user in pid_user.iteritems():
            if user in user_gpu:
                if user_gpu[user] > 0:
                    print("%-12s%-30s OK" % (pid, user))
                else:
                    print("%-12s%-30s Get GPU more than request" % (pid, user))
                user_gpu[user] -= 1
            else:
                print("%-12s%-30s No Request for GPU" % (pid, user))

    except Exception as exp:
        print(exp)


if __name__ == '__main__':
    create_log_on_gpu_error()
