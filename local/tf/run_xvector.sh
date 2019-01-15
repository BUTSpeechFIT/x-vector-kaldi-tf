#!/bin/bash
# Copyright      2018   Hossein Zeinali (Brno University of Technology)
#                2017   David Snyder
#                2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#                2017   Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.

train_cmd=

. ./cmd.sh
set -e

stage=1
train_stage=0
use_gpu=true

minibatch_size=64
data=data/train
nnet_dir=exp/xvector_nnet_1ha/
egs_dir=exp/xvector_nnet_1ha/egs

. ./path.sh
. ./utils/parse_options.sh

# Now we create the nnet examples using sid/nnet3/xvector/get_egs.sh.
# The argument --num-repeats is related to the number of times a speaker
# repeats per archive.  If it seems like you're getting too many archives
# (e.g., more than 200) try increasing the --frames-per-iter option.  The
# arguments --min-frames-per-chunk and --max-frames-per-chunk specify the
# minimum and maximum length (in terms of number of frames) of the features
# in the examples.
#
# To make sense of the egs script, it may be necessary to put an "exit 1"
# command immediately after stage 3.  Then, inspect
# exp/<your-dir>/egs/temp/ranges.* . The ranges files specify the examples that
# will be created, and which archives they will be stored in.  Each line of
# ranges.* has the following form:
#    <utt-id> <local-ark-indx> <global-ark-indx> <start-frame> <end-frame> <spk-id>
# For example:
#    100304-f-sre2006-kacg-A 1 2 4079 881 23

# If you're satisfied with the number of archives (e.g., 50-150 archives is
# reasonable) and with the number of examples per speaker (e.g., 1000-5000
# is reasonable) then you can let the script continue to the later stages.
# Otherwise, try increasing or decreasing the --num-repeats option.  You might
# need to fiddle with --frames-per-iter.  Increasing this value decreases the
# the number of archives and increases the number of examples per archive.
# Decreasing this value increases the number of archives, while decreasing the
# number of examples per archive.
if [ ${stage} -le 4 ]; then
  echo "$0: Getting neural network training egs";
  # dump egs.
  local/tf/get_egs.sh --cmd "${train_cmd} --long 1 --scratch 1" \
    --nj 10 \
    --stage 0 \
    --frames-per-iter 1000000000 \
    --frames-per-iter-diagnostic 1000000 \
    --min-frames-per-chunk 200 \
    --max-frames-per-chunk 400 \
    --num-diagnostic-archives 3 \
    --minibatch-size ${minibatch_size} \
    --num-repeats 35 \
    "${data}" "${egs_dir}"
fi
# This chunk-size corresponds to the maximum number of frames the
# stats layer is able to pool over.  In this script, it corresponds
# to 100 seconds.  If the input recording is greater than 100 seconds,
# we will compute multiple xvectors from the same recording and average
# to produce the final xvector.
max_chunk_size=10000

# The smallest number of frames we're comfortable computing an xvector from.
# Note that the hard minimum is given by the left and right context of the
# frame-level layers.
min_chunk_size=25

mkdir -p ${nnet_dir}
echo "${max_chunk_size}" > ${nnet_dir}/max_chunk_size
echo "${min_chunk_size}" > ${nnet_dir}/min_chunk_size


#dropout_schedule='0,0@0.16,0.25@0.32,0.25@0.64,0@0.96,0'
dropout_schedule='0,0@0.10,0.1@0.50,0'

random_seed=2468
num_targets=$(wc -w ${egs_dir}/pdf2num | awk '{print $1}')
if [ ${stage} -le 6 ]; then
  local/tf/train_dnn.py \
    --stage=${train_stage} \
    --tf-model-class="ModelWithoutDropout" \
    --cmd="${train_cmd} --long 0" \
    --num-targets=${num_targets} \
    --proportional-shrink=10 \
    --minibatch-size=${minibatch_size} \
    --max-param-change=2 \
    --momentum=0.5 \
    --num-jobs-initial=1 \
    --num-jobs-final=1 \
    --initial-effective-lrate=0.001 \
    --final-effective-lrate=0.0001 \
    --random-seed=${random_seed} \
    --num-epochs=2 \
    --dropout-schedule="$dropout_schedule" \
    --egs-dir="${egs_dir}" \
    --preserve-model-interval=10 \
    --use-gpu=yes \
    --dir=${nnet_dir}  || exit 1;
fi

exit 0;
