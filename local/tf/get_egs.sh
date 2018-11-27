#!/bin/bash

# Copyright      2018   Hossein Zeinali (Brno University of Technology)
#                2017   David Snyder
#                2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#                2017   Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0
#
# This script dumps training examples (egs) for multiclass xvector training.
# These egs consist of a data chunk and a zero-based speaker label.
# Each archive of egs has, in general, a different input chunk-size.
# We don't mix together different lengths in the same archive, because it
# would require us to repeatedly run the compilation process within the same
# training job.
#
# This script, which will generally be called from other neural net training
# scripts, extracts the training examples used to train the neural net (and
# also the validation examples used for diagnostics), and puts them in
# separate archives.


# Begin configuration section.
cmd=run.pl

stage=0
nj=6         # This should be set to the maximum number of jobs you are
             # comfortable to run in parallel; you can increase it if your disk
             # speed is greater and you have more machines.

# each minibatch of archives has data-chunks off length randomly chosen between
# $min_frames_per_eg and $max_frames_per_eg.
min_frames_per_chunk=200
max_frames_per_chunk=400
frames_per_iter=10000000 # target number of frames per archive.
                         # This parameter determine number of archives

frames_per_iter_diagnostic=100000 # have this many frames per archive for
                                  # the archives used for diagnostics.

num_diagnostic_archives=1   # we want to test the training likelihoods using a training subset
                            # as well as validation set. In our implementation this number should
                            # be one because we just used the indexed 1 tar file.

num_heldout_utts=200        # number of utterances held out for validation subset

num_repeats=10 # number of times each speaker repeats per archive
minibatch_size=128

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 2 ]; then
  echo "Usage: $0 [opts] <data> <egs-dir>"
  echo " e.g.: $0 data/train exp/xvector_a/egs"
  echo ""
  echo "Main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config file containing options"
  echo "  --nj <nj>                                        # The maximum number of jobs you want to run in"
  echo "                                                   # parallel (increase this only if you have good disk and"
  echo "                                                   # network speed).  default=6"
  echo "  --cmd (utils/run.pl;utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --min-frames-per-chunk <#frames;200>             # The minimum number of frames per chunk that we dump"
  echo "  --max-frames-per-chunk <#frames;400>             # The maximum number of frames per chunk that we dump"
  echo "  --num-repeats <#repeats;1>                       # The (approximate) number of times the training"
  echo "                                                   # data is repeated in the egs"
  echo "  --frames-per-iter <#samples;1000000>             # Target number of frames per archive"
  echo "  --num-diagnostic-archives <#archives;1>          # Option that controls how many different versions of"
  echo "                                                   # the train and validation archives we create (e.g."
  echo "                                                   # train_subset.{1,2,3}.egs and valid.{1,2,3}.egs by default;"
  echo "                                                   # they contain different utterance lengths."
  echo "  --frames-per-iter-diagnostic <#samples;100000>   # Target number of frames for the diagnostic archives"
  echo "                                                   # {train_subset,valid}.*.egs"
  echo "  --stage <stage|0>                                # Used to run a partially-completed training process from somewhere in"
  echo "                                                   # the middle."
  echo "  --minibatch-size                                 # Size of the minibatch used in DNN training."

  exit 1;
fi

data=$1
egs_dir=$2

for f in ${data}/utt2num_frames ${data}/feats.scp ; do
  [ ! -f ${f} ] && echo "$0: expected file ${f}" && exit 1;
done

feat_dim=$(feat-to-dim scp:${data}/feats.scp -) || exit 1

mkdir -p ${egs_dir}/info ${egs_dir}/temp
temp=${egs_dir}/temp

echo ${feat_dim} > ${egs_dir}/info/feat_dim
cp ${data}/utt2num_frames ${egs_dir}/temp/utt2num_frames

if [ ${stage} -le 0 ]; then
  echo "$0: Preparing train and validation lists"
  # Pick a list of heldout utterances for validation egs
  awk '{print $1}' ${data}/utt2spk | utils/shuffle_list.pl | head -${num_heldout_utts} > ${temp}/valid_uttlist || exit 1;
  # The remaining utterances are used for training egs
  utils/filter_scp.pl --exclude ${temp}/valid_uttlist ${temp}/utt2num_frames > ${temp}/utt2num_frames.train
  utils/filter_scp.pl ${temp}/valid_uttlist ${temp}/utt2num_frames > ${temp}/utt2num_frames.valid
  # Pick a subset of the training list for diagnostics
  awk '{print $1}' ${temp}/utt2num_frames.train | utils/shuffle_list.pl | head -${num_heldout_utts} > ${temp}/train_subset_uttlist || exit 1;
  utils/filter_scp.pl ${temp}/train_subset_uttlist < ${temp}/utt2num_frames.train > ${temp}/utt2num_frames.train_subset
  # Create a mapping from utterance to speaker ID (an integer)
  awk -v id=0 '{print $1, id++}' ${data}/spk2utt > ${temp}/spk2int
  utils/sym2int.pl -f 2 ${temp}/spk2int ${data}/utt2spk > ${temp}/utt2int
  utils/filter_scp.pl ${temp}/utt2num_frames.train ${temp}/utt2int > ${temp}/utt2int.train
  utils/filter_scp.pl ${temp}/utt2num_frames.valid ${temp}/utt2int > ${temp}/utt2int.valid
  utils/filter_scp.pl ${temp}/utt2num_frames.train_subset ${temp}/utt2int > ${temp}/utt2int.train_subset
fi

# first for the training data... work out how many archives.
num_train_frames=$(awk '{n += $2} END{print n}' <${temp}/utt2num_frames.train)
num_train_subset_frames=$(awk '{n += $2} END{print n}' <${temp}/utt2num_frames.train_subset)

echo ${num_train_frames} >${egs_dir}/info/num_frames
num_train_archives=$[(${num_train_frames}*${num_repeats})/${frames_per_iter} + 1]
echo "$0: Producing $num_train_archives archives for training"
echo ${num_train_archives} > ${egs_dir}/info/num_archives
echo ${num_diagnostic_archives} > ${egs_dir}/info/num_diagnostic_archives

if [ ${stage} -le 1 ]; then
  echo "${0}: Allocating training examples"
  ${cmd} ${egs_dir}/log/allocate_examples_train.log \
    local/tf/create_egs.py \
      --num-repeats=${num_repeats} --num-jobs=${nj} \
      --minibatch-size=${minibatch_size} \
      --min-frames-per-chunk=${min_frames_per_chunk} \
      --max-frames-per-chunk=${max_frames_per_chunk} \
      --frames-per-iter=${frames_per_iter} \
      --num-archives=${num_train_archives} \
      --utt2len-filename=${egs_dir}/temp/utt2num_frames.train \
      --utt2int-filename=${egs_dir}/temp/utt2int.train \
      --egs-dir=${egs_dir} || exit 1

  echo "${0}: Allocating training subset examples"
  ${cmd} ${egs_dir}/log/allocate_examples_train_subset.log \
    local/tf/create_egs.py \
      --prefix=train_subset \
      --num-repeats=8 --num-jobs=1 \
      --min-frames-per-chunk=${min_frames_per_chunk} \
      --max-frames-per-chunk=${max_frames_per_chunk} \
      --randomize-chunk-length=false \
      --frames-per-iter=${frames_per_iter_diagnostic} \
      --num-archives=${num_diagnostic_archives} \
      --utt2len-filename=${egs_dir}/temp/utt2num_frames.train_subset \
      --utt2int-filename=${egs_dir}/temp/utt2int.train_subset \
      --egs-dir=${egs_dir} || exit 1

  echo "${0}: Allocating validation examples"
  ${cmd} ${egs_dir}/log/allocate_examples_valid.log \
    local/tf/create_egs.py \
      --prefix=valid \
      --num-repeats=8 --num-jobs=1 \
      --min-frames-per-chunk=${min_frames_per_chunk} \
      --max-frames-per-chunk=${max_frames_per_chunk} \
      --randomize-chunk-length=false \
      --frames-per-iter=${frames_per_iter_diagnostic} \
      --num-archives=${num_diagnostic_archives} \
      --utt2len-filename=${egs_dir}/temp/utt2num_frames.valid \
      --utt2int-filename=${egs_dir}/temp/utt2int.valid \
      --egs-dir=${egs_dir} || exit 1

fi

if [ ${stage} -le 2 ]; then

  echo "${0}: Filtering feats.scp for training archives"
  for g in $(seq ${num_train_archives}); do
    utils/filter_scp.pl ${temp}/ranges.${g} ${data}/feats.scp > ${temp}/feats.scp.${g}
  done

  echo "${0}: Filtering feats.scp for validation and train subset archives"
  for g in $(seq ${num_diagnostic_archives}); do
    utils/filter_scp.pl ${temp}/train_subset_ranges.${g} ${data}/feats.scp > ${temp}/train_subset_feats.scp.${g}
    utils/filter_scp.pl ${temp}/valid_ranges.${g} ${data}/feats.scp > ${temp}/valid_feats.scp.${g}
  done

fi

# At this stage we'll have created the ranges files that define how many egs
# there are and where they come from.  If this is your first time running this
# script, you might decide to put an exit 1 command here, and inspect the
# contents of exp/$dir/temp/ranges.* before proceeding to the next stage.

if [ ${stage} -le 3 ]; then
  random_seed=2468
  echo "$0: Generating training examples on disk"
  rm ${egs_dir}/.error 2>/dev/null
  for g in $(seq ${nj}); do
    ${cmd} ${egs_dir}/log/train_create_examples.${g}.log \
      local/tf/create_tar_files.py \
        --random-seed=${random_seed} \
        --feature-dim=${feat_dim} \
        --minibatch-size=${minibatch_size} \
        --outputs-file=${temp}/outputs.${g} \
        --shuffle=True \
        --egs-dir=${egs_dir} || touch ${egs_dir}/.error &
  done

  echo "$0: Generating training subset examples on disk"
  ${cmd} ${egs_dir}/log/train_subset_create_examples.log \
    local/tf/create_tar_files.py \
      --prefix=train_subset \
      --random-seed=${random_seed} \
      --feature-dim=${feat_dim} \
      --minibatch-size=${minibatch_size} \
      --outputs-file=${temp}/train_subset_outputs.1 \
      --shuffle=True \
      --egs-dir=${egs_dir} || touch ${egs_dir}/.error &
  wait

  echo "$0: Generating validation examples on disk"
  ${cmd} ${egs_dir}/log/valid_create_examples.log \
    local/tf/create_tar_files.py \
      --prefix=valid \
      --random-seed=${random_seed} \
      --feature-dim=${feat_dim} \
      --minibatch-size=${minibatch_size} \
      --outputs-file=${temp}/valid_outputs.1 \
      --shuffle=True \
      --egs-dir=${egs_dir} || touch ${egs_dir}/.error &
  wait

  if [ -f ${egs_dir}/.error ]; then
    echo "$0: Problem detected while dumping examples."
    exit 1
  fi
fi

echo "$0: Finished preparing training examples"

exit 0
