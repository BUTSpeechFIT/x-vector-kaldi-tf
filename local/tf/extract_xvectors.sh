 
#!/bin/bash

# Copyright     2018 Hossein Zeinali (Brno University of Technology)

# Apache 2.0.

# This script extracts embeddings (called "xvectors" here) from a set of
# utterances, given features and a trained DNN.  The purpose of this script
# is analogous to sid/extract_ivectors.sh: it creates archives of
# vectors that are used in speaker recognition.  Like ivectors, xvectors can
# be used in PLDA or a similar backend for scoring.

# Begin configuration section.
nj=30
cmd="run.pl"

chunk_size=-1     # The chunk size over which the embedding is extracted.
                  # If left unspecified, it uses the max_chunk_size in the nnet directory.
use_gpu=false
stage=0

echo "${0} $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "Usage: ${0} <nnet-dir> <data> <xvector-dir>"
  echo " e.g.: ${0} exp/xvector_nnet data/train exp/xvectors_train"
  echo "main options (for others, see top of script file)"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --use-gpu <bool|false>                           # If true, use GPU."
  echo "  --nj <n|30>                                      # Number of jobs"
  echo "  --stage <stage|0>                                # To control partial reruns"
  echo "  --chunk-size <n|-1>                              # If provided, extracts embeddings with specified"
  echo "                                                   # chunk size, and averages to produce final embedding"
fi

srcdir=$1
data=$2
dir=$3

for f in ${srcdir}/model_final/model.meta ${srcdir}/min_chunk_size ${srcdir}/max_chunk_size ${data}/feats.scp ${data}/vad.scp ; do
  [ ! -f ${f} ] && echo "No such file $f" && exit 1;
done

min_chunk_size=`cat ${srcdir}/min_chunk_size 2>/dev/null`
max_chunk_size=`cat ${srcdir}/max_chunk_size 2>/dev/null`

model_dir=${srcdir}/model_final

if [ ${chunk_size} -le 0 ]; then
  chunk_size=${max_chunk_size}
fi

if [ ${max_chunk_size} -lt ${chunk_size} ]; then
  echo "${0}: specified chunk size of ${chunk_size} is larger than the maximum chunk size, ${max_chunk_size}" && exit 1;
fi

mkdir -p ${dir}/log

utils/split_data.sh --per-utt ${data} ${nj}
echo "${0}: extracting xvectors for ${data}"
sdata=${data}/split${nj}utt/JOB

# Set up the features
feature_rspecifier="apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 scp:${sdata}/feats.scp ark:- | select-voiced-frames ark:- scp,s,cs:${sdata}/vad.scp ark:- |"

if [ ${stage} -le 0 ]; then
  echo "${0}: extracting xvectors from nnet"
  if ${use_gpu}; then
    for g in $(seq ${nj}); do
      ${cmd} --gpu 1 "${dir}/log/extract.${g}.log" \
        local/tf/extract_embedding.py \
          --use-gpu=no --min-chunk-size=${min_chunk_size} --chunk-size=${chunk_size} \
          --feature-rspecifier="`echo ${feature_rspecifier} | sed s/JOB/${g}/g`" \
          --vector-wspecifier="| copy-vector ark:- ark,scp:${dir}/xvector.${g}.ark,${dir}/xvector.${g}.scp" \
          --model-dir="${model_dir}" || exit 1 &
    done
    wait
  else
    ${cmd} JOB=1:${nj} "${dir}/log/extract.JOB.log" \
      local/tf/extract_embedding.py \
        --use-gpu=no --min-chunk-size=${min_chunk_size} --chunk-size=${chunk_size} \
        --feature-rspecifier="${feature_rspecifier}" \
        --vector-wspecifier="| copy-vector ark:- ark,scp:${dir}/xvector.JOB.ark,${dir}/xvector.JOB.scp" \
        --model-dir="${model_dir}" || exit 1;
  fi
fi

if [ ${stage} -le 1 ]; then
  echo "${0}: combining xvectors across jobs"
  for j in $(seq ${nj}); do cat ${dir}/xvector.${j}.scp; done > ${dir}/xvector.scp || exit 1;
fi

if [ ${stage} -le 2 ]; then
  # Average the utterance-level xvectors to get speaker-level xvectors.
  echo "${0}: computing mean of xvectors for each speaker"
  run.pl ${dir}/log/speaker_mean.log \
    ivector-mean ark:${data}/spk2utt scp:${dir}/xvector.scp \
    ark,scp:${dir}/spk_xvector.ark,${dir}/spk_xvector.scp ark,t:${dir}/num_utts.ark || exit 1;
fi
