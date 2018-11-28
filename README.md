# Tensorflow implementation of x-vector topology on top of Kaldi recipe

This is a Tensorflow implementation of x-vector topology (speaker embedding) which was proposed by David Snyder in [Deep Neural Network Embeddings for Text-Independent Speaker Verification](http://www.danielpovey.com/files/2017_interspeech_embeddings.pdf). It uses Kaldi for data processing, feature extraction, data augmentation, and VAD. Also, the back-end is Kaldi PLDA. Here we just train the model using Tensorflow and also extract speaker embeddings (x-vectors) using it and save them in Kaldi format. So, the overall x-vector pipeline is the same as the original Kaldi except training and x-vector extraction.

In this version, we create a different format training archives for Tensorflow. In the future, we will add the ability to use Kaldi training archives. The first steps of run.sh to end of stage 3 is same as Kaldi run.sh script and training archive generation starts from stage 4.

Some parts of this codes actually are modified version of scripts from Kaldi. Also, we have used the following repository as an initial Tensorflow implementation:
https://github.com/qqueing/SR_with_kaldi

Also, for reading and writing data in Kaldi format following Kaldi-io repository is used.
https://github.com/vesis84/kaldi-io-for-python

## Usage
For using the codes, you first should install Kaldi and clone the codes in egs/sre16 (or somewhere else that you want, by changing the symlinks to proper positions). You can use the previously processed data and start the recipe from stage 4 (archive generation).

If you want to test different topology, you can add a subclass of Model to local/tf/models.py and overwrite the build_model function (in the file you can see several topologies). Then you just need to pass class name to  local/tf/train_dnn.py by changing --tf-model-class in local/tf/run_xvector.sh.

You can see some comparison of this implementation with Kaldi in terms of performance in the following paper. Also, you can find the effects of different tricks on the performance. We tried to keep all the useful tricks from the paper in the codes.
```
@article{zeinali2018improve,
  title={How to Improve Your Speaker Embeddings Extractor in Generic Toolkits},
  author={Zeinali, Hossein and Burget, Lukas and Rohdin, Johan and Stafylakis, Themos and Cernocky, Jan},
  journal={arXiv preprint arXiv:1811.02066},
  year={2018}
}
```

## Requirements
- Kaldi from https://github.com/kaldi-asr/kaldi
- Python
- NumPy
- TensorFlow

## License
Apache License, Version 2.0 ('LICENSE')

## Contact
- If you have an extension to share, please create a pull request.
- For feedback and suggestions, please create a GitHub 'Issue' in the project.
- If you want to communicate with me, you can use my email: hsn(dot)zeinali(at)gmail(dot)com
