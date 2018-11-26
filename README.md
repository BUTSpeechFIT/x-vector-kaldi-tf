# x-vector-kaldi-tf
Tensorflow implementation of x-vector topology on top of Kaldi recipe. It uses Kaldi for data processing, feature extraction, data augmentation and VAD. Also, the back-end is Kaldi PLDA. Here we just train model using Tensorflow and also extract speaker embeddings (x-vectors) using it. So, the overall x-vector pipline is same as original Kaldi except training and x-vector extraction.

In this version, we create a different format training archives for Tensorflow. In future we will add ability to use Kaldi training archives.

Some part of this codes actually is modified version of scripts from Kaldi. Also, we have used the following repozitory as initial tensorflow implementation.
https://github.com/qqueing/SR_with_kaldi

Also, for reading and writing data in Kaldi format following kaldi-io repository is used.
https://github.com/vesis84/kaldi-io-for-python

## Requirements
- Python (2.7)
- NumPy
- TensorFlow

##Contact
- If you have an extension to share, please create a pull request.
- For feedback and suggestions, please create a GitHub 'Issue' in the project.
- For the positive reactions =) I am also reachable by email: hsn(dot)zeinali<at>gmail(dot)com

