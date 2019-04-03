
# coding: utf-8

# # MNIST Image Classification with TensorFlow on Cloud ML Engine
#
# This notebook demonstrates how to implement different image models on MNIST using Estimator.
#
# Note the MODEL_TYPE; change it to try out different models

# In[9]:


import os
PROJECT = 'qwiklabs-gcp-32b5e449498c3d9f' # REPLACE WITH YOUR PROJECT ID
BUCKET = 'qwiklabs-gcp-32b5e449498c3d9f' # REPLACE WITH YOUR BUCKET NAME
REGION = 'us-central1' # REPLACE WITH YOUR BUCKET REGION e.g. us-central1
MODEL_TYPE='dnn_dropout'  # 'linear', 'dnn', 'dnn_dropout', or 'cnn'

# do not change these
os.environ['PROJECT'] = PROJECT
os.environ['BUCKET'] = BUCKET
os.environ['REGION'] = REGION
os.environ['MODEL_TYPE'] = MODEL_TYPE
os.environ['TFVERSION'] = '1.8'  # Tensorflow version


# In[10]:


get_ipython().run_cell_magic('bash', '', 'gcloud config set project $PROJECT\ngcloud config set compute/region $REGION')


# ## Run as a Python module
#
# In the previous notebook (mnist_linear.ipynb) we ran our code directly from the notebook.
#
# Now since we want to run our code on Cloud ML Engine, we've packaged it as a python module.
#
# The `model.py` and `task.py` containing the model code is in <a href="mnistmodel/trainer">mnistmodel/trainer</a>
#
# **Complete the TODOs in `model.py` before proceeding!**
#
# Once you've completed the TODOs, set MODEL_TYPE and run it locally for a few steps to test the code.

# In[11]:


get_ipython().run_cell_magic('bash', '', 'rm -rf mnistmodel.tar.gz mnist_trained\ngcloud ml-engine local train \\\n   --module-name=trainer.task \\\n   --package-path=${PWD}/mnistmodel/trainer \\\n   -- \\\n   --output_dir=${PWD}/mnist_trained \\\n   --train_steps=100 \\\n   --learning_rate=0.01 \\\n   --model=$MODEL_TYPE')


# **Now, let's do it on Cloud ML Engine so we can train on GPU:** `--scale-tier=BASIC_GPU`
#
# Note the GPU speed up depends on the model type. You'll notice the more complex CNN model trains significantly faster on GPU, however the speed up on the simpler models is not as pronounced.

# In[16]:


get_ipython().run_cell_magic('bash', '', 'OUTDIR=gs://${BUCKET}/mnist/trained_${MODEL_TYPE}\nJOBNAME=mnist_${MODEL_TYPE}_$(date -u +%y%m%d_%H%M%S)\necho $OUTDIR $REGION $JOBNAME\ngsutil -m rm -rf $OUTDIR\ngcloud ml-engine jobs submit training $JOBNAME \\\n   --region=$REGION \\\n   --module-name=trainer.task \\\n   --package-path=${PWD}/mnistmodel/trainer \\\n   --job-dir=$OUTDIR \\\n   --staging-bucket=gs://$BUCKET \\\n   --scale-tier=BASIC_GPU \\\n   --runtime-version=$TFVERSION \\\n   -- \\\n   --output_dir=$OUTDIR \\\n   --train_steps=10000 --learning_rate=0.01 --train_batch_size=512 \\\n   --model=$MODEL_TYPE --batch_norm')


# ## Monitoring training with TensorBoard
#
# Use this cell to launch tensorboard

# In[17]:


from google.datalab.ml import TensorBoard
TensorBoard().start('gs://{}/mnist/trained_{}'.format(BUCKET, MODEL_TYPE))


# In[18]:


for pid in TensorBoard.list()['pid']:
  TensorBoard().stop(pid)
  print('Stopped TensorBoard with pid {}'.format(pid))


# Here are my results:
#
# Model | Accuracy | Time taken | Model description | Run time parameters
# --- | :---: | ---
# linear | 91.53 | 3 min | linear | 100 steps, LR=0.01, Batch=512
# linear | 92.73 | 8 min | linear | 1000 steps, LR=0.01, Batch=512
# linear | 92.29 | 18 min | linear | 10000 steps, LR=0.01, Batch=512
# dnn | 98.14 | 15 min | 300-100-30 nodes fully connected | 10000 steps, LR=0.01, Batch=512
# dnn | 97.99 | 48 min | 300-100-30 nodes fully connected | 100000 steps, LR=0.01, Batch=512
# dnn_dropout | 97.84 | 29 min | 300-100-30-DL(0.1)- nodes | 20000 steps, LR=0.01, Batch=512
# cnn | 98.97 | 35 min | maxpool(10 5x5 cnn, 2)-maxpool(20 5x5 cnn, 2)-300-DL(0.25) | 20000 steps, LR=0.01, Batch=512
# cnn | 98.93 | 35 min | maxpool(10 11x11 cnn, 2)-maxpool(20 3x3 cnn, 2)-300-DL(0.25) | 20000 steps, LR=0.01, Batch=512
# cnn | 99.17 | 35 min | maxpool(10 11x11 cnn, 2)-maxpool(20 3x3 cnn, 2)-300-DL(0.25), batch_norm (logits only) | 20000 steps, LR=0.01, Batch=512
# cnn | 99.27 | 35 min | maxpool(10 11x11 cnn, 2)-maxpool(20 3x3 cnn, 2)-300-DL(0.25), batch_norm (logits, deep) | 10000 steps, LR=0.01, Batch=512
# cnn | 99.48 | 12 hr | as-above but nfil1=20, nfil2=27, dprob=0.1, lr=0.001, batchsize=233 | (hyperparameter optimization)
#
# Create a table to keep track of your own results as you experiment with model type and hyperparameters!

# ## Deploying and predicting with model
#
# Deploy the model:

# In[19]:


get_ipython().run_cell_magic('bash', '', 'MODEL_NAME="mnist2"\nMODEL_VERSION=${MODEL_TYPE}\nMODEL_LOCATION=$(gsutil ls gs://${BUCKET}/mnist/trained_${MODEL_TYPE}/export/exporter | tail -1)\necho $MODEL_LOCATION\necho "Deleting and deploying $MODEL_NAME $MODEL_VERSION from $MODEL_LOCATION ... this will take a few minutes"\n#gcloud ml-engine versions delete ${MODEL_VERSION} --model ${MODEL_NAME}\n#gcloud ml-engine models delete ${MODEL_NAME}\ngcloud ml-engine models create ${MODEL_NAME} --regions $REGION\ngcloud ml-engine versions create ${MODEL_VERSION} --model ${MODEL_NAME} --origin ${MODEL_LOCATION} --runtime-version=$TFVERSION')


# To predict with the model, let's take one of the example images.

# In[23]:


import json, codecs
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

HEIGHT=28
WIDTH=28

mnist = input_data.read_data_sets('mnist/data', one_hot=True, reshape=False)
IMGNO=45 #CHANGE THIS to get different images
jsondata = {'image': mnist.test.images[IMGNO].reshape(HEIGHT, WIDTH).tolist()}
json.dump(jsondata, codecs.open('test.json', 'w', encoding='utf-8'))
plt.imshow(mnist.test.images[IMGNO].reshape(HEIGHT, WIDTH));


# Send it to the prediction service

# In[24]:


get_ipython().run_cell_magic('bash', '', 'gcloud ml-engine predict \\\n   --model=mnist2 \\\n   --version=${MODEL_TYPE} \\\n   --json-instances=./test.json')


# <pre>
# # Copyright 2017 Google Inc. All Rights Reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #      http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# </pre>
