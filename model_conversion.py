from tensorflow.keras.layers import Input,MaxPool2D,Conv2D,Dense,Flatten,BatchNormalization,GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import LeakyReLU,add,Activation
import tensorflow as tf
from tensorflow.keras.layers import Reshape,UpSampling2D,ZeroPadding2D,Concatenate,Lambda
import tensorflow.keras.backend as K
import numpy as np
import os
import cv2
import random
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger,EarlyStopping,ReduceLROnPlateau,TensorBoard
from tensorflow.keras import optimizers
import sys
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from tensorflow.python.framework import graph_io
import yolov3
import yolov3_api


filepath = '/Users/shashanknigam/desktop/CA_OPEN_CV/Weights/yolov3_retrain.h5py'
model = yolov3_api.fit_model_weights(filepath)
model_pred = yolov3_api.create_prediction_model()
K.set_learning_phase(0)
m =  Model(model.layers[0].input,[model.layers[251].output,model.layers[250].output,model.layers[249].output])
sess = K.get_session()
print(m.input,m.output)
frozen = tf.graph_util.convert_variables_to_constants(sess,sess.graph_def,[out.op.name for out in m.outputs])
graph_io.write_graph(frozen,'./','yolo.pb',as_text=False)

#Read the graph
with tf.gfile.GFile('yolo.pb','rb')  as f:
  graph_def = tf.GraphDef()
  graph_def.ParseFromString(f.read())
  sess.graph.as_default()
  g_in = tf.import_graph_def(graph_def)

with tf.Session() as sess:
  sess.graph.as_default()
  tf.import_graph_def(graph_def,name='')

#Strip const nodes
for i in reversed(range(len(graph_def.node))):
  if graph_def.node[i].op =='Const':
    del graph_def.node[i]
  for attr in ['T','data_format','Tshape','N','Tidx','Tdim','use_cudnn_on_gpu','Index','Tperm','is_training','Tpaddings']:
    if attr in graph_def.node[i].attr:
      del graph_def.node[i].attr[attr] 
tf.train.write_graph(graph_def,'./','yolo.pbtxt',as_text = True)       

