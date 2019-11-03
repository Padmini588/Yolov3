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


m = 0.9
e = 0.0005
l_r = 0.001
ra = 0.1
anchors = np.array([[10,13],[16,30],[33,23],[30,61],[62,45],[59,119],[116,90],[156,198],[373,326]])
mask = 0,1,2
jitter = .3
ignore_thresh=0.7
truth_thresh = 1


dic={'falcon': 0, 'goose': 1, 'great argus': 2, 'kingfishers': 3, 'parrots': 4, 'penguins': 5, 'rails': 6, 'secretary bird': 7, 'bee-eater': 8, 'bittern': 9, 'bulbuls': 10, 'chachalaca': 11, 'crane': 12, 'cuckoos': 13, 'curassow': 14, 'duck': 15, 'eagle': 16, 'erget': 17, 'francolin': 18, 'guan': 19, 'hawk': 20, 'heron': 21, 'hornbill': 22, 'ibis': 23, 'kite': 24, 'motmot': 25, 'mynas': 26, 'owls': 27, 'partridge': 28, 'peafowl': 29, 'pelicans': 30, 'pheasant': 31, 'pigeons': 32, 'pittas': 33, 'quail': 34, 'screamers': 35, 'shorebirds': 36, 'sparrow': 37, 'spoonbill': 38, 'spurfowl': 39, 'storks': 40, 'swan': 41, 'tody': 42, 'toucans': 43, 'tragons': 44, 'trgopan': 45, 'turacos': 46, 'vulture': 47, 'woodpeckers': 48}

num_classes = len(dic)


def residual_darknet53(Inp,num_Filters):
  x = Conv2D(num_Filters,kernel_size=(1,1),use_bias=False,kernel_regularizer = l2(5e-4))(Inp)
  x = BatchNormalization(axis = -1,momentum = m,epsilon=e)(x)
  x = LeakyReLU(alpha = ra)(x)

  x = Conv2D(num_Filters*2,kernel_size=(3,3),padding='same',use_bias=False,kernel_regularizer = l2(5e-4))(x) 
  x = BatchNormalization(axis = -1,momentum=m,epsilon=e)(x)
  x =  LeakyReLU(alpha = ra)(x)
  return x 

def residual_blk_darknet(Inp,num_Filter):
  x = Inp
  y = Inp
  y = residual_darknet53(y,num_Filter)
  x = add([x,y])
  return x


#Standard convolution layers used for striding
def conv2DDownSample(Inp,num_Filter):
  x=ZeroPadding2D(((1,0),(1,0)))(Inp)
  x=Conv2D(num_Filter,kernel_size=(3,3),strides=(2,2),use_bias=False,kernel_regularizer = l2(5e-4))(x)
  x=BatchNormalization(axis=-1,momentum=m,epsilon=e)(x)
  x=LeakyReLU(alpha=ra)(x)
  return x 

#Standard initial convolution layer without striding
def conv2d_block(Inp,filters,kernel):
  x = Conv2D(filters,kernel_size=(kernel,kernel),padding='same',use_bias=False,kernel_regularizer = l2(5e-4))(Inp)
  x = BatchNormalization(axis = -1,momentum=m,epsilon=e)(x)
  x = LeakyReLU(alpha=ra) (x)
  
  return x


def conv2d_block_bias(Inp,filters,kernel):
  x = Conv2D(filters,kernel_size=(kernel,kernel),padding='same',kernel_regularizer = l2(5e-4))(Inp)  
  return x


#Base model:

def darknet_53(inp):
  #Inp = Input(shape=(img_rows,img_cols,num_channels))
  x = conv2d_block(inp,32,3)
  x = conv2DDownSample(x,64) #downsampling part1
  x = residual_blk_darknet(x,32) #residual block with activation linear
  x = conv2DDownSample(x,128) #convolution downsampling
  for _ in range(2):
    x = residual_blk_darknet(x,64)
  x = conv2DDownSample(x,256) #convolution downsampling
  for _ in range(8):
      x = residual_blk_darknet(x,128)
  route_1 = Model(inp,x)
  x = conv2DDownSample(x,512) #convolution downsampling
  for _ in range(8):
      x = residual_blk_darknet(x,256)
  route_2 = Model(inp,x)
  x = conv2DDownSample(x,1024) #convolution downsampling    
  for _ in range(4):
      x = residual_blk_darknet(x,512)
  model = Model(inp,x)
  return route_1,route_2,model

#The pyramidal feature mapping of the yolo. The following layers are added at different layers of the base model The pyramid network extracts the features at 3 different scales. At the output pf the base model we get the data of grid size 13x13
#Since the data has neem down sampled 5 times assuming the data size as 416 it self. 
#The data added at the last layer with some convolution strides. 2 routes are defined for the same: 
#The feature scaling factor is similar to the one used in the standard residual networks. 


def yolo_convolution_block(inputs,filters,out_filters):
  print(out_filters)
  inputs = conv2d_block(inputs,filters,kernel=1)
  inputs = conv2d_block(inputs,filters*2,kernel=3)
  inputs = conv2d_block(inputs,filters,kernel=1)
  inputs = conv2d_block(inputs,filters*2,kernel=3)
  inputs = conv2d_block(inputs,filters,kernel=1)

  route = inputs

  inputs = conv2d_block(inputs,filters*2,kernel=3)
  inputs = conv2d_block_bias(inputs,out_filters,kernel=1)
  
  return route,inputs

#The data obtained from the layers above is obtained as darknet 53 route1 route2  the below function forms the yolo body. The first layer gives the feature maps at layer 13x13 second at 26x26 and 3rd at 52x52 respectively. 
def yolo_body(inputs,num_anchors,num_classes):
  route1,route2,darknet53 = darknet_53(inputs)
  
  #the second last layer of darknet 53 contains 512 filters
  route,last1 = yolo_convolution_block(darknet53.output,512,num_anchors*(num_classes+5))
  #upsampling the 2nd last layer
  route = conv2d_block(route,256,1)
  route = UpSampling2D((2,2))(route)
  
  route = Concatenate()([route,route2.output])
  #repeating the same process for the 2nd last layer
  #256 as the second to the second last layer makes use of 256 filters
  route,last2 = yolo_convolution_block(route,256,num_anchors*(num_classes+5))
  
  #downsampling it further
  route = conv2d_block(route,128,1)
  route = UpSampling2D((2,2))(route)
  route = Concatenate()([route,route1.output])
  
  #repeating the process for the final time
  route,last3 = yolo_convolution_block(route,128,num_anchors*(num_classes+5))
  return Model(inputs,[last1,last2,last3])
  #Adding the upsampled feature with the output layer-1

#Based on the outputs obtained from the last layer we can further generate to predict box center height width and center points. 


def yolo_head(feats,anchors,num_classes,input_shape,calc_loss=False):
  """
  Converting the final layer obtained to bounding box parameters

  Parameters are defined as follows:
  ----------------------------------
  feats: tensor
    Final convolution layer features
    anchors : array-like Anchor-box width and height
    num_classes:int
    Number of target classes

  Returns: 
  box_xy:tensor x,y prediction adjusted by spatial location in convlayer (each of the feature map level)
  box_wh: tensor: w,h predictions adjusted by anchors and conv spatial resolution (each of the feature map level)
  box_conf:tensor: Probability estimate to indicate whether the box contains an object or not (confidence score )
  box_class_pred: tensor: Probability destribution of the box for each of the labels  (class prediction)
  """
  #Defining the number of anchors
  num_anchors = len(anchors)
  #Reshaping it to batch,height,width,num_anchors,box_parameters [1,1,1,9,2 ] 9 anchors and 2 coordinates width and height
  anchors_tensor = K.reshape(K.constant(anchors),[1,1,1,num_anchors,2])
  
  #This assumes the 1st and 3rd index position belongs to the width and height feats is a 5 vector consisting of 0: confidence score 1: height 2: width 3: x_point 4: y_point coordinate 5: onwards class prediction
  grid_shape = K.shape(feats)[1:3] #height and width
  #creating a cell mesh
  #Dymamic implementation of conv dims for fully convolutional mode
  #The first value of the grid shape indicates the height. This will convert the height and width in tha array grid of the mentioned size. Each element is repeated per row value
  grid_y = K.tile(K.reshape(K.arange(0,stop=grid_shape[0]),[-1,1,1,1]),[1,grid_shape[1],1,1])
  #Second value of the grid shape indicates the width. This indicates the 
  grid_x = K.tile(K.reshape(K.arange(0,stop=grid_shape[1]),[1,-1,1,1]),[grid_shape[0],1,1,1])
  """
  e.g if grid[0],grid[1]=3
  then values will be:
  grid_y  grid_x
  0 0 0   0 1 2
  1 1 1   0 1 2  
  2 2 2   0 1 2
  
  Thus forming a grid
  """
  grid= K.concatenate([grid_x,grid_y])
  """
  The above step will do the below
  (0,0) (0,1) (0,2)
  (1,0) (1,1) (1,2)
  (2,0) (2,1) (2,2)
  """
  #Converting the grid obtained data type of the last convolution layer. The grid points are basically at 13x13 26x26 and 52x52

  grid = K.cast(grid,K.dtype(feats))
  #Reshaping the tensor obtained out after the resizing operation
  #Values are reshaped as  -1 : batch size height width number of anchors number of classes
  """
  The value obtained from the last layer are seperated at each grid point. 13x13 26x26 and 52x52 respectively. 
  
  Each point is responsible for predicting a value of the box

  """
  feats =K.reshape(feats,[-1,grid_shape[0],grid_shape[1],num_anchors,num_classes+5])
  
  #box_x = K.sigmoid(feats[...,0]+grid_x)/K.cast(grid_shape[1],K.dtype(feats))
  #box_y = K.sigmoid(feats[...,1]+grid_y)/K.cast(grid_shape[0],K.dtype(feats))
  #Center point creation normalized to height and width coordinates
  
  #K.sigmoid(feats[...,:2]) converts the center coordinates to values between 0 and 1
  #grid adds the height and width values to it
  #This value is furtherd divided with the height and widht to normalization
  """
  x and y values are predicted as: 
  bx = sigmoid(pred_x)+cx/width
  by = sigmoid(pred_y)+cy/height
  """

  box_xy = (K.sigmoid(feats[...,:2])+grid)/K.cast(grid_shape[::-1],K.dtype(feats))
  #Width and height extraction normalized to height and width coordinates
  #Same process follow here apart from the fact that the expected height and width are multiplied with the anchor tensores. 
  """
  Width and height are calculated as below: 
  bw = pw*e^(tw)
  bh = ph*e^(th)

  pw and ph are the values of the anchor tensors 
  The values of bw and bh are further normalized with respect to height and width values
  """

  box_wh = K.exp(feats[...,2:4])*anchors_tensor/K.cast(input_shape[::-1],K.dtype(feats))
  #Prediction of the confidence of the box
  """
  Box confidence is calculated and normalized between 0 to 1
  """
  box_confidence = K.sigmoid(feats[...,4:5])
  #Preiction of each of the class probabilites 
  """
  Class probabilites of different classes are calculated and normalized between 0 and 1
  """
  
  box_class_probs = K.sigmoid(feats[..., 5:])
  #Incase of testing the not returning confidence
  if calc_loss == True:
    """
    While training of the yolo grid size (feature map size) the last layer resized and further center coordinates x,y and w and h are calculated
    """
    return grid,feats,box_xy,box_wh
  #Returning confidence as a part of the yolo heading  
  return box_xy,box_wh, box_confidence,box_class_probs

"""Defining the yolo loss functions"""

"""
Defining the box IOU (intersection over union). Given 2 boxes this gives the maximum area covered by the 2 boxes. 
IOU being defined as: 

IOU = (Intersection-Area)/(Union Area: box1-Area + box2-Area -  Intersection area)
"""

def box_iou(b1,b2):
  """
  Returning the iou tensor
  Parameters: 
  ----------
  b1: tensor of shape (i1,...,iN,4),xywh
  b2: tensor, shape = (j,4), xywh

  Returns
  ------------------------
  iou tensor,shape = (i1,...,iN,j)
  """
  #Expand dim to apply broadcasting:
  """
  Seperares the box height and width with the center coordinates. 
  """
  b1 = K.expand_dims(b1,-2)
  #First 2 coordinates as heigy and width
  b1_xy = b1[...,:2]
  b1_wh = b1[...,2:4]
  b1_wh_half = b1_wh/2.
  b1_mins = b1_xy-b1_wh_half
  b1_maxes = b1_xy + b1_wh_half

  #Expanding 2nd box
  b2 = K.expand_dims(b2,0)
  b2_xy = b2[...,:2]
  b2_wh = b2[...,2:4]
  b2_wh_half = b2_wh/2.
  b2_mins = b2_xy-b2_wh_half
  b2_maxes =b2_xy + b2_wh_half
  intersect_mins = K.maximum(b1_mins,b2_mins)
  intersect_maxes = K.minimum(b1_maxes,b2_maxes)
  intersect_wh = K.maximum(intersect_maxes-intersect_mins,0)
  intersect_area = intersect_wh[...,0]* intersect_wh[...,1]
  b1_area = b1_wh[...,0]*b1_wh[...,1]
  b2_area = b2_wh[...,0]*b2_wh[...,1]
  iou = intersect_area/(b1_area + b2_area - intersect_area)

  return iou

"""
  defining the yolo_loss tensor
  The input loss function is defined as 
  args: This contains 2 arguments: 
  1. true boxes
  2. predicted boxes
  
  Parameters:
  ----------------------
  args:  y_predicted : x_center, y_center,width, height, class
         y_true: actual values it onsidts of following blocks: Number of batches of data, grid shape on the prediction, anchors and 4 + 1 + number pf classes
  
  yolo_outputs: tensor representing a single loss   mean localiztion loss across minibatch
  
  Final convolution layer features
  true_boxes:tensor:
  --Ground truth boxes, tensor with shape[batch,num_true_boxes,5]
  --containing box x_center,y_center, width, height and class 
  detector_mask: array
  
  anchors: tensor: 
  Anchor boxes for model 
  num_classes:int Number of object classes

  Parameters here are as follows:
  1. yolo_outputs: list of tensor representing the output of yolo_body
  y_true: list of array, the output of preprocess_true_boxes
  anchors:array, shape=(N,2) wh
  num_classes:integer
  ignore_thres: float, the iou threshold whether to ignore the object confidence loss

  Returns
  -----------
  loss: tensor,shape=(1,)
  """
"""
  (yolo_output,true_boxes,detector_mask,matching_true_boxes) = args

  """
"""
The yolo loss function defined as below: 

The YOLO loss is defined as 
1/N(samples) (xy_loss+wh_loss+confidence_loss+class_loss) : total loss

"""
"""
  Anchor mask usage logic: 
  [yolo]
  mask = 6,7,8
  [yolo]
  mask = 3,4,5
  [yolo]
  mask = 0,1,2
  #Darknet
  if(mask) l.mask = mask
  else{
  l.mask = calloc(n,sizeof(int));
  for(i=0;i<n;i++)
  {
    l.mask[i] = i;

  }

  }

  Every layer has to know about all of the anchors but it is only predicting a subset of them. 
  The first yolo predicts 6,7,8 because it is the largest ones and its on the coarse scale
  The second layer predicts more smaller one's etc

  If the layer assumens of it isn't passed a mask it is reponsible for all bounding boxes a

  The [yolo] layers simply apply logistic activation to some of the neurons, mainly the ones predicting (x,y) offset, objectivenss and class probabilities. 

  E.g. 

  If there is a dog in a picture 

  Anchors are initial sizes some of which will be resized to the object size some using the outputs from the neural network

  mask = 6,7,8 just replace the 3 maximum values and the 

  There are 4 important variables: 
  anchors: predetermined set of boxes with particular height and width ratios
  mask: list of ids of the bounding boxes that the layer is responsible for predicting
  num: total number of anchors
  filter = (num_classes+5)*k where k is the number of mask in one yolo layer

  YOLOv3 predicts offsets from a predetermined set of boxes with particular height-width ratios. Anchors are initialized (width,height) sizes,
  some of which will be resized to the object size

  Default configuration of YOLOv3: 
  [yolo]
  anchors = 10,13 16,30 33,23 30,61 62,45 59,119 116,90 156,198 373,326
        
  mask = 0,1,2

  [[6,7,8],[3,4,5],[0,1,2]]
  mask 0,1,2 
  the last layer of the yolo config file is reponsible for predicting the bounding boxes related to
  0 (10,13) 3 (30,61)  6 (116,90)
  1 (16,30) 4 (62,45)  7 (156,198)
  2 (33,23) 5 (59,119) 8 (373,326) 

  
  Thus the first yolo layer is reponsible for predicting the 6,7,8 coarse one
  second 3,4,5 (fine tuned using up sampling)
  thrid 0,1,2  (more fine tuned)
 
  """
"""
  The first and second output of the yolo layers are scaled to 32 as the original image is down sampled by that amount by the 
  5 down sampling layers
"""

"""
  For each of the layers: get the resulting yolo outputs: 
  1. yolo_outputs[1]
  grid_shape  : Getting the feature maps of the outputs : 13x13 26x26 52x52
"""

"""
  The first value of the yolo_output returns the batch size 
"""

"""
    The final layers obtained are passed to the yolo head. This function returns the grid used at each feature map. The value of box xy width and height
    and output tensor in the required format. 
    """
"""
    The width and center points are concatenated with each other this is used for calculation of loss function
"""

"""
    Extracting the true values: 

    the raw xy is obtained scaled to the grid size and then substracted by the grid off set. This is to have a value between 0 and 1 that can be used for the prediction of the center coordinates
        raw wh is obtained and is scaled to be compared with predicted values of width and height as : 
        tw = log (bx*scale)/achor_width
        ty = log (by*scale)/anchor_height
        In case the object score is 0 then the value is 0 instead of -inf. No loss is calculated when there are no object present
        Since the box is rescaled y_true[l][...,2:3] and y_true[l][...,3:4] box scale loss is calculated. 2 - width* height
        ideally the max area covered by the box is 1 since the values are normalized. This can further be used as a penalizing factor
    """   
"""
    ignore mask is generated according to the IoU theshold and the prediction. and the value of trye box are calculated as IoU. 
    The value of the anchor box whose IoU is smaller than the maximum threshold is suppressed
    The shape of the Ignore mask is of ?,?,?,3,1 0th is batch size 1 and 2 feature map 3 anchors and 1 prediction
"""

def yolo_loss(args,anchors,num_classes,ignore_thres=.5,print_loss=True):
  #The number of feature scaling. 
  num_layers = len(anchors)//3 #default setting This defined the number of anchors used per scale. More than one scale values used will cause the number of layers to be increased. accordingly. 
  #Prediction made at each of the feature level
  yolo_outputs = args[:num_layers]
  #yolo_outputs = K.print_tensor(yolo_outputs,message="yolo_layer_output:")
  #Predicted output of the mentioned inputs
  y_true = args[num_layers:]
  #y_true = K.print_tensor(y_true,message="y_true:")
  #The anchor mask outputs
  
  anchor_mask = [[6,7,8],[3,4,5],[0,1,2]] if num_layers==3 else [[3,4,5],[1,2,3]]
  
  input_shape = K.cast(K.shape(yolo_outputs[0])[1:3]*32,K.dtype(y_true[0]))
  #Grid shape based on the yolo outputs
  
  grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3],K.dtype(y_true[0])) for l in range(num_layers)]
  """
  loss value is initialized to 0
  """
  
  loss = 0
  #Batch size 
 

  m = K.shape(yolo_outputs[0])[0] #batch size, tensor
  #Converting batch size to float
  mf = K.cast(m,K.dtype(yolo_outputs[0]))
  #mf = K.print_tensor(mf,message="mf:")
  for l in range(num_layers):
    #Getting the object mask layer this gives the confidence layer. Objectiveness of each of the true values
    object_mask = y_true[l][...,4:5]
    #object_mask = K.print_tensor(object_mask,message="object_mask:")
    #Getting the true class probabilities. Each of the confidence score for the class values
    true_class_probs = y_true[l] [...,5:]
    #true_class_probs = K.print_tensor(true_class_probs,message="true_class_probs:")
    
    grid,raw_pred,pred_xy,pred_wh = yolo_head(yolo_outputs[l],anchors[anchor_mask[l]],num_classes,input_shape,calc_loss=True)
    #grid = K.print_tensor(grid,message="grid:")
    #raw_pred = K.print_tensor(raw_pred,message="raw_pred:")
    #pred_xy = K.print_tensor(pred_xy,message="pred_xy:")
    #Concatenating the the center point and width and height coordinates
    
    pred_box = K.concatenate([pred_xy,pred_wh])
    #pred_box = K.print_tensor(pred_box,message="pred_box:")
    
    #Darknet raw box to calculate loss. 
    #Converting the raw true values in to values yolo predicts
    raw_true_xy = y_true[l][...,:2]*grid_shapes[l][::-1] -grid #grid acts as an offset
    #raw_true_xy = K.print_tensor(raw_true_xy,message="raw_true_xy:")
    raw_true_wh = K.log(y_true[l][...,2:4]/anchors[anchor_mask[l]]*input_shape[::-1])
    #raw_true_wh = K.print_tensor(raw_true_wh,message="raw_true_wh:")
    raw_true_wh = K.switch(object_mask,raw_true_wh,K.zeros_like(raw_true_wh))  #this is to avoid log(0) = -inf
    #raw_true_wh = K.print_tensor(raw_true_wh,message="raw_true_wh:")
    box_loss_scale = 2 - y_true[l][...,2:3] * y_true[l][...,3:4]
    #box_loss_scale = K.print_tensor(box_loss_scale,message="box_loss_scale:")
    
    #Find ignore mask, iterate over each of the batch.....

    
    ignore_mask = tf.TensorArray(K.dtype(y_true[0]),size=1,dynamic_size=True)
    #ignore_mask = K.print_tensor(ignore_mask,message="ignore_mask:")
    """
    For each of the batches this gives the value of the objectiveness score as either true or fakse
    """
    object_mask_bool = K.cast(object_mask,'bool')

    def loop_body(b,ignore_mask):
      """
      This returns a value of true box which is masked by boolean threshold
      If the objectiveness of the mask is 0 not not present then the loss is 
      simply not calculated. 
      """
      #Gives only those boxes which has object in them
      true_box = tf.boolean_mask(y_true[l][b,...,0:4],object_mask_bool[b,...,0])
      """
      At a particular batch the IOU is calculated and returned. The best of the predicting anchor is stored as 
      best_iou: the values of the best IOU remain: batch, index of anchor matching the most

      """
      #IoU
      iou = box_iou(pred_box[b],true_box)
      best_iou = K.max(iou,axis=-1)
      """
      Incase the threshold covered is very low then it is writtened in the ignore_mask tensor. 
      """
      ignore_mask = ignore_mask.write(b,K.cast(best_iou<ignore_thres,K.dtype(true_box)))
      return b+1,ignore_mask
    """
    For all the values the input argument and the batch size the values are calculated. for the mask whose IOU is less then the threshold
    """
    _,ignore_mask=tf.while_loop(lambda b,*args: b<m,loop_body,[0,ignore_mask])
    #K.control_flow_ops.while_loop(lambda b,*args: b<m,loop_body,[0,ignore_mask])
    """
    The mask are further stacked. This is used in the calculation of the confidence loss
    """
    ignore_mask = ignore_mask.stack()
    ignore_mask = K.expand_dims(ignore_mask,-1)
    #ignore_mask = K.print_tensor(ignore_mask,message="ignore_mask:")
    t1 = K.sum(raw_true_wh)
    raw_true_wh = K.switch(tf.math.is_inf(t1),K.zeros_like(raw_true_wh),raw_true_wh)
    #t2 = K.print_tensor(K.sum(raw_pred[...,2:4]),message = 'pred_wh:')
    xy_loss = object_mask*box_loss_scale*K.binary_crossentropy(raw_true_xy,raw_pred[...,0:2], from_logits = True)
    #xy_loss = K.print_tensor(xy_loss,message="xy_loss:")
    
    wh_loss = object_mask *box_loss_scale*0.5*K.square(raw_true_wh - raw_pred[...,2:4])
    #K.print_tensor(K.sum(wh_loss),message="wh_loss:")
    
    confidence_loss = object_mask * K.binary_crossentropy(object_mask,raw_pred[...,4:5],from_logits=True) + (1-object_mask) * K.binary_crossentropy(object_mask,raw_pred[...,4:5],from_logits=True)*ignore_mask 
    #confidence_loss = K.print_tensor(confidence_loss,message="confidence_loss:")
    
    
    class_loss = object_mask *K.binary_crossentropy(true_class_probs,raw_pred[...,5:],from_logits=True)
    #class_loss = K.print_tensor(class_loss,message="class_loss:")
    """
    Loss is calculated: 

    xy_loss: objective mask*box_loss_Scale*binary_cross_entropy(true and predicted values ) 
   
    wh_loss: 1/2(raw_wh-raw_wh)^2*box_loss*objective_mask
    confidence_loss: objective_mask*binary_crossentropy(confidence_true,predicted)  + (1-objective_mask)*binary_crossentropy(object_mask,raw_predictions[...,4:5])
    class_loss: true class v/s binary_cross entropy
    The box loss is further normalized
    width and height are further normalized. 
    confidence_loss is summed and normalized 
    class_loss is normalized per batch 
    loss are summed xy_loss+ confidence_loss and class_loss
    """  
    
    xy_loss = K.sum(xy_loss) /mf

    
    
    wh_loss = K.sum(wh_loss) /mf
    #wh_loss = K.print_tensor(wh_loss,message="\nwh_loss after:")
    confidence_loss = K.sum(confidence_loss)/mf
    #confidence_loss = K.print_tensor(confidence_loss,message="confidence_loss:")
    class_loss = K.sum(class_loss)/mf

    #class_loss = K.print_tensor(class_loss,message="class_loss:")
    
    loss += xy_loss + wh_loss + confidence_loss + class_loss
    
   # loss = K.print_tensor(loss,message="loss")
    
    #tf.print("loss:",loss,[loss,xy_loss,wh_loss,confidence_loss,class_loss,K.sum(ignore_mask)],output_stream=sys.stdout)
  return loss





