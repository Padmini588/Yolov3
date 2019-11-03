
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

"""
Trained model in colab
"""

model=None
m = None
anchors = np.array([[10,13],[16,30],[33,23],[30,61],[62,45],[59,119],[116,90],[156,198],[373,326]])


dic={0:'falcon', 1:'goose', 2:'great argus', 3:'kingfishers', 4:'parrots', 5:'penguins', 6:'rails', 7:'secretary bird', 8:'bee-eater', 9:'bittern', 10:'bulbuls', 11:'chachalaca', 12:'crane', 13:'cuckoos', 14:'curassow', 15:'duck', 16:'eagle',17:'erget', 18:'francolin', 19:'guan', 20:'hawk',21: 'heron', 22:'hornbill',23: 'ibis',24: 'kite',25: 'motmot',26: 'mynas',27: 'owls',28: 'partridge',29: 'peafowl',30: 'pelicans',31: 'pheasant',32: 'pigeons',33: 'pittas',34: 'quail',35: 'screamers',36: 'shorebirds',37: 'sparrow',38: 'spoonbill',39: 'spurfowl',40: 'storks',41: 'swan',42: 'tody',43: 'toucans',44: 'tragons',45: 'trgopan',46: 'turacos',47: 'vulture',48: 'woodpeckers'}
num_classes = len(dic)
img_shape = (416,416)


def read_img(index,shape):
  img = cv2.imread(Img_dir+'/'+image_id[index]) 
  w,h = img.shape[:2]
  if (w,h)<shape:
    img = cv2.resize(img,shape,interpolation=cv2.INTER_CUBIC)
  img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  img = np.array(img)/255.
  return img 

def display_img(index):
  img = cv2.imread(Img_dir+'/'+image_id[index])
  img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  plt.figure()
  plt.imshow(img)
  plt.axis('off')
  plt.show()
  plt.close()

def readLabel(index):
  #print(label_id[index])
  lines=open(Lab_dir+'/'+label_id[index],'r')
  s = lines.readlines()
  l=[]
  for i in s:
    i = i.strip()
    if len(i)>1:
      a = i.strip().split(' ')
      d = []
      k = 0
      for j in a:
        if len(j)!=0:
          if k==0:
            d.append(float(j))
            k = k+1
          else:
             if float(j)>1:
               d.append(float(1)) 
             else: 
               d.append(float(j))  
      a = np.array(d)
      a = np.roll(a,-1)
      l.append(a)
  l = np.array(l)  
  lines.close()
  return l




def preprocess_box_anchors(true_boxes,input_shape,anchors,num_classes):
  """
  true boxes: array of shape (batch_Size,size of labels per image,labels)
  input_shape is the shape of the array multiples of 32
  anchors are the anchors used shape = N,2 w h  format
  num_classes:integer
  Returns
  -------------------------------
  y_true : list of array shaped liked yolo output for loss comparision
  y_true is of the size m:batch_size,h,w :feature map size,objectivenss,bounding_box+objectivness_class prediction
  """
  #print(true_boxes)
  num_layers = len(anchors)//3
  anchor_mask = [[6,7,8],[3,4,5],[0,1,2]] if num_layers == 3 else [[3,4,5],[1,2,3]]
  boxes_xy = true_boxes[...,0:2]*input_shape

  boxes_wh = true_boxes[...,2:4]*input_shape  
  #print(boxes_xy)
  #print(boxes_wh)
  true_boxes =np.array(true_boxes,dtype='float32')
  input_shape = np.array(input_shape,dtype='int32')
  m = true_boxes.shape[0] #batch_size
  grid_shapes = [input_shape//{0:32,1:16,2:8}[l] for l in range(num_layers)]
  # initializing y_true values {Array of predictions across the grid} number of anchors 4 priors 1 confidence and number of clases
  """
  y_true has size b:batches width,height,number of anchor per layer, 4 box + 1 confidence + class
  """
  y_true = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+num_classes),dtype = 'float32') for l in range(num_layers)]
  anchors = np.expand_dims(anchors,0)
  anchor_maxes = anchors/2.
  anchor_mins = -anchor_maxes
  valid_mask = boxes_wh[...,0]>0

  for b in range(m):
    # discard zero rows. 
    # Calculating the width and height of the valid mask. 
    wh = boxes_wh[b,valid_mask[b]]
    if len(wh)!=0:
      wh = np.expand_dims(wh,-2)
      box_maxes = wh/2.
      box_mins = -box_maxes
      intersect_mins = np.maximum(box_mins,anchor_mins) #This gives the coordinates of the minimum value of the anchor box in the specified area
      intersect_maxes = np.minimum(box_maxes,anchor_maxes) #This gives the coordinates of the area lying within the anchor min
      """
    (a)--------------  (anchor_mins)
      |(i)---------(b) |   (negative)  box min intersect mean
      |  |        |    | 
      ---------------- (center point)            
      |  |        |    |   (positive)
      |  ---------(b i)|               box_max intersect max
      --------------(anchor_maxs)  

      """
      intersect_wh = np.maximum(intersect_maxes- intersect_mins,0.) 
      intersect_area = intersect_wh[...,0]*intersect_wh[...,1] #area = width*height
      box_area = wh[...,0] * wh[...,1]
      anchor_area = anchors[...,0]*anchors[...,1]
      iou = intersect_area / (box_area+anchor_area - intersect_area) #This gives intersect over union
      #finding box with heighest iou
      best_anchor = np.argmax(iou,axis=-1)
      for i1,i2 in enumerate(best_anchor):
        for l in range(num_layers):
          #Check if the value matches any anchor mask or not
          if i2 in anchor_mask[l]:
            #Calculating the height and width of the maximum obtained true box
            
            #print(true_boxes)
            i = np.floor(true_boxes[b,i1,0]*grid_shapes[l][1]).astype('int32') #Scaling to original coordinates
            if i == grid_shapes[l][1]:
              i = i-1
            j = np.floor(true_boxes[b,i1,1]*grid_shapes[l][0]).astype('int32') #Scaling to the value
            if j == grid_shapes[l][0]:
              j = j-1
            #print(i,j)
            #For each anchor mask trying to find which of the anchor mask fits the best
            #Value of the i2 if present in the anchor mask
            k = anchor_mask[l].index(i2)
            #Calculating the confidence score of the value predicted
            c = true_boxes[b,i1,4].astype('int32')
            #Assigning the box priors to y_true value
            y_true[l][b,j,i,k,0:4] = true_boxes[b,i1,0:4]
            #confidence
            y_true[l][b,j,i,k,4] = 1
            #class predictions
            y_true[l][b,j,i,k,5+c] = 1
  return y_true

def get_data(index,input_shape,max_boxes = 20):  
    box_data = np.zeros((max_boxes,5))
    img = read_img(index,input_shape)
    labels = readLabel(index)
    if len(labels)>max_boxes:
      labels = labels[:max_boxes]
    box_data[:len(labels)] = labels
    #print(box_data)
    return img,box_data
class data_generator:
  def __init__(self,input_shape,anchors,num_classes,indexes,batch_size=32):
    self.cur_train_index=0
    self.batch_size= batch_size
    self.input_shape = input_shape
    self.anchors = anchors
    self.num_classes = num_classes
    self.indexes = indexes
    #self.num_samples = len(indexes)
  #The function would return the image data and the label value for the same
  #maximum number of boxes per image

def next_train_data(self):
    while 1:
      image_data=[]
      boxes=[]
      for i in range(self.batch_size):
        if self.cur_train_index >= len(self.indexes):
          self.cur_train_index =0
        img,box = get_data(self.indexes[self.cur_train_index],self.input_shape) 
        #print(box)
        self.cur_train_index +=1
        image_data.append(img)
        boxes.append(box)
        #print(boxes)
      image_data = np.array(image_data)
      box_data = np.array(boxes)
      #print(box_data.shape)
      y_true = preprocess_box_anchors(box_data,self.input_shape,self.anchors,self.num_classes) 
      #print([image_data,*y_true],np.zeros(batch_size))  
      yield [image_data,*y_true],np.zeros(batch_size)

"""Creating the model for training images:"""

#Creating training model
def create_model(input_shape,anchors,num_classes):
  #creating of the input space for the model to be trained
  K.clear_session()
  input_image = Input(shape=(None,None,3))
  h,w = input_shape

  num_anchors = len(anchors)
  #default value of the true value value remains a list of 3 elements. 
  y_true = [Input(shape=(h//{0:32,1:16,2:8}[l],w//{0:32,1:16,2:8}[l],num_anchors//3,num_classes+5)) for l in range(3)]
  model_body = yolov3.yolo_body(input_image,num_anchors//3 ,num_classes) #3 anchors per layer
  print(model_body.summary())
  """
  Loading pretrained yolov3 model weights
  """
  #model_body.load_weights(weights_dir+'yolo.h5')
  """
  Freezing the model till the dark net layer
  """
  #num = len(model_body.layers)-3
  #for i in range(num):
  #  model_body.layers[i].trainable = False
  #Creating customized loss layer
  #3 layers predicted tensor + true value of the same
  model_loss =  Lambda(yolov3.yolo_loss, output_shape=(1,), name='yolo_loss',arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thres': 0.5})([*model_body.output, *y_true])
  model= Model([model_body.input,*y_true],model_loss)
  return model


def train_yolo_model(epochs,data_dir,model_name,file_path,log_dir=None,split=0.9,bs=10):
    global model
    l_r = 0.001

    from google.colab import drive
    drive.mount('/content/drive')

    data_dir= data_dir
    Img_dir = data_dir+"/Image"
    Lab_dir = data_dir+"/Label"

    image_data = sorted(os.listdir(Img_dir))
    label_data = sorted(os.listdir(Lab_dir))

    
    print(f"The number of Images in the dataset include {len(image_data)}")
    print(f"The corresponding labels in the dataset include {len(label_data)}")

    #Creating dictionary for object access

    image_id = {i: d for i,d in enumerate(image_data)}
    label_id = {i:d for i,d in enumerate(label_data)}
    #Splitting the indexes into testing and training: 
    #Creation of the index datafile: 
    indexes = list(image_id.keys())
    np.random.seed(30)
    total_data = len(indexes)
    random.shuffle(indexes)
    if split<1:
        splits = split
        train_index = indexes[:int(splits*total_data)]
        test_index = indexes[int(splits*total_data):]
        train_len = len(train_index)
        test_len = len(test_index)
        print(train_len,test_len)
        np.random.seed(30)
        optmz = optimizers.Adam(learning_rate = l_r)
        model_name = model_name
        if log_dir:
            log_dir = '/content/drive/My Drive/Colab Notebooks/'
            logging = TensorBoard(log_dir=log_dir)

        filepath = file_path+model_name+".h5py"
    
    chekpoint= ModelCheckpoint(filepath+"/",monitor='val_loss',save_weights_only=True,save_best_only=True)
    csv_logger = CSVLogger(file_path+"/"+model_name+".csv")
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    #Model creation for training
    model = create_model(input_shape,anchors,num_classes)
    model.compile(optimizer=optmz,loss = {'yolo_loss':lambda y_true,y_pred:y_pred})
    """
    Initializing the batches
    """
    batch_size=bs
    gen_train = data_generator(input_shape,anchors,num_classes,train_index,batch_size)
    gen_val = data_generator(input_shape,anchors,num_classes,test_index,batch_size=bs)

    """
    Fitting the model
    """
    model.fit_generator(gen_train.next_train_data(),steps_per_epoch = max(1,train_len//batch_size),validation_data=gen_val.next_train_data(),validation_steps=max(1,test_len//batch_size),epochs=epochs,callbacks=[chekpoint,csv_logger,reduce_lr,early_stopping,logging])

def fit_model_weights(filepath):
    global model
    model.load_weights(filepath)
    return model

def create_prediction_model():
  return Model(model.layers[0].input,[model.layers[251].output,model.layers[250].output,model.layers[249].output])

def sigmoid(x):
  return 1/(1+np.exp(-x))

def boundingboxes(youtput,anchors,num_classes):
  num_anchors = len(anchors)
  anchors = np.reshape(a,(1,1,1,num_anchors,2))
  grid_shape = np.shape(youtput)[1:3]
  grid_y = np.tile(np.reshape(np.arange(0,grid_shape[0]),[-1,1,1,1]),[1,grid_shape[1],1,1])
  grid_x =  np.tile(np.reshape(np.arange(0,stop=grid_shape[1]),[1,-1,1,1]),[grid_shape[0],1,1,1])
  grid = np.concatenate([grid_x,grid_y],axis=-1)
  youtput = np.reshape(youtput,[-1,grid_shape[0],grid_shape[1],num_anchors,num_classes+5])
  box_xy = (sigmoid(youtput[...,:2])+grid)/grid_shape[::-1]
  box_wh =  np.exp(youtput[...,2:4])*a
  box_confidence = sigmoid(youtput[...,4:5])
  box_class_probs = sigmoid(youtput[...,5:])
  box_yx = box_xy[...,::-1]*416
  box_hw = box_wh[...,::-1]
  box_mins = box_yx - (box_hw/2.)
  box_maxes = box_yx + (box_hw/2.)
  boxes = np.concatenate([box_mins[...,0:1],box_mins[...,1:2],box_maxes[...,0:1],box_maxes[...,1:2]],axis=-1)
  boxes = np.reshape(boxes,[-1,4])
  boxes_scores = box_confidence*box_class_probs
  boxes_scores = np.reshape(boxes_scores,[-1,num_classes])
  return boxes,boxes_scores

def non_max_suppression(boxes,thresh):
  if len(boxes)==0:
    return []

  pick = []
  y1 = boxes[:,0]
  x1 = boxes[:,1]
  x2 = boxes[:,2]
  y2 = boxes[:,3]
  area = (x2-x1+1)*(y2-y1+1)
  idxs = np.argsort(y2)
  while len(idxs)>0:
    last = len(idxs) - 1
    i = idxs[last]
    pick.append(i)
    yy1 = np.maximum(y1[i],y1[idxs[:last]])
    xx1 = np.maximum(x1[i],x1[idxs[:last]])
    xx2 = np.maximum(x2[i],x2[idxs[:last]])
    yy2 = np.maximum(y2[i],y2[idxs[:last]])
    w = np.maximum(0,xx2-xx1+1)
    h = np.maximum(0,yy2-yy1+1)
    overlap = (w*h)/area[idxs[:last]]
    idxs = np.delete(idxs,np.concatenate(([last],np.where(overlap>thresh)[0])))
  return pick  

def eval(img,anchors,num_classes=num_classes,max_boxes=20,score_threshold=0.2,iou_threshold=0.5):
  img = np.expand_dims(img,0)
  prediction = model_f.predict(img)
  num_layers = len(prediction)
  mask = [[6,7,8],[3,4,5],[0,1,2]]
  boxes = []
  box_scores = []
  for l in range(num_layers):
    _boxes,_box_scores = boundingboxes(prediction[l],anchors[mask[l]],num_classes)
    boxes.append(_boxes)
    box_scores.append(_box_scores)
  boxes = np.concatenate(boxes,axis=0)
  #print(boxes.shape)
  box_scores = np.concatenate(box_scores,axis=0)
  mask = box_scores >= score_threshold
  boxes_ = []
  scores_ = []
  classes_ = []
  for c in range(num_classes):
    class_boxes = boxes[mask[:,c]]
    #print(class_boxes)
    
    class_box_scores = box_scores[mask[:,c]]
    if class_boxes.shape[0]!=0:
      nms_index =non_max_suppression(class_boxes,0.5)
      class_boxes = class_boxes[nms_index]
      class_box_scores = class_box_scores[nms_index]
      c1=[]
      for i in class_box_scores:
        temp = (np.max(i))
        c1.append(temp)
      c1=np.array(c1)  
      classes = np.ones_like(c1)*c
      boxes_.append(class_boxes)
      scores_.append(c1)
      classes_.append(classes) 
  boxes_ = np.concatenate(boxes_,axis=0)
  scores_=np.concatenate(scores_,axis=0)
  classes_=np.concatenate(classes_,axis=0)
  return boxes_,scores_,classes_   

def draw_bounding(img_file,anchors,score_threshold=0.2):
  img = cv2.imread(img_file)
  img1 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  img1 = cv2.resize(img1,(416,416))
  img1 = np.array(img1)/255.
  anchor,score,classes = eval(img1,anchors,score_threshold=score_threshold)
  #print(anchor[0][1],score[0],classes)
  for i in range(len(anchor)):
    #print(i)
    color = np.random.randint(0,255)
    #if score[i]>score_threshold:
    cv2.rectangle(img,(int(anchor[i][1]+0.5)-70,int(anchor[i][0]+0.5)-70),(int(anchor[i][3]+0.5)+70,int(anchor[i][2]+0.5)+70),(color,color,color),2)
    cv2.putText(img,dic[int(classes[i])],(int(anchor[i][1]+0.5+50)-1,int(anchor[i][0]+0.5+50)+1),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
  img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  plt.figure()
  plt.imshow(img)
  plt.axis('off')
  plt.show()
  plt.close()    



model = create_model((416,416),anchors,num_classes)