#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import cv2
import numpy as np
import matplotlib.pyplot as plt

dic={'falcon': 0, 'goose': 1, 'great argus': 2, 'kingfishers': 3, 'parrots': 4, 'penguins': 5, 'rails': 6, 'secretary bird': 7, 'bee-eater': 8, 'bittern': 9, 'bulbuls': 10, 'chachalaca': 11, 'crane': 12, 'cuckoos': 13, 'curassow': 14, 'duck': 15, 'eagle': 16, 'erget': 17, 'francolin': 18, 'guan': 19, 'hawk': 20, 'heron': 21, 'hornbill': 22, 'ibis': 23, 'kite': 24, 'motmot': 25, 'mynas': 26, 'owls': 27, 'partridge': 28, 'peafowl': 29, 'pelicans': 30, 'pheasant': 31, 'pigeons': 32, 'pittas': 33, 'quail': 34, 'screamers': 35, 'shorebirds': 36, 'sparrow': 37, 'spoonbill': 38, 'spurfowl': 39, 'storks': 40, 'swan': 41, 'tody': 42, 'toucans': 43, 'tragons': 44, 'trgopan': 45, 'turacos': 46, 'vulture': 47, 'woodpeckers': 48}
yoloconfig='yolo.pbtxt'
yoloweights='yolov3.pb'
anchors = np.array([[10,13],[16,30],[33,23],[30,61],[62,45],[59,119],[116,90],[156,198],[373,326]])

net=cv2.dnn.readNet(yoloweights,yoloconfig)

def cv2plt(img):
    plt.axis('off')
    if np.size(img.shape) == 3:
        plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img,cmap='gray',vmin=0,vmax=255)
    plt.show()


def getOutputLayers(net):
    layers=net.getLayerNames()
    outLayers=[layers[i[0]-1] for i in net.getUnconnectedOutLayers()]
    return outLayers


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

confidences=[]

    
  
  
    
    
    
    classes = np.ones_like(c1)*c
    boxes_.append(class_boxes)
    scores_.append(c1)
    classes_.append(classes)
boxes_ = np.concatenate(boxes_,axis=0)
scores_=np.concatenate(scores_,axis=0)
classes_=np.concatenate(classes_,axis=0)
return boxes_,scores_,classes_



def yoloV3Detect(img,scFactor=1/255,nrMean=(0,0,0),RBSwap=True,scoreThres=0.5,nmsThres=0.4):
    confidences=[]
    blob=cv2.dnn.blobFromImage(image=img,scalefactor=scFactor,size=(416,416),mean=(0,0,0),swapRB=True,crop=False)
    imgHeight=img.shape[0]
    imgWidth=img.shape[1]
    net.setInput(blob)
    outLyrs=getOutputLayers(net)
    preds = net.forward(outLyrs)
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
        class_box_scores = box_scores[mask[:,c]]
        selected = cv2.dnn.NMSBoxes(bboxes=class_boxes,scores=class_box_scores,score_threshold=scoreThres,nms_threshold=0.5)
        class_boxes = class_boxes[selected]
        class_box_scores = class_box_scores[selected]
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

##########


# b. Create a function that receives an image, fboxes, fclasses and classes
#       and produce the output that looks like 'wks3_3_b.jpg'
#
#   The name of the function should be 'pltDetect'


def draw_bounding(img_file,anchors,score_threshold=0.2):
    img = cv2.imread(img_file)
    img1 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img1 = cv2.resize(img1,(416,416))
    img1 = np.array(img1)/255.
    anchor,score,classes = yoloV3Detect(img,scFactor=1/255,nrMean=(0,0,0),RBSwap=True,scoreThres=0.5,nmsThres=0.4)
    #print(anchor[0][1],score[0],classes)
    for i in range(len(anchor)):
        color = np.random.randint(0,255)
    cv2.rectangle(img,(int(anchor[i][1]+0.5)-70,int(anchor[i][0]+0.5)-70),(int(anchor[i][3]+0.5)+70,int(anchor[i][2]+0.5)+70),(color,color,color),2)
    cv2.putText(img,dic[int(classes[i])],(int(anchor[i][1]+0.5+50)-1,int(anchor[i][0]+0.5+50)+1),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    plt.close()
