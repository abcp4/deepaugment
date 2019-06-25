from deepaugment import DeepAugment

import glob
import cv2

def load(ttype,label,l,cv_img,labels):
  for img in glob.glob('data/'+ttype+'/'+label+'/*.jpg'):
    n= cv2.imread(img)
    width = 64
    height = 64
    dim = (width, height)
    # resize image
    n = cv2.resize(n, dim, interpolation = cv2.INTER_AREA) 
    cv_img.append(n)
    labels.append(l)
  return cv_img,labels
                       

cv_img = []
labels = []
y_test = []
y_train = []
x_train = []
x_test = []

x_train,y_train = load('train','good',0,x_train,y_train)
x_train,y_train = load('train','bad',1,x_train,y_train)
x_train,y_train = load('train','ugly',2,x_train,y_train)
x_train,y_train = load('valid','good',0,x_train,y_train)
x_train,y_train = load('valid','bad',1,x_train,y_train)
x_train,y_train = load('valid','ugly',2,x_train,y_train)
x_test,y_test = load('test','good',0,x_test,y_test)
x_test,y_test = load('test','bad',1,x_test,y_test)
x_test,y_test = load('test','ugly',2,x_test,y_test)
                       
import numpy as np
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

my_config = {
    "model": "basiccnn",
    "method": "bayesian_optimization",
    "train_set_size": 2000,
    "opt_samples": 1,
    "opt_last_n_epochs": 3,
    "opt_initial_points": 1,
    "child_epochs": 1,
    "child_first_train_epochs": 0,
    "child_batch_size": 64,
    "notebook_path": '/content/results' 
    
}

# X_train.shape -> (N, M, M, 3)
# y_train.shape -> (N)
deepaug = DeepAugment(images=x_train, labels=y_train, config=my_config)
