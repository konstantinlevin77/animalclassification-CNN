# %% [code]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import PIL
import cv2
from tensorflow import keras
from tensorflow.keras import layers
from glob import glob
import os
import time
from sklearn.model_selection import train_test_split

cat_paths = glob("../input/animal-faces/afhq/train/cat"+"/*") + glob("../input/animal-faces/afhq/val/cat"+"/*")
dog_paths = glob("../input/animal-faces/afhq/train/dog/*") + glob("../input/animal-faces/afhq/val/dog/*")
wild_paths = glob("../input/animal-faces/afhq/train/wild/*") + glob("../input/animal-faces/afhq/val/wild/*")

print(len(cat_paths))
print(len(dog_paths))
print(len(wild_paths))

IMG_SIZE = (256,256)
def readImage(path,label):
    img = np.asarray(PIL.Image.open(path).resize(IMG_SIZE))
    return img,label

img,lbl = readImage(cat_paths[0],0)
plt.imshow(img)
plt.axis("off")
plt.title(lbl)
plt.show()

label_map = {0:"cat",
             1:"dog",
             2:"wild"
            }
import json
json.dump(label_map,open("labelmap.json",mode="w"))

def makeDataset(paths):
    
    start = time.time()
    
    imgs = []
    lbls = []
    count = 0
    for label,clss in enumerate(paths):
        for img_path in clss:
            img,lbl = readImage(img_path,label)
            imgs.append(img)
            lbls.append(lbl)
            count += 1
            
            if count % 500 == 0 and count != 0:
                print("Processing image {}".format(count))
    
    # one hot encoding with pandas :d
    lbls = np.asarray(pd.get_dummies(pd.DataFrame(lbls)))
    imgs = np.asarray(imgs)
    
    full = round(time.time() - start,2)
    print("Dataset has been prepared, process time: {}".format(full))
    
    return imgs,lbls



x,y = makeDataset([cat_paths,dog_paths,wild_paths])

def convPart(filterNum,inputShape=None):
    part = keras.Sequential()
    if inputShape is not None:
        part.add(layers.Conv2D(filterNum,kernel_size=(4,4),strides=1,padding="same",input_shape=inputShape))
    else:
        part.add(layers.Conv2D(filterNum,kernel_size=(4,4),strides=1,padding="same"))
    
    part.add(layers.Conv2D(filterNum,kernel_size=(4,4),strides=2,padding="same"))
    part.add(layers.BatchNormalization())
    part.add(layers.ReLU())
    part.add(layers.MaxPooling2D(pool_size=(2,2)))
    
    return part

testPart = convPart(64,inputShape=(256,256,3))
testPart.summary()

def makeModel(parts):
    model = keras.Sequential()
    for part in parts:
        model.add(part)
    
    model.compile(optimizer="RMSprop",loss="categorical_crossentropy",metrics=["accuracy"])
    return model
    

model = makeModel([convPart(64,inputShape=(256,256,3)),
                   convPart(512),
                   convPart(1024),
                   layers.Flatten(),
                   layers.Dense(3,activation="softmax")
                  ])

model.summary()

y_enc = pd.get_dummies(pd.Series(np.squeeze(y)))
y_enc.head()
results = model.fit(x,np.asarray(y_enc),validation_split=0.25,epochs=10,batch_size=128)
