import os
import scipy
from os import listdir
from PIL import Image as PImage
import matplotlib.pyplot as plt
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import matplotlib.image as img
import PIL as Image
from keras.preprocessing.image import ImageDataGenerator
import sys
from keras.callbacks import TensorBoard

def predict_images(directory,model):
    print("importing images ")

    filesList = os.listdir(directory)
    # print(filesList)
    imagesList = []
    for file in filesList:
        if(file.endswith('.gif')):
            imagesList.append(file)

    image_data = []
    for image in imagesList:
        fname,fext = os.path.splitext(image)
        im = PImage.open(directory + image).convert('RGB')
        width, _ = im.size
        # print(width, _)
        for i, px in enumerate(im.getdata()):
            y = int(i / width)
            x = int(i % width)
            if (px == (0,0,0)):
                im.putpixel((x, y), (255,255,255))
            if ((x<4*400/100)or(_*0.88<y)):
                im.putpixel((x, y), (255,255,255))
        im.save('cleaned_'+fname+'.jpg', format='jpeg')

    filesList = os.listdir('.')
    imagesList = []
    for image in filesList:
        if(image.endswith('.jpg')):
            imagesList.append(image)

    training_data = np.array( [img.imread(image) for image in imagesList])
    training_data = np.divide(training_data,255)
    week_data_train = []

    if(len(imagesList)%7 != 0):
        print('num images not divisible by 7, non-full weeks will be truncated!')

    if(len(training_data)<6):
        print('you need at least 7 images!')
    while(len(training_data)>6):
        week = np.array([training_data[k] for k in range(7)])
        week_data_train.append(week)
        training_data = np.delete(training_data,[i for i in range(7)],axis=0)

    week_data_train = np.array(week_data_train)
    conv_model = keras.models.load_model(model)
    predict_img = conv_model.predict(week_data_train)

    for j in range(len(predict_img)):
        for i in range(7):
            scipy.misc.imsave("predicted_" + imagesList[i], predict_img[0,i])

if __name__ == "__main__":
    directory = sys.argv[1]
    model = sys.argv[2]
    predict_images(directory,model)
