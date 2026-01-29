from django.shortcuts import render

import os
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import tkinter
import numpy as np
import cv2
import imutils
import pandas as pd


# Create your views here.
def index(request):
    return render(request, 'AdminApp/index.html')


def login(request):
    return render(request, 'AdminApp/Admin.html')


def LogAction(request):
    username = request.POST.get('username')
    password = request.POST.get('password')
    if username == 'Admin' and password == 'Admin':
        return render(request, 'AdminApp/AdminHome.html')
    else:
        context = {'data': 'Login Failed ....!!'}
        return render(request, 'AdminApp/Admin.html', context)


def home(request):
    return render(request, 'AdminApp/AdminHome.html')


global dataset


def loaddataset(request):
    global dataset
    dataset = "dataset\\MedicinalPlant dataset"
    context = {'data': 'Medicinal Plant Dataset Uploaded Successfully..!!'}
    return render(request, 'AdminApp/AdminHome.html', context)


global training_set, test_set


def ImageGenerate(request):
    global training_set, test_set
    train_datagen = ImageDataGenerator(shear_range=0.1, zoom_range=0.1, horizontal_flip=True)
    test_datagen = ImageDataGenerator()
    training_set = train_datagen.flow_from_directory(dataset,
                                                     target_size=(48, 48),
                                                     batch_size=32,
                                                     class_mode='categorical',
                                                     shuffle=True)
    test_set = test_datagen.flow_from_directory(dataset,
                                                target_size=(48, 48),
                                                batch_size=32,
                                                class_mode='categorical',
                                                shuffle=False)

    context = {'data': "Generated Training And Testing Images successfully"}
    return render(request, 'AdminApp/AdminHome.html', context)


global classifier


def generateCNN(request):
    global classifier
    if os.path.exists("model\\model_weights.h5"):
        classifier = Sequential()
        classifier.add(Convolution2D(32, kernel_size=(3, 3), input_shape=(48, 48, 3), activation='relu'))
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Convolution2D(32, kernel_size=(3, 3), activation='relu'))
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Flatten())
        classifier.add(Dense(activation="relu", units=128))
        classifier.add(Dense(activation="softmax", units=40))
        classifier.load_weights('model/model_weights.h5')
        context = {"data": "CNN Model Loaded Successfully.."}
        return render(request, 'AdminApp/AdminHome.html', context)
    else:
        classifier = Sequential()
        classifier.add(Convolution2D(32, kernel_size=(3, 3), input_shape=(48, 48, 3), activation='relu'))
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Convolution2D(32, kernel_size=(3, 3), activation='relu'))
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Flatten())
        classifier.add(Dense(activation="relu", units=128))
        classifier.add(Dense(activation="softmax", units=40))
        classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model = classifier.fit_generator(training_set,
                                 steps_per_epoch=50,
                                 epochs=100,
                                 validation_data=test_set,
                                 validation_steps=50)
        classifier.save_weights('model/model_weights.h5')
        final_val_accuracy = model.history['accuracy'][-1]
        msg=f'Final Accuracy: {final_val_accuracy:.4f}'
        context = {"data": "CNN Model Generated Successfully..","msg":msg}
        return render(request, 'AdminApp/AdminHome.html', context)



def uploadPlantImage(request):
    return render(request, 'AdminApp/upload.html')


global filename, uploaded_file_url


def fileUpload(request):
    global filename, uploaded_file_url
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        location = myfile.name
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        imagedisplay = cv2.imread(BASE_DIR + "/" + uploaded_file_url)
        cv2.imshow('uploaded Image', imagedisplay)
        cv2.waitKey(0)
    context = {'data': 'Test Image Uploaded Successfully'}
    return render(request, 'AdminApp/upload.html', context)




def IdentifyMedicinalPlant(request):

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    imagetest = image.load_img(BASE_DIR + "/" + uploaded_file_url, target_size=(48, 48))
    imagetest = image.img_to_array(imagetest)
    imagetest = np.expand_dims(imagetest, axis=0)
    loaded_classifier = Sequential()
    loaded_classifier.add(Convolution2D(32, kernel_size=(3, 3), input_shape=(48, 48, 3), activation='relu'))
    loaded_classifier.add(MaxPooling2D(pool_size=(2, 2)))
    loaded_classifier.add(Convolution2D(32, kernel_size=(3, 3), activation='relu'))
    loaded_classifier.add(MaxPooling2D(pool_size=(2, 2)))
    loaded_classifier.add(Flatten())
    loaded_classifier.add(Dense(activation="relu", units=128))
    loaded_classifier.add(Dense(activation="softmax", units=40))
    loaded_classifier.load_weights('model/model_weights.h5')
    pred = loaded_classifier.predict(imagetest)
    print(str(pred) + " " + str(np.argmax(pred)))
    predict = np.argmax(pred)
    print(training_set.class_indices)
    global msg;
    for x in training_set.class_indices.values():
        if predict == x:
            msg = list(training_set.class_indices.keys())[list(training_set.class_indices.values()).index(x)]
    print("predicted number: " + str(predict))
    dataset=pd.read_excel("dataset/usage.xlsx")
    imagedisplay = cv2.imread(BASE_DIR + "/" + uploaded_file_url)
    oring = imagedisplay.copy()
    output = imutils.resize(oring, width=400)
    data = "Plant Identified As: "+msg
    cv2.putText(output, data, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Predicted image result", output)
    cv2.waitKey(0)
    context = {'data': data,'usage':dataset.usage[predict]}
    return render(request,'AdminApp/PlantUsage.html', context)
