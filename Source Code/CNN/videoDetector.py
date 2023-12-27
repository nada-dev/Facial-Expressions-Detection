import os
import cv2
import numpy as np
from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
from keras.preprocessing.image import load_img, img_to_array 
from keras.models import  load_model
import matplotlib.pyplot as plt
import numpy as np

# load model
myModel = load_model("bestModel.h5")
haarCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
vid = cv2.VideoCapture(0)
while True:
    bol, capturedImage = vid.read()  # captures frame and returns boolean value and captured image
    if not bol:
        continue
    RGB = cv2.cvtColor(capturedImage, cv2.COLOR_BGR2RGB)

    faces = haarCascade.detectMultiScale(RGB, 1.32, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(capturedImage, (x, y), (x + w, y + h), (125, 125, 125), thickness=5)
        roi_gray = RGB[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
        roi_gray = cv2.resize(roi_gray, (224, 224))
        pxls = image.img_to_array(roi_gray)
        pxls = np.expand_dims(pxls, axis=0)
        #we scale the image by 255 to be inputted to the model
        pxls /= 255
        #predicting the image using the loaded model
        predictions = myModel.predict(pxls)

        #Find the label of the image by getting the label corresponding to the highest score prediction
        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise')
        predicted_emotion = emotions[max_index]

        cv2.putText(capturedImage, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0, 255, 255), 2)
        # cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 125, 125), 2)

    resized_img = cv2.resize(capturedImage, (1000, 700))
    cv2.imshow('Facial emotion analysis ', resized_img)

    if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
        break

vid.release()
cv2.destroyAllWindows