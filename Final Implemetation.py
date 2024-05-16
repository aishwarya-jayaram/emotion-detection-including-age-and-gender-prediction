import cv2
import argparse
import numpy as np
from keras.models import load_model
from tensorflow.keras.utils import img_to_array
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

# Load face detection model
#Initializes the OpenCV Cascade Classifier to detect frontal faces in an image or video frame
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load emotion detection model
emotion_classifier = load_model('emotion_detection_model.h5')
#creates a list of labels for each detected emotion.
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

# Load age and gender detection model binary file which contains the weights for the neural network.
faceProto="opencv_face_detector.pbtxt"
faceModel="opencv_face_detector_uint8.pb"
ageProto="age_deploy.prototxt"
ageModel="age.caffemodel"
genderProto="gender_deploy.prototxt"
genderModel="gender.caffemodel"

#Normalizing the pixel values of the input image
MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)

#Arrays that hold the labels for the age and gender ranges respectively.
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']

#loads the DNN models from the specified files.
faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

# Create a function to open a file dialog for image upload
def open_file_dialog():
    file_path = filedialog.askopenfilename()
    if file_path:
        process_image(file_path)

# Create a function to capture video from camera
def capture_video():
    video = cv2.VideoCapture(0)
    while cv2.waitKey(1) < 0:
        hasFrame, frame = video.read()
        if not hasFrame:
            cv2.waitKey()
            break
        process_frame(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()

# Create a function to process image
def process_image(image_path):
    frame = cv2.imread(image_path)
    process_frame(frame)

# Create a function to process frame
def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        if np.sum([roi_gray]) != 0:
            # Emotion detection
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            emotion_prediction = emotion_classifier.predict(roi)[0]
            emotion_label = emotion_labels[emotion_prediction.argmax()]
            cv2.putText(frame, emotion_label, (x, y+h+37), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            # Age and Gender detection
            blob=cv2.dnn.blobFromImage(frame[y:y+h,x:x+w], 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds=genderNet.forward()
            gender=genderList[genderPreds[0].argmax()]
            ageNet.setInput(blob)
            agePreds=ageNet.forward()
            age=ageList[agePreds[0].argmax()]
            age_gender_label = f'{gender}, {age}'
            cv2.putText(frame, age_gender_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
    return frame


# Create a function for opening file dialog
def open_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        frame = cv2.imread(file_path)
        processed_frame = process_frame(frame)
        cv2.imshow('Detecting Emotion, Age and gender for An Image', processed_frame)


# Create a function for capturing frame from camera
def capture_frame():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        processed_frame = process_frame(frame)
        cv2.imshow('Detection Using Camera', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


# Create GUI window
root = tk.Tk()
root.title('Emotion Detection Including Age and Gender Prediction')

# Create buttons
upload_button = tk.Button(root, text='Upload Image', command=open_file)
upload_button.pack()

camera_button = tk.Button(root, text='Use Camera', command=capture_frame)
camera_button.pack()

# Run GUI loop
root.mainloop()
