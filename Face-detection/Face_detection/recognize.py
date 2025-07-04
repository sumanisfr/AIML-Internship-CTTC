import urllib.request
import cv2
import numpy as np
from keras.models import load_model

# Load the Haar Cascade Classifier
classifier = cv2.CascadeClassifier('C:/Users/suman/OneDrive/Documents/Desktop/Face-detection/Face_detection/haarcascade_frontalface_default.xml')

# Load the trained CNN model
model = load_model("C:/Users/suman/OneDrive/Documents/Desktop/Face-detection/Face_detection/final_model.h5")

# IP camera image stream URL
URL = 'http://100.71.166.221:8080/shot.jpg'

# Label mapping
def get_pred_label(pred):
    labels = ["Asish","Suman","Surya2"]
    return labels[pred]

# Preprocess the face image
def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (100, 100))
    img = cv2.equalizeHist(img)
    img = img.reshape(1, 100, 100, 1)
    img = img / 255
    return img

# Main loop to read from the IP camera
while True:
    try:
        # Get the image from IP camera
        img_url = urllib.request.urlopen(URL)
        image = np.array(bytearray(img_url.read()), np.uint8)
        frame = cv2.imdecode(image, -1)

        # Detect faces
        faces = classifier.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)

        # Draw rectangles and predictions
        for x, y, w, h in faces:
            face = frame[y:y+h, x:x+w]
            processed = preprocess(face)
            prediction = model.predict(processed)
            label = get_pred_label(np.argmax(prediction))

            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.9, (0, 255, 0), 2)

        # Show the result
        cv2.imshow("Face Recognition", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print("Error:", e)
        break

cv2.destroyAllWindows()
