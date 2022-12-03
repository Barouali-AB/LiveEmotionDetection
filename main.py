import time
import cv2
from tensorflow.keras.models import model_from_json
import numpy as np




# load json and create model
json_file = open('model2_cnn.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model2_cnn.h5")


# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# To capture video from webcam.
cap = cv2.VideoCapture(0)
# To use a video file as input
# cap = cv2.VideoCapture('filename.mp4')

emotions = ["Angry", "Afraid", "Happy", "Sad", "Surprised", "Neutral"]
emotion_weights = [1.67, 25, 0.42, 2.78, 33.34, 0.56]
captured_emotions = []
score = 0


def change_image_format(image):
    new_image = []
    for i in range(len(image)):
        l = []
        for j in range(len(image[0])):
            l.append([image[i][j]])
        new_image.append(l)
    new_image = np.array(new_image)

    return new_image

i = 0

while True:

    # Read the frame
    start_time = time.time()
    boolean, img = cap.read()

    if boolean == True:

      # Convert to grayscale
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      # Detect the faces
      faces = face_cascade.detectMultiScale(gray, 1.1, 4)
      # Draw the rectangle around each face
      for (x, y, w, h) in faces:
          extracted = img[y:y + h + 20 , x:x + w + 20]
          cv2.rectangle(img, (x, y), (x+w+20, y+h+20), (255, 0, 0), 2)
          cv2.rectangle(img , (x, y + h + 20), (x+w+20, y + h + 50), (255, 0, 0), cv2.FILLED)
          font = cv2.FONT_HERSHEY_SIMPLEX
          extracted_gray = cv2.cvtColor(extracted, cv2.COLOR_BGR2GRAY)
          resized = cv2.resize(extracted_gray, (48, 48))
          print(resized.shape)
          #image = color.rgb2gray(io.imread('test11.jpg'))
          image = resized
          image = image / 255
          image = np.array([image])
          #image = change_image_format(image)
          pred = loaded_model.predict(image)
          emotion = pred[0].argmax()
          cv2.putText(img, str(emotion), (x + 96, y + h + 42), font, 1.0, (255, 255, 255), 1) # replace text by emotions[emotion]

          # Save the emotions each 5 seconds
          if ( i%5 == 0 ):
              captured_emotions.append(emotion)
              score += emotion_weights[emotion]
              if ( score >= 100 ) :
                  print("Recommendation")
                  # Recommend(captured_emotions, emotion_weights)  to implement
                  score = 0

      # Display
      cv2.imshow('Capture', img)

      # Stop if 'x' is pressed
      if cv2.waitKey(5) == ord('x'):
          break

      #time.sleep(1.0 - time.time() + start_time)
      time.sleep(1.0)

    i+=1


# Release the VideoCapture object
cap.release()












