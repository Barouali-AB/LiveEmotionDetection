from time import sleep
import cv2
from tensorflow.keras.models import model_from_json
import numpy as np

##########################
# Reading the model
##########################

# load json and create model
json_file = open('model_final_6.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model_final_6.h5")
loaded_model.verbose= False


##########################
# Initializations
##########################
# Load the cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# To capture video from webcam.
cap = cv2.VideoCapture(0)
emotions = ["Angry", "Afraid", "Happy", "Sad", "Surprised", "Neutral"]
emotion_weights = [1.67, 25, 0.42, 2.78, 33.34, 0.56] #values are debatable.

captured_emotions = []
score = 0

i = 0 
# " i " is used to know if we reached the 5s limit to add an emotion detected to the list
# only needed for the demo version of the code 
# it will be deleted for the official code, that only takes pictures every 5s by default


##########################
# Main process
##########################

while True:

    # Read the frame
    boolean, img = cap.read()
    if boolean: #if a picture was captured
      # Convert to grayscale
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      # Detect the faces
      faces = face_cascade.detectMultiScale(gray, 1.1, 4)
      # Draw the rectangles and extract emotions from each face
      for (x, y, w, h) in faces:
          extracted = img[y:y + h + 20 , x:x + w + 20] # Find the face
          ## Draw rectangle around face
          cv2.rectangle(img, (x, y), (x+w+20, y+h+20), (255, 0, 0), 2)
          cv2.rectangle(img , (x, y + h + 20), (x+w+20, y + h + 50), (255, 0, 0), cv2.FILLED)
          font = cv2.FONT_HERSHEY_SIMPLEX
          ## Preprocess the image
          extracted_gray = cv2.cvtColor(extracted, cv2.COLOR_BGR2GRAY)
          resized = cv2.resize(extracted_gray, (48, 48)).reshape(-1,48,48,1)
          image = resized
          image = image / 255
          ## Predict
          pred = loaded_model.predict(image)
          emotion = pred[0].argmax()
          cv2.putText(img, emotions[emotion], (x + 96, y + h + 42), font, 1.0, (255, 255, 255), 1) # Write emotion on picture next to face

          # Save the emotions each 5 seconds
          if ( i%10 == 0 ):
              captured_emotions.append(emotion)
              score += emotion_weights[emotion]
              if ( score >= 2 ) : # If we reached Score threshold to give recommendation 100
                  print("Recommendation") #replace by recomm fct

                  #### to delete
                  print(captured_emotions)
                  unique, counts = np.unique(np.array(captured_emotions), return_counts=True)
                  print(np.asarray((unique, counts)).T)
                  print("Score : ",score)
                  ###
                  
                  # Record reaction after 5 seconds from receiving the quote to give it a rating
                  sleep(5)
                  while(True): #keep trying to take a picture as soon as the 5s are done
                    boolean, img = cap.read()
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                    if boolean and len(faces)!=0:
                        #get emotion
                        (x,y,w,h) = faces[0]
                        extracted = img[y:y + h + 20 , x:x + w + 20]
                        extracted_gray = cv2.cvtColor(extracted, cv2.COLOR_BGR2GRAY)
                        resized = cv2.resize(extracted_gray, (48, 48)).reshape(-1,48,48,1)
                        image = resized
                        image = image/255
                        pred = loaded_model.predict(image)
                        emotion = pred[0].argmax() #Reaction to quote
                        #####
                        # update score of quote.
                        ##################
                        print("after :",emotions[emotion])
                        break
                  exit() # to_delete
                  score = 0
                  captured_emotions = []
          if ( i == 60):
            #### to delete
            print(captured_emotions)
            unique, counts = np.unique(np.array(captured_emotions), return_counts=True)
            print(np.asarray((unique, counts)).T)
            print("Score : ",score)
            exit()

      # Display
      cv2.imshow('Capture', img)

      # Stop if 'x' is pressed
      if cv2.waitKey(5) == ord('x'):
          break

      sleep(0.5)

    i+=1


# Release the VideoCapture object
cap.release()












