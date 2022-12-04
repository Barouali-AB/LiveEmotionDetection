from recomm import *
from time import sleep
import cv2
from tensorflow.keras.models import model_from_json
import numpy as np

###########################
#     Reading the model   #
###########################

# load json and create model
json_file = open('model_final_6.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model_final_6.h5")
loaded_model.verbose= False


###########################
#     Initializations     #
###########################

# Load the cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# To capture video from webcam.
cap = cv2.VideoCapture(0)
emotions = ["Angry", "Afraid", "Happy", "Sad", "Surprised", "Neutral"]
emotion_weights = [1.67, 25, 0.42, 2.78, 33.34, 0.56]
emotion_levels = [0, 0, 2, 0, 0, 1]

captured_emotions = []
score = 0

i = 0 
# " i " is used to know if we reached the 5s limit to add an emotion detected to the list
# only needed for the demo version of the code 
# it will be deleted for the official code, that only takes pictures every 5s by default


###########################
#       Main process      #
###########################

while True:

    # Read the frame (take a picture)
    boolean, img = cap.read()

    if boolean: #if a picture was captured
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect the faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Draw the rectangles and extract emotions from each face
        for (x, y, w, h) in faces:
            extracted = img[y:y + h + 20 , x:x + w + 20] # Find the face
            # Preprocess the image
            extracted_gray = cv2.cvtColor(extracted, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(extracted_gray, (48, 48)).reshape(-1,48,48,1)
            image = resized
            image = image / 255
            # Predict emotion
            pred = loaded_model.predict(image)
            emotion = pred[0].argmax()

            ###################### SAVE EMOTION : ##########################################
            # Save the emotions each 5 seconds
            captured_emotions.append(emotion)
            score += emotion_weights[emotion]

            ##################################### RECOMMEND ##########################################
            if ( score >= 100 ) : # If we reached Score threshold to give recommendation 100
                unique, counts = np.unique(np.array(captured_emotions), return_counts=True)
                for i in range(len(counts)):
                    counts[i] *= emotion_weights[unique[i]]
                before = unique[np.argmax(counts)]
                q=recommendation(before) #parameter : the emotion that has biggest (weight*occurences)
                # q is the index of the recommended quote
                print(df_quotes.loc[q].QUOTE)

                ################################# REACTION : ##################################################
                # Record reaction after 5 seconds from receiving the quote to give it a rating
                sleep(5)
                while(True): #keep trying to take a picture as soon as the 5s are done
                    boolean, img = cap.read() #take pic of reaction
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
                        ################################# UPDATE USER INFOS #############################################
                        rating = emotion_levels[emotion] - emotion_levels[before]
                        df_user=df_user.append({"quote_id":q,"rating":rating},ignore_index=True)
                        update_weights()
                        print("before :",emotions[before])
                        print("after :",emotions[emotion])
                        break
                score = 0
                captured_emotions = []
        sleep(5)