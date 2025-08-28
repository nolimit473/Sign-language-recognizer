import streamlit as st
from streamlit_option_menu import option_menu
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque
import time

        
def preprocess_features(landmarks):
                            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
 

                            wrist = landmarks[0]
                            landmarks = landmarks - wrist
                            scale = np.linalg.norm(landmarks[9] - landmarks[0])
                            if scale > 0:
                                landmarks = landmarks / scale
                            return landmarks.flatten()





st.title("Sign Language Gesture Recognition")
st.write("Welcome!")
    
model = tf.keras.models.load_model(r"landmarks.csv") 

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

landmark_buffer = deque(maxlen=5)

# Buffer for stabilizing gesture recognition
gesture_buffer = deque(maxlen=5) 
time_threshold=1.5


recognized_text = ""

last_letter = None  # Store the last recognized letter
last_detection_time = time.time()
text_display = st.sidebar.empty() 


camera = cv2.VideoCapture(0)

if st.button("Start Recognition",key="key0"):
   


    stframe = st.empty()  
    class_names = ["A", "B","C","D","E","F","G","H","I","J","K","L","M","N","O","P",
                "Q","R","S","T","U","V","W","X","Y","Z","Thankyou","Yes"
                ]

    while True:

        ret, frame = camera.read()
        if not ret:
            st.error("Failed to access the camera!")
            break

        # Flip the frame for a mirror effect
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hands
        results = hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                features = preprocess_features(hand_landmarks.landmark)
                features = features.reshape(1, -1) 
                prediction = model.predict(features) 
                prediction = tf.nn.softmax(prediction).numpy() 

                
                predicted_class = np.argmax(prediction)
                
                
                confidence = np.max(prediction)
                gesture_buffer.append(class_names[predicted_class])

                if len(gesture_buffer) == 5 and all(g ==class_names[predicted_class] for g in gesture_buffer):
                    current_time=time.time()
                    if confidence > 0.7 and ((class_names[predicted_class] != last_letter) 
                                            or (current_time - last_detection_time > time_threshold)) :
                        recognized_text+=class_names[predicted_class]
                        last_letter = class_names[predicted_class]
                        last_detection_time = current_time

                                
                text_display.write("Recognized Text: " + "".join(recognized_text))
          
                cv2.putText(frame, f"Letter: {class_names[predicted_class]}",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        stframe.image(frame, channels="BGR")

                
camera.release()
hands.close()
cv2.destroyAllWindows()

