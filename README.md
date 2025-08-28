# Sign-language-recognizer
Hand gesture recognition system (A–Z) with Python, TensorFlow, and MediaPipe

This project is a real-time Sign Language Interpreter that recognizes 
hand gestures (A–Z + extra symbols) using hand landmarks from MediaPipe
and a TensorFlow/Keras model.

##Tech Stack
- Python, TensorFlow/Keras
- MediaPipe (for hand landmark extraction)
- OpenCV (for real-time webcam integration)
- Pandas & NumPy (for dataset handling)
- Matplotlib (for training/landmark visualization)

##Dataset
- Custom dataset of **3333** samples.
- Extracted **21 hand landmarks (x,y,z)** per frame.
- Balanced via **augmentation**.

##Results
-The model is able to recognize most hand gestures in real-time.
-Some gestures are harder to distinguish due to their similarity.
-Performance improves significantly with data augmentation.
-Future work includes collecting more samples and improving the deep learning architecture.


