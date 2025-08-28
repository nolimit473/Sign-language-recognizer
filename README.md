# Sign-language-recognizer
Hand gesture recognition system (A–Z) with Python, TensorFlow, and MediaPipe

This project is a real-time Sign Language Interpreter that recognizes 
hand gestures (A–Z + extra symbols) using hand landmarks from MediaPipe
and a TensorFlow/Keras model.

## Tech Stack
- Python, TensorFlow/Keras
- MediaPipe (for hand landmark extraction)
- OpenCV (for real-time webcam integration)
- Pandas & NumPy (for dataset handling)
- Matplotlib (for training/landmark visualization)
- Streamlit

## Dataset ##
- Custom dataset of **3332** samples.
- Extracted **21 hand landmarks (x,y,z)** per frame.
- Balanced via **augmentation**.
- **Note**: gestures for (J,Z,etc) that involve movement have been replaced with alternative static gestures

**Results**
The model is able to recognize most hand gestures in real-time.
Some gestures are harder to distinguish due to their similarity and model struggles in very bright/poor lighting conditions.
Performance improves significantly with data augmentation.
Future work includes collecting more samples and improving the deep learning architecture.

**How to Run**
```bash
git clone https://github.com/nolimit473/Sign-language-recognizer.git
cd Sign-language-recognizer
pip install -r requirements.txt
python src/train.py
streamlit run src/sign.py
