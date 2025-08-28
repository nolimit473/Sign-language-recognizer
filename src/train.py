import pandas as pd
import numpy as np
from tensorflow import keras
import keras
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense, Conv1D, Flatten, Dropout, LSTM
import tensorflow as tf
from keras import layers


# Load the dataset
df = pd.read_csv("landmarks.csv")
df.dropna(inplace=True)
#print(df.isna().sum())
# Replace with the actual CSV filename

# Split features and labels
X = df.iloc[:,:-1].values
y = df.iloc[:, -1].values  
scale = np.linalg.norm(X.reshape(-1, 21, 3)[:, 9, :] - X.reshape(-1, 21, 3)[:, 0, :], axis=1)
scale = scale[:, np.newaxis] 
X = X.reshape(-1, 21, 3)

wrist = X[:, 0:1, :]          # shape (num_samples, 1, 3)
X = X - wrist    
 

X = X.reshape(-1, 63)
X = X / scale


np.save("landmark1_features.npy",X)

# All columns except the last (landmarks)
# Last column (labels)

gesture_labels=np.array(y)
np.save("gesture1_labels.npy",gesture_labels)

features = np.load("landmark1_features.npy",allow_pickle=True)  # Hand landmarks extracted with MediaPipe
labels = np.load("gesture1_labels.npy",allow_pickle=True)  # 0 for thumbs up, 1 for thumbs down

label_mapping = {"A": 0, "B": 1,"C":2,"D":3,"E":4,"F":5,"G":6, "H":7, "I":8, "J":9, "K":10, "L":11, "M":12, "N":13, "O":14, "P":15,
                 "Q":16, "R":17, "S":18, "T":19, "U":20, "V":21, "W":22, "X":23,
                 "Y":24, "Z":25, "Thankyou":26 ,"Yes":27}

#print("Unique labels in dataset:", set(labels))


numeric_labels = np.array([label_mapping[label] for label in labels])
np.save("numeric1_gesture_labels.npy", numeric_labels)

num_labels1=np.load("numeric1_gesture_labels.npy",allow_pickle=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, num_labels1, test_size=0.1, random_state=42)


# Reshape data for CNN (since CNNs expect 3D data)
#X_train = X_train.reshape(-1, 21, 3, 1)  # (samples, 21 landmarks, 3 coordinates, 1 channel)
#X_test = X_test.reshape(-1, 21, 3, 1)

tf.random.set_seed(42)
model = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(63, 1)),
    Conv1D(128, 3, activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(28, activation='linear')
])


model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), 
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))



# Train the model
# X_train = np.expand_dims(X_train, axis=-1)  # Add channel dimension
# X_test = np.expand_dims(X_test, axis=-1)
model.fit(X_train, y_train, epochs=40, validation_data=(X_test, y_test))

# Save the trained model
model.save("augmented_rotation.h5",save_format="h5")
