import os
import pickle
import mediapipe as mp
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Data directory
DATA_DIR = './data'

# Prepare data and labels
data = []
labels = []
expected_landmarks = 21  # Expected number of landmarks per hand
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        x_ = []
        y_ = []

        # Read and process image
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        if img is None:
            print(f"Failed to read image: {img_path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)

                # Ensure we have the expected number of landmarks
                if len(x_) == expected_landmarks and len(y_) == expected_landmarks:
                    for lm in hand_landmarks.landmark:
                        data_aux.append(lm.x - min(x_))
                        data_aux.append(lm.y - min(y_))
                else:
                    print(f"Unexpected number of landmarks in {img_path}")
                    continue  # Skip this image if the landmark count is unexpected

            if data_aux:
                data.append(data_aux)
                labels.append(dir_)

# Ensure consistent data length
data = np.array([np.array(d) for d in data])
labels = np.array(labels)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train the RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Evaluate the model
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print(f'{score * 100}% of samples were classified correctly!')

# Save the trained model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
