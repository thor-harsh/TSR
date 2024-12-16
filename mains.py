import os
import numpy as np
import tensorflow as tf
import cv2
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from yolov5 import YOLOv5
from object_detection.utils import config_util, label_map_util, visualization_utils as viz_utils
from object_detection.builders import model_builder

# Define paths
DATASET_PATH = "Dataset"
LABEL_FILE = "labels.csv"
FRCNN_CONFIG_PATH = "path/to/faster_rcnn/pipeline.config"
FRCNN_CHECKPOINT_PATH = "path/to/faster_rcnn/checkpoint"
YOLO_WEIGHTS = "yolov5s.pt"

# Parameters
IMAGE_DIMENSIONS = (32, 32, 3)
BATCH_SIZE = 32
EPOCHS = 10
TEST_RATIO = 0.2
VALIDATION_RATIO = 0.2

# Load dataset
def load_data():
    images, classNo = [], []
    class_dirs = os.listdir(DATASET_PATH)
    for class_id, class_dir in enumerate(class_dirs):
        image_paths = os.listdir(os.path.join(DATASET_PATH, class_dir))
        for image_path in image_paths:
            img = cv2.imread(os.path.join(DATASET_PATH, class_dir, image_path))
            img = cv2.resize(img, (IMAGE_DIMENSIONS[0], IMAGE_DIMENSIONS[1]))
            images.append(img)
            classNo.append(class_id)
    images, classNo = np.array(images), np.array(classNo)
    return images, classNo, len(class_dirs)

images, classNo, noOfClasses = load_data()
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=TEST_RATIO)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=VALIDATION_RATIO)

# Preprocessing
def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img) / 255.0
    return img

X_train = np.array([preprocess(img) for img in X_train]).reshape(-1, 32, 32, 1)
X_validation = np.array([preprocess(img) for img in X_validation]).reshape(-1, 32, 32, 1)
X_test = np.array([preprocess(img) for img in X_test]).reshape(-1, 32, 32, 1)

y_train = to_categorical(y_train, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)

# CNN Model
def build_cnn():
    model = Sequential([
        Conv2D(60, (5, 5), activation='relu', input_shape=(32, 32, 1)),
        Conv2D(60, (5, 5), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(30, (3, 3), activation='relu'),
        Conv2D(30, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.5),
        Flatten(),
        Dense(500, activation='relu'),
        Dropout(0.5),
        Dense(noOfClasses, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Faster R-CNN Setup
def load_frcnn():
    configs = config_util.get_configs_from_pipeline_file(FRCNN_CONFIG_PATH)
    detection_model = model_builder.build(model_config=configs['model'], is_training=False)
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(FRCNN_CHECKPOINT_PATH, 'ckpt-0')).expect_partial()
    return detection_model

@tf.function
def detect_frcnn(image, detection_model):
    image, shapes = detection_model.preprocess(image)
    predictions = detection_model.predict(image, shapes)
    return detection_model.postprocess(predictions, shapes)

# YOLO Model
def load_yolo():
    return YOLOv5(YOLO_WEIGHTS)

# ResNet Model
def build_resnet():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=IMAGE_DIMENSIONS)
    base_model.trainable = False
    model = Sequential([
        base_model,
        Flatten(),
        Dense(128, activation='relu'),
        Dense(noOfClasses, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train and Evaluate Models
def evaluate_model(model, model_name):
    if model_name == 'cnn' or model_name == 'resnet':
        history = model.fit(
            X_train, y_train,
            validation_data=(X_validation, y_validation),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1
        )
        score = model.evaluate(X_test, y_test, verbose=0)
        print(f"{model_name.upper()} Test Accuracy: {score[1]:.2f}")
        return score[1]

    elif model_name == 'yolo':
        yolo = model
        results = []
        for img in X_test:
            results.append(len(yolo.predict(img)))  # Example metric
        print(f"YOLO Avg Detections: {np.mean(results):.2f}")
        return np.mean(results)

    elif model_name == 'frcnn':
        frcnn = model
        detections = []
        for img in X_test:
            detections.append(detect_frcnn(img, frcnn))  # Example metric
        print(f"FRCNN Avg Detections: {len(detections) / len(X_test):.2f}")
        return len(detections) / len(X_test)

# Initialize models
cnn_model = build_cnn()
resnet_model = build_resnet()
yolo_model = load_yolo()
frcnn_model = load_frcnn()

# Evaluate models
cnn_acc = evaluate_model(cnn_model, 'cnn')
resnet_acc = evaluate_model(resnet_model, 'resnet')
yolo_metric = evaluate_model(yolo_model, 'yolo')
frcnn_metric = evaluate_model(frcnn_model, 'frcnn')

# Print final results
print("\nFinal Results:")
print(f"CNN Accuracy: {cnn_acc}")
print(f"ResNet Accuracy: {resnet_acc}")
print(f"YOLO Metric: {yolo_metric}")
print(f"Faster R-CNN Metric: {frcnn_metric}")
