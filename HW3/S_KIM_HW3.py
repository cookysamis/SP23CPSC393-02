import warnings
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf
import tensorflow.keras as kb
from tensorflow.keras import backend
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix

np.random.seed(42)
tf.random.set_seed(42)

# data downloaded from Kaggle
X_train_dir = '../content/Skin_Data/Cancer/Training' # cancer train
y_train_dir = '../content/Skin_Data/Non_Cancer/Training'# non-cancer train
X_test_dir = '../content/Skin_Data/Cancer/Testing' # cancer test
y_test_dir = '../content/Skin_Data/Non_Cancer/Testing' # non-cancer test

train_data = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
test_data = ImageDataGenerator(rescale=1./255)

X_train = train_data.flow_from_directory(X_train_dir,
                                                           target_size=(224, 224),
                                                           batch_size=32,
                                                           class_mode='binary')

y_train = train_data.flow_from_directory(y_train_dir,
                                                              target_size=(224, 224),
                                                              batch_size=32,
                                                              class_mode='binary')

X_test = test_data.flow_from_directory(X_test_dir,
                                                         target_size=(224, 224),
                                                         batch_size=32,
                                                         class_mode='binary')

y_test = test_data.flow_from_directory(y_test_dir,
                                                            target_size=(224, 224),
                                                            batch_size=32,
                                                            class_mode='binary')

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(512, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(lr=1e-4),
              metrics=['accuracy'])

# cancer
X_history = model.fit(X_train,
                           steps_per_epoch=len(X_train),
                           epochs=100,
                           validation_data=X_test,
                           validation_steps=len(X_test))

# non-cancer
y_history = model.fit(y_train,
                           steps_per_epoch=len(y_train),
                           epochs=100,
                           validation_data=y_test,
                           validation_steps=len(y_test))

train_loss, train_acc = model.evaluate(X_train)
print('Train Acc for cancer:', train_acc)
train_loss, train_acc = model.evaluate(y_train)
print('Train Acc for non-cancer:', train_acc)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(X_test)
print('Test Acc for cancer:', test_acc)
test_loss, test_acc = model.evaluate(y_test)
print('Test Acc for non-cancer:', test_acc)

X_test.reset()
y_test.reset()
y_pred_test_cancer = model.predict(X_test)
y_pred_test_noncancer = model.predict(y_test)

y_pred_test_cancer[y_pred_test_cancer <= 0.5] = 0
y_pred_test_cancer[y_pred_test_cancer > 0.5] = 1
y_pred_test_noncancer[y_pred_test_noncancer <= 0.5] = 0
y_pred_test_noncancer[y_pred_test_noncancer > 0.5] = 1

print('Confusion Matrix for cancer:')
print(confusion_matrix(X_test.classes, y_pred_test_cancer))
print('Confusion Matrix for non-cancer:')
print(confusion_matrix(y_test.classes, y_pred_test_noncancer))