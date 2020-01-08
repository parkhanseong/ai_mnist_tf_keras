# Install TensorFlow
try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass

import tensorflow as tf
import keras
import numpy as np
from google.colab.patches import cv2_imshow

print(tf.__version__) #version check

mnist = tf.keras.datasets.mnist

# 학습데이터, 평가데이터 불러오기
# x_train, x_test --> 손 글씨 이미지
# y_train, y_test --> 디지털 숫자
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 데이터 전처리
# 이미지 값의 범위를 [0, 255] 에서 [0.0, 1.0] 으로 줄인다 
x_train, x_test = x_train / 255.0, x_test / 255.0

print(x_train.shape, y_train.shape) #check your data shape

# tensorflow 에서 지원하는 데이터 셋
dir(tf.keras.datasets)

# 텐서플로우의 케라스를 이용하면 쉽게 모델을 만들 수 있다
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28), name="InputLayer"),
  tf.keras.layers.Dense(128, activation='relu', name="DenseLayer"),
  tf.keras.layers.Dropout(0.2, name="Dropout"),
  tf.keras.layers.Dense(10, activation='softmax', name="OutputLayer")
])

# adam 최적화기와 sparse_categorical_crossentropy 손실함수는 다중 분류 문제에서 자주 사용된다
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary() #After building your model, you can see summarization of the architecture
tf.keras.utils.plot_model(model)

# model 학습 시키기
model.fit(x_train, y_train, batch_size = 100, epochs=5)

# test 데이터로 모델 평가하기
model.evaluate(x_test,  y_test, verbose=2)

model_cnn = tf.keras.models.Sequential()
model_cnn.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model_cnn.add(tf.keras.layers.MaxPooling2D((2, 2)))
model_cnn.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))

model_cnn.add(tf.keras.layers.Flatten())
model_cnn.add(tf.keras.layers.Dense(64, activation='relu'))
model_cnn.add(tf.keras.layers.Dense(10, activation='softmax'))

model_cnn.summary()

model_cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# 데이터와 입력레이어의 차원이 맞지 않기때문에 에러 발생!
model_cnn.fit(x_train, y_train, epochs=5)
model_cnn.evaluate(x_test,  y_test, verbose=2)

# 컨볼루션 레이어에 이미지를 넣어주기 위해서는, 반드시 이미지는 (N 개수, W 너비, H 높이, C 채널) 형태를 가지고 있어야 한다
# It's a promise with TensorFlow!

# 훈련 데이터의 차원 확장
x_train = tf.expand_dims(x_train, axis = 3)
x_test = tf.expand_dims(x_test, axis=3)

# 데이터 형태 확인
print(x_train.shape, y_train.shape)

model_cnn.fit(x_train, y_train, batch_size = 100, epochs=5)
model_cnn.evaluate(x_test,  y_test, verbose=2)
print(model_cnn.layers)

first_layer = model_cnn.layers[0]
print(first_layer)

print(first_layer.get_weights()[0].shape)
print(first_layer.get_weights()[1].shape)

first_layer.get_weights()[0][:,:,0,0].shape

from matplotlib import pyplot as plt

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(first_layer.get_weights()[0][:,:,0,i], cmap='gray')
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
plt.show()

model_cnn.save("model.h5")
load_model = tf.keras.models.load_model("/content/model.h5")

import cv2
num = cv2.imread("num.png")
num = cv2.cvtColor(num, cv2.COLOR_BGR2GRAY)
num = cv2.resize(num, (28, 28), interpolation=cv2.INTER_AREA)
num = num.reshape(1, 28, 28, 1)

prediction = load_model.predict_classes(num)
print(prediction[0])