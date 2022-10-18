import numpy as np

#scalar
x = np.array(12)
print(x.ndim)

#vector
x = np.array([12, 3, 6, 14, 7])
print(x.ndim)

#matrix
x = np.array([[5, 78, 2, 34, 0],
             [6, 79, 3, 35, 1],
             [7, 80, 4, 36, 2]])
print(x.ndim)

#3D tenser
x = np.array([[[5, 78, 2, 34, 0],
             [6, 79, 3, 35, 1],
             [7, 80, 4, 36, 2]],
             [[5, 78, 2, 34, 0],
             [6, 79, 3, 35, 1],
             [7, 80, 4, 36, 2]],
             [[5, 78, 2, 34, 0],
             [6, 79, 3, 35, 1],
             [7, 80, 4, 36, 2]]])
print(x.ndim)


## show the fifth image
import matplotlib.pyplot as plt
digit = train_images[4]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()


# MNIST
import keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

## data preprocess
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

## network
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28, )))
network.add(layers.Dense(10, activation='softmax'))

## compile
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

## train
network.fit(train_images, train_lavels, epochs=5, batch_size=128)

### dimensionality, array shape, data type
print(train_images.ndim, train_images.shape, train_images.dtype)

### slicing
my_slice = train_images[10:100]
print(my_slice.shape)  ## 90, 28, 28
my_slice = train_images[10:100, :, :]  #same
my_slice = train_images[10:100, 0:28, 0:28]  #same

my_slice = train_images[:, 14:, 14:]
my_slice = train_images[:, 7:-7, 7:-7]

### batch
batch = train_images[:128]
batch = train_images[128:256]
batch = train_images[128 * n : 128 * (n+1)]


## 원소별 연산, element-wise operation
### relu
def naive_relu(x):
  assert len(x.shape) == 2

  x = x.copy()  # 입력 텐서 자체를 바꾸지 않도록 복사
  for i in range(x.shape[0]):
    for j in range(x.shape[1]):
      x[i, j] = max(x[i, j], 0)
  
  return x

### add
def naive_add(x, y):
  assert len(x.shape) == 2  # x, y는 2D 넘파이 배열이다.
  assert x.shape == y.shape

  x = x.copy()
  for i in range(x.shape[0]):
    for j in range(x.shape[1]):
      x[i, j] += y[i, j]

  return x

### numpy를 활용해 구현
z = x + y  # 원소별 덧셈

z = np.maximum(z, 0.) # 원소별 relu

## 브로드캐스팅
def naive_add_matrix_and_vector(x, y):
  assert len(x.shape) == 2
  assert len(y.shpae) == 1
  assert x.shape[1] == y.shape[0]

  x = x.copy()
  for i in range(x.shape[0]):
    for j in range(x.shape[1]):
      x[i, j] += y[j]

  return x

### numpy 활용 브로드캐스팅
x = np.random.random((64, 3, 32, 10))
y = np.random.random((32, 10))

z = np.maximum(x, y)


## 텐서 점곱
z = np.dot(x, y)

## 텐서 크기 변환, 전치
x = np.array([[0., 1.],
             [2., 3.],
             [4., 5.]])

x = x.reshape((6, 1))

x = np.transpose(x)