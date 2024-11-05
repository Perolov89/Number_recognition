
from tensorflow import keras
import matplotlib.pyplot as plt
# load data
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# check data shapes
# print("Training data shape:", X_train.shape) # 60000,28,28
# print("Testing data shape:", X_test.shape) # 10000, 28, 28

# for i in range(5):
#     prints 5 images to check if everything works as excpected
#     plt.figure(figsize=(2, 2))
#     plt.imshow(X_train[i], cmap='gray')
#     plt.title(f"Label: {y_train[i]}")
#     plt.axis('off')
#     plt.show()


# normalize the pixel values to be between 0 and 1
X_train = X_train / 255.0
X_test = X_test / 255.0

# reshapes the data to add channel dimension 
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

print("Training data shape after reshaping:", X_train.shape)
print("Testing data shape after reshaping:", X_test.shape)

