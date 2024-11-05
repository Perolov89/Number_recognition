
from tensorflow import keras
import matplotlib.pyplot as plt
# load data
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# check data shapes
print("Training data shape:", X_train.shape) # 60000,28,28
print("Testing data shape:", X_test.shape) # 10000, 28, 28

# prints 5 images to check if everything works as excpected
for i in range(5):
    plt.figure(figsize=(2, 2))
    plt.imshow(X_train[i], cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')
    plt.show()