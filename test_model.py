from main import X_train,X_test,y_test,y_train
import matplotlib.pyplot as plt

from keras.api.models import Sequential
from keras.api.layers import *

# Simple model for testing
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Run a quick training test
model.fit(X_train, y_train, epochs=1, validation_data=(X_test, y_test), batch_size=32)

test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test accuracy:", test_acc)


predictions = model.predict(X_test)

# Visualize a few predictions
for i in range(5):
    plt.figure(figsize=(2, 2))
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {predictions[i].argmax()}, Actual: {y_test[i]}")
    plt.axis('off')
    plt.show()