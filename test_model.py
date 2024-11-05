from main import X_train,X_test,y_test,y_train

# for some reason i have to import like this...
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

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

# Run a quick training test (use fewer epochs for a quick check)
model.fit(X_train, y_train, epochs=1, validation_data=(X_test, y_test), batch_size=32)