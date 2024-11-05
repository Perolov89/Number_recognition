# Number_recognition

Small project to try tensorflow 
-------------------------------

Why Normalize the Data?

Definition: Normalization means scaling the data so that its values fall within a certain range, typically between 0 and 1.

Reason: The pixel values in the MNIST dataset are between 0 and 255 (grayscale images). Neural networks learn better and converge faster when input data is scaled to a smaller range (e.g., [0, 1]). This prevents large input values from causing inefficient training and numerical instability.

How: Divide each pixel value by 255 to transform it from a range of [0, 255] to [0, 1].

Summary: Normalization helps the model train faster and more effectively by scaling down large input values.



Reshape the Data

Reason: When working with convolutional neural networks (CNNs), the input shape should match the expected input format. The MNIST dataset images are originally in 2D arrays (28x28). CNNs typically expect a 3D shape (height, width, channels), where:
    Height and Width: The size of the image (28x28 for MNIST).
    Channels: The depth of the image; for grayscale images, this is 1. For RGB images, it would be 3.

How: Reshape the dataset to add an extra dimension for the channel.

Summary: Reshaping prepares the data to be fed into a CNN by adding a channel dimension, which indicates the image depth.



Why Reshaping is Important

Matching Input Requirements: Most deep learning frameworks expect the input data to have a specific shape. For example, in TensorFlow, CNNs expect a 4D input (batch size, height, width, channels). Without reshaping, the network won't process the data correctly and could raise errors.
Compatibility: If you use dense (fully connected) layers only, reshaping isn't necessary. But for CNNs, which analyze images using filters that slide over the spatial dimensions, the channel dimension is essential.

Visualizing the Change: 

Before reshaping:
    Each image is a 2D array of shape (28, 28).
After reshaping:
    Each image becomes a 3D array of shape (28, 28, 1), making it compatible with CNN input.
