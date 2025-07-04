# IMAGE-CLASSIFICATION-MODEL

*COMPANY* = CODTECH IT SOLUTIONS

*NAME* = G DEVA DHEERAJ REDDY

*INTERN ID*= CT04DF2074

*DOMAIN*=MACHINE LEARNING

*DURATION*=4 WEEKS

*MENTOR* =NEELA SANTOSH

Image Classification using CNN on CIFAR-10
The notebook presents a deep learning-based image classification project using TensorFlow and Keras, two of the most widely used frameworks in artificial intelligence (AI) and machine learning (ML). The goal is to classify images from the CIFAR-10 dataset, which consists of 60,000 32x32 color images across 10 distinct categories, including airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks.

1. Importing Libraries
The necessary libraries such as tensorflow, keras, and matplotlib are imported. TensorFlow is used for building and training the CNN model, while matplotlib is used for visualization.

python
Copy
Edit
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
2. Loading and Preprocessing CIFAR-10
The dataset is loaded using Keras's built-in CIFAR-10 loader. It automatically splits the dataset into training and testing sets:

python
Copy
Edit
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
To improve training efficiency, pixel values are normalized by dividing by 255.0 so that they range from 0 to 1:

python
Copy
Edit
x_train, x_test = x_train / 255.0, x_test / 255.0
3. Building the CNN Model
The core part of this notebook is the design and construction of the Convolutional Neural Network (CNN) using Keras’ Sequential API. The model includes:

Three convolutional layers, each followed by ReLU activation.

MaxPooling layers after the first two convolutional layers to reduce dimensionality.

A flattening layer to convert the 3D feature maps to 1D.

Two dense layers, with the final layer using softmax activation for multi-class classification.

python
Copy
Edit
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
4. Compiling the Model
The model is compiled using:

Adam optimizer, known for its adaptive learning rate.

Sparse categorical cross-entropy as the loss function (appropriate for integer-labeled classes).

Accuracy as the performance metric.

python
Copy
Edit
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
5. Training the Model
The model is trained using the .fit() method for 10 epochs on the training dataset. The training history (loss and accuracy per epoch) is stored for later visualization.

python
Copy
Edit
history = model.fit(x_train, y_train, epochs=10, 
                    validation_data=(x_test, y_test))
6. Evaluation and Visualization
After training, the model’s performance is evaluated on the test dataset. Plots of training and validation accuracy and loss are typically created to analyze overfitting or underfitting.

Conclusion
This notebook provides a complete pipeline for image classification using CNNs. It demonstrates the effectiveness of convolutional architectures for visual recognition tasks and highlights the use of Keras APIs for fast prototyping. For further improvement, one could apply techniques like data augmentation, dropout, or experiment with deeper networks and transfer learning.

*OUTPUT*=



