# Cat vs Dog Image Classification Using CNN
# 1. Installations
This project was implemented in Python using Jupyter Notebook on Anaconda. The relevant Python libraries and packages required for this project are as follows:

* tensorflow (for building and training the CNN model)
* keras (integrated within TensorFlow for easy neural network implementation)
* numpy (for numerical computations)
* matplotlib (for data visualization)
* os (to handle file paths and directory operations)
* ImageDataGenerator (from tensorflow.keras.preprocessing.image, for data augmentation and preprocessing)
* image (for loading and preprocessing single images for prediction)

# 2. Project Motivation
This project involves building and training a Convolutional Neural Network (CNN) to classify images of cats and dogs. The project serves as a practical implementation of deep learning techniques for image classification tasks.

The main objectives of this project are:

1. Train a CNN Model: Build and train a CNN model on a dataset containing:
    * 4000 images of cats
    * 4000 images of dogs
  
2. Evaluate the Model: Test the model on a separate test dataset containing:
   * 1000 images of cats
   * 1000 images of dogs

3.Make Predictions: Use the trained model to classify unseen images (provided in a single_prediction folder) as either cat or dog 


# 3. File Descriptions
This project contains the following files:


* **Cat_Dog_Classification.ipynb:**
  
This is the Jupyter Notebook that contains the complete implementation of the project, including:

    * Model definition
    * Data preprocessing
    * Model training and evaluation
    * Prediction on unseen images
* dataset/training_set/:
  
This folder contains the training images organized into subfolders:

    * cats/
    * dogs/
* dataset/test_set/:
  
This folder contains the test images organized into subfolders:

    * cats/
    * dogs/
* single_prediction/:
  
This folder contains unseen images for prediction, which will be classified as either cat or dog using the trained model.

* cat_dog_cnn_model.h5:
  
This file contains the saved CNN model after training.

# 4. Results
The CNN model was successfully trained and evaluated on the dataset. Key observations include:

The model achieved a reasonable accuracy on both the training and test datasets.
The performance of the model can be visualized using accuracy and loss plots over training epochs.
Key Steps Taken:

Used data augmentation techniques such as rescaling, zooming, and horizontal flipping for better generalization.
Built a 3-layer Convolutional Neural Network with pooling, dropout, and fully connected dense layers.
Achieved predictions on unseen images in the single_prediction folder.
# 5. Licensing, Authors, Acknowledgements, etc.
The image dataset was sourced for educational purposes.
The project was completed as part of a machine learning class project.
The implementation was done using TensorFlow/Keras in Python.
