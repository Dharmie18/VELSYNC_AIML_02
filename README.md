# Horses vs Humans Image Classifier

## What This Script Does

This Python script builds and trains a Convolutional Neural Network (CNN) to classify images as containing either a "Horse" or a "Human". It utilizes the TensorFlow library to handle the deep learning pipeline, from loading the dataset to training the model and visualizing the results. The script creates a model that learns distinguishing features of horses and humans from a dataset and then tests its ability to correctly label new images.

## How It Works

1. **Data Loading**: It downloads and loads the "Horses or Humans" dataset using `tensorflow_datasets`.
   - The data is split into training and testing sets.
   - Images are resized to a standard 128x128 pixels and normalized (pixel values scaled between 0 and 1).

2. **Model Architecture**:
   - The script constructs a CNN using `tf.keras.Sequential`.
   - It layers Convolutional layers (to detect edges and features) with Max Pooling layers (to reduce data size).
   - A Flatten layer converts the 2D features into 1D, followed by Dense layers for the final classification.
   - The final layer uses a sigmoid activation function to output a probability (0 for Horse, 1 for Human).

3. **Training**:
   - The model is compiled with the Adam optimizer and binary cross-entropy loss function.
   - It trains for 10 epochs (iterations over the dataset).

4. **Visualization**:
   - **Training History**: It plots the accuracy and loss over time and saves it as `training_history.png`.
   - **Sample Predictions**: It takes a batch of test images, predicts their labels, and creates a visual grid showing the image, the predicted label, and the true label, saved as `sample_predictions.png`. (Note: The script finishes after showing one batch).

## Dependencies

To run this script, you need these Python libraries:
- `tensorflow` (for deep learning)
- `tensorflow-datasets` (to easily load the dataset)
- `matplotlib` (for plotting images and graphs)

## Setup Steps

1. **Install Python**: Ensure Python 3 is installed.

2. **Install Dependencies**: Run the following command in your terminal:
   ```bash
   pip install tensorflow tensorflow-datasets matplotlib
   ```

3. **Download the Script**: Ensure `horses_vs_humans.py` is in your working directory.

## How to Test It Locally

1. Open your terminal or command prompt.

2. Navigate to the folder containing the script:
   ```bash
   cd Desktop/VELSYNC_AIML_INTERNSHIP/VELSYNC_AIML_02
   ```

3. Run the script:
   ```bash
   python horses_vs_humans.py
   ```

## Expected Output

1. **Console Output**:
   - You will see logs indicating the dataset is downloading (first run only).
   - Training progress bars for each of the 10 epochs.
   - A final message: `Task 2 Complete! Horses vs Humans classifier trained.`

2. **Files Created**:
   - `training_history.png`: A graph showing how accuracy improved during training.
   - `sample_predictions.png`: A grid of images showing how the model performed on test data (Green text = Correct, Red text = Incorrect).

3. **Popup Windows**:
   - Two windows will pop up during execution showing the plots (close them to ensure the script finishes).
