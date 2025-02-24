# Enhancing-Emotional-Well-Being-through-ML-based-MER

This project implements machine learning techniques to classify emotions in music. By leveraging Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) with Long Short-Term Memory (LSTM), the project analyzes musical melodies and their emotional impact. The goal is to enhance Music Emotion Recognition (MER) with applications in music recommendation, mental health therapy, and mood-based personalization.

## Project Overview

- Uses the DEAM (Database for Emotional Analysis of Music) dataset for training.
- Implements Mel-Frequency Cepstral Coefficients (MFCC) for feature extraction.
- Compares CNN and RNN (LSTM) models for predicting Arousal-Valence values.
- Achieves 76.22% accuracy with the RNN (LSTM) model and 65.53% accuracy with the CNN model.

## Dataset

- **Source**: [DEAM Dataset](https://cvml.unige.ch/databases/DEAM/)
- **Format**: 2058 song snippets in `.mp3` format with pre-extracted MFCC features.
- **Emotion Labels**: Songs annotated with Arousal-Valence scores.

## Installation and Setup

### Clone the Repository
```bash
git clone https://github.com/yourusername/music-emotion-recognition.git
cd music-emotion-recognition
```
### Install Dependencies
- Ensure Python 3.8 or higher is installed, then install the required libraries:
```bash
pip install -r requirements.txt
```
### Prepare Dataset
- Download and extract the DEAM dataset and place it inside the project directory:
```bash
mkdir dataset
mv path_to_downloaded_dataset dataset/
```
## Running the Model
### Train the Models
- To train the CNN model:
```bash
python train_cnn.py
```
- To train the RNN (LSTM) model:
```bash
python train_rnn.py
```
### Test the Models
- To evaluate a trained model:
```bash
python evaluate.py --model cnn  # or rnn
```
### Predict Emotions for a New Song
```bash
python predict.py --file path/to/audio.mp3 --model rnn
```
## Results

| Model        | Accuracy |
|-------------|---------|
| RNN (LSTM)  | 76.22%  |
| CNN         | 65.53%  |

- The RNN (LSTM) model demonstrated better performance, showing its capability to process sequential features more effectively.

## Future Work
- Optimize model hyperparameters for improved accuracy.
- Experiment with larger and more diverse datasets.
- Develop a real-time emotion recognition system for music applications.




