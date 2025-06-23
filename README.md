# Facial-emotion-recognition
A deep learning project using CNN to detect facial emotions from images. Useful in health, education, and retail sectors for emotion-based decision-making.

This project implements a **Facial Emotion Recognition** system using **Convolutional Neural Networks (CNNs)** to detect human emotions from facial expressions in images. The system is trained on the FER-2013 dataset and classifies emotions into categories like Happy, Sad, Angry, Fear, Surprise, Disgust, and Neutral.

## 📁 Project Structure  
- `model.py`: CNN model definition and training  
- `test.py`: Script for emotion detection from images  
- `fer2013.csv`: Dataset used for training (FER-2013)  
- `README.md`: Project documentation  
- `requirements.txt`: List of Python dependencies  
- `images/`: Folder containing sample emotion images  
- `trained_model.h5`: Saved Keras model  

## 💡 Key Features  
- Built using CNN with Keras and TensorFlow  
- Trained on 48x48 grayscale images from FER-2013 dataset  
- Emotion categories: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral  
- Supports real-time webcam-based detection  
- Data preprocessing and augmentation applied  
- Achieves ~60%+ accuracy on test set  

## 📊 Dataset  
**FER-2013**: A facial expression dataset consisting of 35,000+ images across 7 categories.  
- Format: 48x48 grayscale images  
- Source: [FER-2013 via Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)  

## 🧪 Technologies Used  
- Python  
- TensorFlow / Keras  
- OpenCV  
- Numpy / Pandas / Seaborn / Matplotlib  
- ImageDataGenerator (for augmentation)  

## 🧠 Applications  
- Retail: Customer emotion feedback in marketing  
- Healthcare: Emotion monitoring in therapy or elderly care  
- Education: Student engagement tracking  
- Smart Cars: Driver alertness detection  

## 👨‍💻 Author  
**Akiti Sri Kalyan Reddy**  
B.Tech – Data Science and Artificial Intelligence  
ICFAI University, Hyderabad  
