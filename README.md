# Faulty-Tyres-Classification

#FaultFindy: Intelligent System to Predict Faulty Tyres in Manufacturing FaultFindy is an intelligent system developed using deep learning techniques to predict faulty tyres in a manufacturing setting by analyzing various tyre manufacturing parameters and process data. The project focuses on classifying tyre images into two categories: Defective and Good.

Table of Contents Project Overview Technologies Used Installation Instructions Dataset Model Training Evaluation and Results Future Work Project Overview The goal of the FaultFindy system is to identify faulty tyres from a set of images taken from a manufacturing line. Using Convolutional Neural Networks (CNN) and the EfficientNetB4 pre-trained model, the system classifies tyre images as either Good or Defective. The model is trained on a balanced dataset, with an accuracy of 95% after fine-tuning.

Technologies Used Python 3.8+ TensorFlow 2.x Keras Matplotlib Plotly NumPy Pandas Installation Instructions To run this project on your local machine, follow these steps:

Clone the repository:
git clone [https://github.com/your-username/FaultFindy.git](https://github.com/shubhamv8308/Faulty-Tyres-Classification)
cd FaultFindy

Create a virtual environment:
python -m venv venv source venv/bin/activate # For Linux/MacOS venv\Scripts\activate # For Windows

Install the required dependencies:
pip install -r requirements.txt Make sure you have access to the dataset, and place it in the appropriate folder:

Dataset Structure: /defective /good You can also modify the paths in the code to suit your environment.

Dataset : https://kh3-ls-storage.s3.us-east-1.amazonaws.com/Updated Project guide data set/Faultfindy.zip The dataset consists of images of Defective and Good tyres. The images are classified into two categories:

Defective Tyres (Class 0) Good Tyres (Class 1) The dataset was split into training and validation subsets with a ratio of 87.5% for training and 12.5% for validation.

Model Training The project uses the EfficientNetB4 model, which is pre-trained on the ImageNet dataset, and fine-tunes it for the tyre defect classification task. The training process includes image augmentation, optimization with the Nadam optimizer, and binary cross-entropy loss for classification.

Data preprocessing involves resizing the images to 224x224 pixels, normalizing the pixel values, and augmenting the dataset. The model was first trained with frozen base layers, followed by fine-tuning the last layers for better accuracy. Evaluation and Results Initial Model Accuracy: 90% After Fine-Tuning Accuracy: 91% The model was evaluated on the validation set, achieving a high accuracy of 91% in predicting whether a tyre is Good or Defective.

Future Work Further Model Optimization: Experiment with different architectures like ResNet and Inception. Data Augmentation: Increase the dataset size using synthetic data generation techniques. Real-Time Integration: Implement the system for real-time classification in the manufacturing line.
