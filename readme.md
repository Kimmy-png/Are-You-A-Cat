# are you a cat?

🐱Welcome to the "Are You A Cat?" project! This is an interactive web application that uses a deep learning model to classify whether an uploaded image contains a cat or a human. The app not only provides a prediction but also displays the percentage of similarity to a cat, according to the model.

This project serves as an end-to-end demonstration of a machine learning workflow, from data collection and model training on Kaggle to deployment as a web application using Streamlit.


# 🚀 Live DemoTry the application live here:

[are-you-a-cat-aezkxcfim9p7umteca3brb](https://are-you-a-cat-aezkxcfim9p7umteca3brb.streamlit.app/) ✨ 


# Key Features

- Image Classification: Capable of distinguishing between images of cats and humans with high accuracy.

- Similarity Percentage: Provides a confidence score (percentage) of how similar an image is to a cat.

- Interactive Web Interface: Built with Streamlit for an easy-to-use and responsive user experience.

- Deep Learning Model: Powered by a Convolutional Neural Network (CNN) trained using TensorFlow (Keras).


# 🛠️ Tech Stack

- Language: Python

- Machine Learning: TensorFlow (Keras)

- Web Application: Streamlit

- Image Processing: Numpy, Pillow

- Training Environment: Kaggle Notebooks with GPU

- Deployment: Streamlit Community Cloud

- Version Control: Git & GitHub


# ⚙️ Local Setup and Installation

If you want to run this application on your own computer, follow these steps:

Clone this repository:
```bash
git clone [https://github.com/](https://github.com/)[YOUR_USERNAME]/cat-classifier-app.git 
```
Navigate into the project 
```bash
directory:cd cat-classifier-app
```
Create and activate a virtual environment 
```bash
(optional but recommended):python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```
Install the required dependencies:
```bash
pip install -r requirements.txt
```


# 🏃 How to Run the App
Once all dependencies are installed, run the following command in your terminal:
```bash
streamlit run app.py
```
Open your browser and navigate to 
http://localhost:8501 to see the application running.


# 📂 Project Structurecat-classifier-app/
├── app.py              # Main Streamlit application script

├── model.keras         # The pre-trained machine learning model

├── requirements.txt    # List of Python dependencies

├── train.py            # Script used to train the model on Kaggle (for documentation)

└── README.md           # You are reading it!


# 📊 Dataset
This model was trained on a combination of two public datasets available on Kaggle. The dataset was balanced during preprocessing to ensure an equal number of samples for each class, preventing bias in the model.
