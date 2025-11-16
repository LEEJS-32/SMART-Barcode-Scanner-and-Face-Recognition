# How to run

~~ in terminal

1. go to file directory

cd Image Prepocessing

2. Create virtual environment

python -m venv venv

3. Enable virtual environment

.\venv\Scripts\Activate.ps1

4.install all the library needed

pip install -r requirements.txt

5.run streamlit ui

python -m streamlit run src/app.py

if face error C:\Users\shenl\OneDrive\Desktop\Image Preprocessing\venv\Scripts\python.exe: No module named streamlit

.\venv\Scripts\pip.exe install streamlit opencv-contrib-python numpy pillow


6. run again 

 .\venv\Scripts\streamlit.exe run src\app.py

# Dataset used
We used the Face Recognition Dataset by Vasuki Patel on Kaggle: “Face Data of 31 different classes.” 
(URL: http://kaggle.com/datasets/vasukipatel/face-recognition-dataset).

# Exploration Data Analysis
<img width="930" height="474" alt="image" src="https://github.com/user-attachments/assets/f525d41c-6c7d-4784-8b27-a5684c0f3bbb" />

# Data Preprocess
Grayscale -> CLAHE(for brightness normalization) -> Resized(into 112x112pixel)
<img width="973" height="573" alt="smart-face" src="https://github.com/user-attachments/assets/4ada3425-f49d-4f3f-ad69-b595b160d421" />

# Train Models
Explanation for models used:
<img width="1116" height="976" alt="image" src="https://github.com/user-attachments/assets/c9f6b647-91d7-4bdb-810b-ace63b9bf306" />

# Model Evaluation
<img width="1175" height="335" alt="image" src="https://github.com/user-attachments/assets/b0d9f663-cce1-443d-b04d-8d71973a62a3" />


# Preview of page (Streamlit App)
 <img width="687" height="526" alt="image" src="https://github.com/user-attachments/assets/75e1dc44-6ef8-4de9-a437-7781904e7ece" />


