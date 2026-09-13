# Heart Disease Prediction App

A Streamlit machine-learning application that estimates heart-disease risk from user-provided health measurements.

## Features

- Interactive Streamlit input form
- KNN model prediction
- Input scaling using the saved scaler
- Estimated risk probability when supported by the model

## Project Files

- `app.py` - Streamlit application
- `training/heart.ipynb` - model exploration and training notebook
- `training/heart.csv` - heart-disease training dataset
- `knn_heart.pkl` - trained prediction model
- `scaler.pkl` - feature scaler used during training
- `columns.pkl` - expected model input columns
- `requirements.txt` - Python dependencies

## Training Materials

The `training` folder contains the notebook and dataset used to explore the data,
prepare features, train the models, and save the model artifacts. Open the notebook
from inside the `training` folder so its `heart.csv` relative path works correctly.

The deployed Streamlit app does not need to load the dataset because it uses the
already-trained `.pkl` files.

## Run Locally

1. Create and activate a virtual environment:

   ```bash
   python3 -m venv myenv
   source myenv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the application:

   ```bash
   streamlit run app.py
   ```

The application will open in your browser at the local Streamlit URL.

## Deploy

This project can be deployed on Streamlit Community Cloud or another service that supports Streamlit applications.

Upload or commit these files:

- `app.py`
- `training/heart.ipynb`
- `training/heart.csv`
- `knn_heart.pkl`
- `scaler.pkl`
- `columns.pkl`
- `requirements.txt`

Do not upload the `myenv` virtual-environment folder. The deployment platform installs dependencies from `requirements.txt`.

## Important Notice

This application is for educational purposes only. Its prediction is not medical advice or a diagnosis. Consult a qualified healthcare professional for medical decisions.

also add dataset 