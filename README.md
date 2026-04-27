# Agentic-AI Cybersecurity Threat Detection

This project implements the foundational **Detection Agent Model** for a cyber security threat detection system, as outlined in the "Agentic-AI driven Cybersecurity Threat Detection" research document. It uses a Deep Learning approach (CNN-LSTM) to detect network anomalies and cyber attacks.

## Project Structure

- `kaggle_dataset/`: Contains the CICIDS2017 network traffic dataset CSV files.
- `src/`
  - `data_preprocessing.py`: Handles loading the Kaggle dataset, cleaning missing values, standardizing features, and shaping data for the neural network.
  - `model.py`: Defines the CNN-LSTM deep learning model architecture using TensorFlow/Keras.
  - `train_evaluate.py`: Contains the logic for training the model (with early stopping) and evaluating performance metrics (Precision, Recall, F1-Score, PR-AUC).
- `artifacts/`: Directory where the training plots (`training_history.png`) and evaluation reports (`evaluation_report.txt`) are saved after running the model.
- `saved_models/`: Directory where the trained model weights (`best_model.keras`) are automatically saved during training.
- `main_pipeline.py`: The main orchestrator script. Running this file executes the entire pipeline from data loading to evaluation.

## Prerequisites

Ensure you have Python installed along with the following packages:
- `tensorflow`
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`

*(These are already installed in your current environment).*

## How to Run the Project

To train and evaluate the threat detection model, simply run the main pipeline script from your terminal or command prompt:

```bash
python main_pipeline.py
```

### What Happens When You Run It:
1. **Step 1 (Data Preprocessing)**: It loads a subset of the dataset (20,000 rows per file by default to ensure fast execution). It handles data cleaning and scales the features.
2. **Step 2 (Model Building)**: It compiles the CNN-LSTM model.
3. **Step 3 (Training)**: The model trains on the data for a maximum of 5 epochs (can be changed in `main_pipeline.py`). The best model is saved to `saved_models/best_model.keras`.
4. **Step 4 (Evaluation)**: The model is tested on unseen data. Metrics and a classification report are printed and saved.

## Checking the Results

After the pipeline finishes, navigate to the `artifacts/` folder:
- Open `evaluation_report.txt` to view the classification report (Precision, Recall, F1-scores) and PR-AUC scores for each attack type.
- Open `training_history.png` to view a graph of the model's loss and accuracy improving over time.
