# Project 1: Cardiovascular Disease Prediction

Heart disease (cardiovascular disease) is one of the most serious illnesses worldwide.  
This project builds a complete machine-learning pipeline to predict the presence of cardiovascular disease using the standard `cardio_train` dataset.

## Dataset

- **File**: `data/cardio_train.csv`
- **Source**: Public cardiovascular disease dataset (70,000 patient records)
- **Target**: `cardio` (0 = no disease, 1 = disease)
- **Features**: age, gender, height, weight, systolic/diastolic blood pressure, cholesterol, glucose, smoke, alcohol, physical activity + derived BMI

## What this project does

1. **Data Preprocessing**
   - Convert age from days → years
   - Create BMI feature
   - Remove unrealistic outliers (blood pressure, height, weight, BMI)

2. **Exploratory Data Analysis & Visualizations**
   - Target distribution
   - Age, gender, cholesterol, glucose, blood pressure, BMI, lifestyle factors
   - Pairplot of key numeric features

3. **Correlation Matrix** of all features

4. **Model Comparison** (accuracy + ROC-AUC)
   - Support Vector Machine (SVM)
   - K-Nearest Neighbors (KNN)
   - Decision Tree (DT)
   - Logistic Regression (LR)
   - Random Forest (RF)

5. **Final Model**
   - Best model is saved as `models/best_model.joblib`
   - Scaler and feature list also saved for future predictions

## Project Structure

```
Cardiovascular_Disease_Prediction/
├── data/
│   └── cardio_train.csv
├── plots/                  # All generated visualizations
├── models/                 # Saved best model + scaler
├── cardio_disease_prediction.py   # Main script
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.8+
- pandas, numpy, scikit-learn, matplotlib, seaborn, joblib

## How to Run

### 1. Install dependencies

```bash
cd Cardiovascular_Disease_Prediction
pip install -r requirements.txt
```

### 2. Run the full analysis

```bash
python cardio_disease_prediction.py
```

This will:
- Clean the data
- Generate all plots (saved in `plots/`)
- Train & compare the 5 models
- Save the best model in `models/`

### 3. (Optional) Run in Jupyter / Colab / VS Code

You can also open a notebook and copy the cells from the script, or run:

```bash
jupyter notebook
```

## Platform Recommendations

| Platform              | Command / Notes                                      |
|-----------------------|------------------------------------------------------|
| **Local (Windows/Mac/Linux)** | `python cardio_disease_prediction.py`              |
| **Google Colab**      | Upload the zip → extract → `!pip install -r requirements.txt` → run the script |
| **Kaggle Notebook**   | Upload dataset + script, or clone the folder         |
| **VS Code / PyCharm** | Open folder and run the Python file                  |
| **Anaconda**          | Create environment then run the same command         |

## Expected Output

- Console prints accuracy of every model
- 13 plots saved in `plots/` folder
- Best model (usually Random Forest or Logistic Regression) saved in `models/`

## License

Educational / academic use.
