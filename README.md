# Student Marks Prediction

A machine learning project that predicts student final exam marks based on their academic performance and personal background factors. Built with Python, Scikit-learn, and Streamlit.

## Overview

This project implements a regression-based prediction system that analyzes student data to forecast final grades. The model is trained on historical student performance data and provides an interactive web interface for making predictions.

## Features

- **Machine Learning Models**: Linear Regression, Ridge Regression, and Lasso Regression
- **Interactive Web Interface**: Built with Streamlit for easy data input and visualization
- **Performance Metrics**: MAE, MSE, RMSE, and R² Score for model evaluation
- **Data Visualization**: Plotly-based charts for result analysis

## Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Programming language |
| Scikit-learn | Machine learning algorithms |
| Pandas | Data manipulation |
| NumPy | Numerical computing |
| Matplotlib | Data visualization |
| Streamlit | Web application framework |
| Plotly | Interactive charts |

## Dataset

The model uses the following key features for prediction:

- **studytime**: Weekly study time (1-4 scale)
- **failures**: Number of past class failures
- **G1**: First period grade (0-20)
- **G2**: Second period grade (0-20)

**Target Variable**: G3 - Final grade (0-20)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Student-Marks-Prediction-ML-main
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Analysis Script

To train and evaluate the models:
```bash
python analysis.py
```

This will:
- Load and preprocess the student data
- Train Linear, Ridge, and Lasso regression models
- Display performance metrics
- Save the trained model as `model.pkl`

### Running the Web Application

To launch the Streamlit web interface:
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

## Model Performance

The models are evaluated using the following metrics:

- **Mean Absolute Error (MAE)**: Average absolute difference between predicted and actual values
- **Mean Squared Error (MSE)**: Average squared difference between predicted and actual values
- **Root Mean Squared Error (RMSE)**: Square root of MSE
- **R² Score**: Proportion of variance explained by the model (0-1, higher is better)

## Project Structure

```
Student-Marks-Prediction-ML-main/
├── analysis.py          # Model training and evaluation
├── app.py               # Streamlit web application
├── student_data.csv     # Dataset
├── requirements.txt     # Project dependencies
├── README.md            # Project documentation
└── LICENSE              # License file
```

## Author

**Tarundeep Singh**

## License

This project is licensed under the terms included in the LICENSE file.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.




