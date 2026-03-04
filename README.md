COMP5530M-Group-Project-Inflation-Forecasting
=====================

## Overview:
This project investigates the effectiveness of classical statistical models, machine learning techniques, and deep learning architectures for forecasting inflation using macroeconomic time-series data. The study focuses on predicting U.S. inflation measured by the Personal Consumption Expenditure Price Index (PCEPI) and evaluates model performance across multiple forecast horizons (1, 3, 6, and 12 months).

The project implements and compares over fifteen forecasting approaches, ranging from traditional econometric models such as ARDL and SARIMAX to modern deep learning architectures including LSTM, Temporal Fusion Transformers (TFT), and Neural Hierarchical Interpolation for Time Series (N-HiTS).

Our evaluation demonstrates that N-HiTS consistently achieved the highest predictive accuracy across all horizons, highlighting the potential of hierarchical neural forecasting architectures for modelling complex macroeconomic time series.

This repository contains the full implementation of the models, the experimental workflow used to evaluate them, and the final deliverables produced for the COMP5530M group project at the University of Leeds.


## Project Deliverables:

Group Report  
Inflation_Forecasting_with_Machine_Learning_Models__Group_Report.pdf 
![Alt text](https://github.com/Sandraag/Group-Project-Inflation-Forecasting/blob/main/Inflation_Forecasting_with_Machine_Learning_Models__Group_Report.pdf)

Project Poster  
GroupProjectPoster.pdf 
![Alt text]https://github.com/Sandraag/Group-Project-Inflation-Forecasting/blob/main/5761489c-1.png)

## Key Features:

- Multi-Model Forecasting Framework: Implementation of classical econometric models, machine learning algorithms, and deep learning architectures for time-series forecasting.

- Macroeconomic Data Integration: Inflation forecasting using the Personal Consumption Expenditure Price Index (PCEPI) combined with multiple exogenous macroeconomic indicators.

- Feature Engineering and Statistical Preprocessing: Use of Granger causality testing, cross-correlation analysis, and cointegration tests to select relevant economic predictors.

- Walk-Forward Forecasting Evaluation: All models are evaluated using a walk-forward validation strategy across multiple forecasting horizons.

- Comprehensive Model Comparison: Performance comparison using MAE, RMSE, R², and statistical hypothesis testing.

- Hierarchical Neural Forecasting: Evaluation of advanced neural time-series models such as N-BEATSx and N-HiTS.


## Tasks Breakdown

### 1. Data Collection and Preparation:
Macroeconomic datasets were gathered from multiple publicly available sources including FRED.

Preprocessing included:
- Aggregating data with different frequencies (daily, weekly, monthly)
- Handling missing values via interpolation
- Stationarity testing and differencing
- Feature engineering including lagged values, rolling statistics, and seasonal encodings


### 2. Exploratory Data Analysis:
Exploratory analysis was conducted to understand relationships between inflation and economic indicators.

Techniques included:
- Autocorrelation and partial autocorrelation analysis (ACF / PACF)
- Cross-correlation analysis
- ANOVA statistical testing
- Cointegration analysis using Johansen tests


### 3. Model Development:

Models were implemented across three major categories.

Classical Statistical Models:
- ARDL
- VAR
- ARIMA / SARIMA
- ARIMAX / SARIMAX

Machine Learning Models:
- Random Forest
- XGBoost
- MARS

Deep Learning Models:
- RNN
- GRU
- LSTM
- Temporal Fusion Transformer (TFT)
- Temporal Convolutional Network (TCN)
- TiDE
- N-BEATSx
- N-HiTS


### 4. Hyperparameter Optimisation:
Hyperparameters were optimised using the Optuna framework across multiple trials to identify optimal configurations for each model and forecasting horizon.


### 5. Forecast Evaluation:
Model performance was evaluated using the following metrics:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)

Statistical comparison against a naive forecasting baseline was conducted using the Diebold-Mariano test to determine whether improvements were statistically significant.


### 6. Results:
Across all forecast horizons, N-HiTS consistently outperformed competing models, demonstrating strong predictive accuracy and robustness when modelling complex macroeconomic time-series behaviour.

The hierarchical architecture allowed the model to capture multi-scale temporal patterns while effectively integrating exogenous economic indicators.


## Repository Structure:

training/
    ModelName/
        weights/
        training scripts

predictions/
    ModelName.npy

notebooks/
    exploratory analysis
    model experimentation

data/
    processed datasets

reports/
    group_report.pdf
    poster.pdf


## Execution Instructions:

Ensure your current working directory matches the directory of this repository.

(optional) Create a virtual environment:

python -m venv .venv


Activate the environment:

Windows
.venv\Scripts\activate

Linux / Mac
source .venv/bin/activate


Install dependencies:

pip install -r requirements.txt


Browse the notebooks and run experiments using the appropriate Python kernel.


## Technologies Used:

- Python
- PyTorch
- Scikit-learn
- Statsmodels
- XGBoost
- NeuralForecast
- Darts Time-Series Library
- Optuna
- NumPy
- Pandas
- Matplotlib


## Contributors:

George Bignall  
Muhammed Murat Kurmaz  
Ayesha Rahman  
James Zhangly  
Kevin Raffaelli  
Sandra Guran  
Natalie Leung  


Supervisor:

Dr Timon S. Gutleb


## Disclaimer:

This repository was created for academic purposes as part of the COMP5530M course at the University of Leeds. The models and predictions produced by this project are for research purposes only and should not be used for financial or investment decision-making.

## Individual Report – Sandra Guran
![Alt text](https://github.com/Sandraag/Group-Project-Inflation-Forecasting/blob/main/Individual_Report_SandraGuran.pdf)
My individual contribution to this project focused on the design, implementation, and optimisation of deep learning and machine learning models for inflation forecasting.

I was the primary contributor responsible for developing and evaluating three forecasting models:

- Recurrent Neural Network (RNN)
- Gated Recurrent Unit (GRU)
- XGBoost Regressor

These models were implemented to explore sequence-based approaches for modelling macroeconomic time series and to compare their forecasting performance against other statistical and machine learning methods used within the project.

### Key Contributions

- Designed and implemented RNN and GRU models for multi-horizon inflation forecasting.
- Developed and maintained the **Training/RNN**, **Training/GRU**, and **Training/XGBoost** modules within the project repository.
- Implemented feature engineering techniques including:
  - Lagged inflation features
  - Rolling statistics (mean and standard deviation)
  - Seasonal encodings using sine and cosine transformations.
- Integrated models into the shared project pipeline using **PyTorch Forecasting**.
- Conducted extensive **hyperparameter optimisation using Optuna**, running up to 200 optimisation trials per model and forecasting horizon.
- Implemented multi-horizon forecasting for prediction windows of **1, 3, 6, and 12 months**.
- Generated forecast outputs in `.npy` format for integration with the team’s central evaluation framework.
- Authored the **methodology sections for the RNN and GRU models in the group report**.

### Model Implementation

The RNN model served as a baseline sequential model implemented using PyTorch with batch-first sequence inputs.  
The GRU model extended this architecture by introducing gating mechanisms to improve memory retention and mitigate vanishing gradient issues.

Both models were implemented with the following configurable parameters:

- `input_size` – number of input features
- `hidden_size` – tuned between 32 and 512
- `num_layers` – 1 to 6 stacked recurrent layers
- `output_size` – corresponding to forecast horizons

Training configuration included:

- Adam optimiser
- Mean Squared Error loss
- Batch size of 32
- Input sequence length of 12 months
- 80/20 training-validation split

### Model Evaluation

The RNN and GRU models were evaluated across rolling forecast horizons of:

- 1 month
- 3 months
- 6 months
- 12 months

Results showed that:

- RNN slightly outperformed GRU at short horizons (1 month)
- GRU achieved better stability at medium horizons due to its gating mechanisms
- Both models showed declining performance at longer horizons due to error accumulation and vanishing gradient effects.

Despite careful optimisation, the recurrent models were ultimately outperformed by more advanced architectures such as **N-HiTS**, which provided more stable long-term forecasting.

### Reflection

This project significantly strengthened my understanding of deep learning for time-series forecasting. Implementing RNN-based architectures from scratch and integrating them into a collaborative research pipeline improved my skills in:

- PyTorch model development
- Hyperparameter optimisation
- Time-series feature engineering
- Experiment reproducibility
- Collaborative machine learning development

Working as part of a team also improved my ability to communicate technical ideas and contribute to shared project objectives.
