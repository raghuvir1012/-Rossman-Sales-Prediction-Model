# Rossman Sales Prediction Model

This project presents an end-to-end machine learning solution for predicting daily sales in Rossman drugstores. The work encompasses data cleaning, feature engineering, model development, rigorous evaluation, and deployment through an interactive Streamlit application for real-time sales forecasting.

## Project Overview

The Rossman Sales Prediction Model addresses a critical business need: forecasting daily sales volume to enable effective inventory management and staffing decisions. The model integrates multiple data sources and factors including store characteristics, promotional activities, competitive proximity, and temporal patterns. By combining these variables through advanced machine learning techniques, the model produces actionable sales predictions that support operational planning at the store level.

## Data Cleaning and Preparation

Data quality directly impacts model performance. Raw data frequently contains inconsistencies, missing values, and anomalies that would compromise model training. The cleaning process followed several critical steps to ensure data integrity.

First, records from closed stores (where Open equals 0) were excluded from training. Including these records would teach the model to associate store closure with sales patterns, introducing fundamental errors into predictions. Similarly, entries with zero or negative sales values were removed as these represent data entry errors or exceptional circumstances rather than normal operational conditions. The model's purpose is to predict sales under standard operations, so including these anomalies would systematically bias predictions downward.

Missing values in the competition distance variable were handled by imputing with the maximum observed distance. This conservative approach prevents the model from incorrectly inferring that missing distance data implies reduced competitive threat. After cleaning, the dataset contained approximately one million valid transactions representing typical store operations.

The prepared data was then enriched by merging store-specific characteristics including store type, assortment level, and competition information with daily transaction records. This integration provided the model with contextual information necessary for accurate predictions.

## Feature Engineering

Feature engineering transforms raw data into meaningful inputs that enable machine learning models to identify underlying patterns. Rather than relying solely on raw values, carefully constructed features guide the model toward discovering the true relationships between business conditions and sales outcomes.

Temporal features were created by decomposing the date variable into year, month, day, and week of year components. Different periods exhibit distinct sales patterns due to seasonal effects, holiday shopping, and promotional calendars. By explicitly encoding these temporal dimensions, the model gains direct access to seasonal patterns that would otherwise require significantly more training data to discover.

Competition features were engineered by calculating the duration between store opening dates and the current period. This metric captures how established competitive threats are, reflecting the principle that newer competitors exert stronger effects on sales than long-established alternatives. Similarly, promotional duration features measure how long current and recurring promotions have been active, as sales impacts of promotional campaigns typically diminish over time as customer response normalizes.

Promotion features specifically identify months that qualify for recurring promotional campaigns and calculate the duration of ongoing promotion periods. These engineered features allow the model to distinguish between initial promotional impact and sustained promotional effects across different seasonal contexts.

## Model Selection: XGBoost Regressor

XGBoost (Extreme Gradient Boosting) was selected as the primary algorithm following systematic evaluation of multiple approaches. XGBoost operates through an ensemble technique where multiple decision trees sequentially refine predictions. The first tree makes an initial estimate, subsequent trees identify and correct residual errors from previous iterations, and this process continues iteratively. The final prediction aggregates votes from all trees in the ensemble.

XGBoost demonstrates particular strength for this application due to several characteristics. The algorithm naturally handles heterogeneous input types, processing both continuous variables like competition distance and categorical variables like store type without requiring separate preprocessing pipelines. The tree-based architecture captures non-linear relationships between features and sales; for example, promotional effects differ substantially between holidays and regular weekdays, a non-linearity that linear models cannot capture. The algorithm exhibits robustness to outliers and unusual sales patterns that might arise from exceptional events, preventing isolated anomalies from dominating model training.

Alternative approaches were evaluated and rejected for specific reasons. Simple linear regression assumes direct proportional relationships between features and sales, insufficient for capturing the complex interactions present in retail operations. Random forest ensembles provide competitive baseline performance but require substantially more computational resources than XGBoost while often delivering inferior accuracy. Neural network approaches introduce additional complexity without commensurate performance benefits for tabular retail data and prove more difficult to interpret regarding which factors drive specific predictions.

## Model Configuration

The final model configuration uses 100 decision trees with a maximum depth of 5 levels per tree. The learning rate is set to 0.1, meaning each tree conservatively corrects previous errors rather than aggressively overwriting predictions. Subsampling is configured at 0.9, indicating each tree trains on 90 percent of the data, introducing beneficial variance across the ensemble. Column subsampling is set to 0.7, further promoting diversity by having each tree access 70 percent of available features.

This configuration reflects a deliberate balance between model complexity and generalization. Shallower trees prevent the model from memorizing training data specifics that would not generalize to future predictions. Conservative learning rates produce stable predictions that gradually incorporate patterns rather than making abrupt shifts. Subsampling in both observations and features reduces overfitting by ensuring no individual tree dominates the ensemble.

## Evaluation Methodology: RMSE

Root Mean Squared Error (RMSE) was selected as the primary evaluation metric. RMSE quantifies prediction accuracy by calculating the square root of the average squared differences between actual and predicted sales. The squaring operation emphasizes larger errors more heavily than smaller errors, reflecting the reality that significantly inaccurate predictions are substantially more costly than minor deviations for inventory planning purposes.

RMSE maintains interpretability by retaining the original measurement units. An RMSE value of 500 indicates the model is off by approximately 500 units on average, making the metric directly comparable to typical sales volumes. This interpretability advantages RMSE over normalized metrics when communicating model performance to business stakeholders.

Model performance was evaluated through a five-fold cross-validation procedure. The training dataset was divided into five subsets, and the model was trained five times, each time using four subsets for training and one subset for validation. This approach provides a robust estimate of model generalization to unseen data while utilizing all available training data efficiently.

## Performance Analysis

Cross-validation results demonstrated consistent model behavior across data subsets. Average training RMSE was approximately 600-800 units, while average validation RMSE was approximately 1,000-1,200 units. The modest gap between training and validation error indicates appropriate regularization without significant overfitting, though the model does not perfectly generalize all patterns from training data to novel situations.

Feature importance analysis revealed that promotional status ranks as the most influential predictor, indicating promotions substantially increase sales volume. Store type and assortment characteristics rank second, reflecting that certain store formats and product mixes consistently outperform others. Seasonal factors including month and week of year rank third, confirming clear seasonal patterns in purchasing behavior. Day of week effects are also substantial, confirming that weekly shopping patterns are reliable and substantial. Competitive proximity demonstrates measurable but smaller effects, indicating nearby competitors reduce sales but less dramatically than promotional or seasonal factors.

Error analysis identified specific conditions where prediction accuracy diminishes. Holiday periods and unexpected promotional changes generate larger prediction errors due to atypical shopping behavior that diverges from historical patterns. Stores with unusual or inconsistent sales characteristics are more difficult to predict than those with stable demand patterns. Conversely, the model produces most accurate predictions during regular weekdays, in established promotional contexts, and for stores with consistent historical performance patterns.

## Model Assessment and Practical Utility

The model achieves 80-85 percent accuracy in the sense that predictions fall within reasonable operational bounds for approximately four out of five instances. For inventory planning purposes, this accuracy level represents substantial improvement over traditional seasonal forecasting or naive statistical approaches. Store managers can confidently base safety stock calculations and replenishment scheduling on model predictions for routine operational periods.

The model performs well on regular business days and effectively captures weekly and seasonal patterns. Promotional effects are reliably predicted when promotions follow established patterns. Competition proximity is appropriately factored into predictions. However, the model cannot predict truly unexpected events such as pandemic-driven disruptions, major store renovations, or viral trends. During holiday periods when shopping behavior deviates substantially from typical patterns, the model may underestimate sales volume. New stores not represented in training data may receive less accurate predictions than established stores.

For practical implementation, the model should be used as a baseline forecast that incorporates historical patterns and established business relationships. Store managers should apply domain knowledge to adjust predictions for known special circumstances, major events, or anticipated changes in competitive environment. The model provides substantial value for routine operations while recognizing that business intuition and real-time knowledge remain essential for optimal decision-making.

## Usage and Deployment

The model is deployed through two primary interfaces. An interactive Streamlit application provides non-technical users with a web-based tool for inputting store characteristics and dates and receiving sales predictions immediately. For programmatic access and integration with existing systems, the complete machine learning workflow is documented in the rossman_ml.ipynb notebook file, which can be adapted for batch processing or real-time prediction pipelines.

## Project Files

The project includes several key components. The rossman_ml.ipynb notebook contains the complete machine learning pipeline from data loading through model evaluation. The rossman_model.pkl file contains the trained XGBoost model ready for prediction tasks. The scaler.pkl file contains the fitted feature scaling parameters, ensuring new data is normalized consistently with training data. The encoder.pkl file contains categorical encoding information, maintaining consistent numerical representation of categorical variables. The streamlit_app.py file provides the web-based user interface for making predictions without coding.

## Conclusion

The Rossman Sales Prediction Model successfully demonstrates that systematic application of data science techniques produces practically useful predictions for retail operations. By carefully cleaning data, engineering meaningful features, selecting an appropriate algorithm, and validating performance rigorously, the model achieves accuracy sufficient for inventory planning and operational decision-making. While the model cannot account for unprecedented events or radical shifts in consumer behavior, it reliably captures historical patterns and established business relationships, providing substantial value for routine sales forecasting in retail environments.
