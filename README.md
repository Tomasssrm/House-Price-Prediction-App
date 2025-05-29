### House-Price-Prediction-Project-and-APP

### Project Overview
Predicting U.S. house prices is a classic ML problem with real-world impact. In this project, I build and deploy a LightGBM regression model (split into low/high-price segments) via a Streamlit app for realtime predictions and data exploration.

Built a Streamlit web application that enables users to:

Input features like state, city, number of bedrooms, number of bathrooms, and size of house in square footage.
Receive an instant house price prediction with down payment estimates and a visual with house price distribution in the state selected.

Navigate to a second page featuring interactive data visualizations.

### Modeling Approach
Tested and compared multiple regression models that perform well with numeric and categorical data. I made two different models for predictions since the first one had bigger errors for houses priced below $300000. 
The models that I tested were:

- XGBoost.
- Multilayer Perceptron (MLP Neural Network). 
- LightGBM.

### Key steps included:
- Extensive data cleaning and imputation for missing values.
- Feature engineering where appropriate.
- Hyperparameter tuning via Grid Search and Optuna with 3-fold Cross-Validation to optimize performance.
- Made two models for different house price ranges and compared performance with other machine learning models based on MAPE(This was one of the most important metrics since prices range from 50000 to around 1000000), R², RMSE, and MAE scores.
- Built streamlit applications that complements the model. 

### 📏 Why MAPE?  
- Houses range from \$50 K to \$1 M, so absolute errors aren’t directly comparable.  
- MAPE gives a **relative** error, e.g. “±15%,” which stakeholders find more intuitive for different price ranges.

### 🔍 Why Two Models?  
- A single model yielded **~35% MAPE** on homes <\$300 K vs **~15% MAPE** on homes ≥\$300 K.  
- Splitting the dataset into “low-price” and “high-price” segments reduced MAPE in the low segment to **24%**.

### The best-performing model was: LightGBM Regressor.
Overall performance metrics:
MAE: 73668.1104
RMSE: 110551.1297
MAPE: 0.2403
R2 Score: 0.7400
MAPE by segment: 
segment
<300k        35.216622
300k-400k    15.646312
400k-500k    15.309972
500k-1M      16.850587

Performance of model with houses below $300000:
MAPE: 24%.
MAE: 36482.
By making another model for houses below $300000, I was able to decrease the mape from 35% to 24% for houses in that range. 

Author Tomas Rodriguez Marengo | MS in Business Analytics | Data Science & Finance Enthusiast 
📫 LinkedIn https://www.linkedin.com/in/tomas-rodriguez-marengo/
🌎 Based in the U.S. & Argentina
