### House-Price-Prediction-Project-and-APP

### Project Overview
Predicting U.S. house prices is a classic ML problem with real-world impact. In this project, I build and deploy an XGBoost regression model (split into low/high-price segments) via a Streamlit app for realtime predictions and data exploration.

Built a Streamlit web application that enables users to:

Input features like state, city, number of bedrooms, number of bathrooms, and size of house in square footage.
Receive an instant house price prediction.
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
- Hyperparameter tuning via Grid Search with 3-fold Cross-Validation to optimize performance.
- Made two models for different house price ranges and compared performance with other machine learning model based on MAPE(This was one of the most important metrics since prices range from 50000 to around 1000000), R², RMSE, and MAE scores.
- Built streamlit applications that complements the model. 

### 📏 Why MAPE?  
- Houses range from \$50 K to \$1 M, so absolute errors aren’t directly comparable.  
- MAPE gives a **relative** error, e.g. “±15%,” which stakeholders find more intuitive.

### 🔍 Why Two Models?  
- A single model yielded **~37% MAPE** on homes <\$300 K vs **~15% MAPE** on homes ≥\$300 K.  
- Splitting the dataset into “low-price” and “high-price” segments reduced MAPE in the low segment to **24%**.

### The best-performing model was: XGBoost Regressor.
MAPE by segment: 
<300k        37.254866
300k-400k    16.742555
400k-500k    15.401889
500k-1M      15.486529
R² Score: 0.76.
MAE: 73231.480444976. 
RMSE: 106592.24842821828

Performance of model with houses below $300000:
MAPE: 24%.
MAE: 36482.
By making another model for houses below $300000, I was able to decrease the mape from 37% to 24% for houses in that range. 

Author Tomas Rodriguez Marengo | MS in Business Analytics | Data Science & Finance Enthusiast 
📫 LinkedIn https://www.linkedin.com/in/tomas-rodriguez-marengo/
🌎 Based in the U.S. & Argentina
