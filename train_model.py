import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

#Load your dataset
df = pd.read_csv(r"C:\Users\ayush\OneDrive\Desktop\cardekho_dataset.csv") 

#Define features (X) and target (y)
X = df.drop(columns=['selling_price', 'car_name'])  # Drop target + irrelevant
y = df['selling_price']

#Define preprocessing
categorical_cols = ['brand', 'model', 'seller_type', 'fuel_type', 'transmission_type']
numerical_cols = ['vehicle_age', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']

#Transformers
num_transformer = StandardScaler()
cat_transformer = OneHotEncoder(handle_unknown='ignore')

#ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, numerical_cols),
        ('cat', cat_transformer, categorical_cols)
    ],
    remainder='drop'
)

#Create pipeline with model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, max_depth=15, max_features=10))
])

#Split data (optional but good practice)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Fit the pipeline
model_pipeline.fit(X_train, y_train)

#Save the complete pipeline
joblib.dump(model_pipeline, 'gradient_model.pkl')  #Save full model

#Save preprocessor separately if you ever need
joblib.dump(preprocessor, 'preprocessor.pkl')

print("Model pipeline and preprocessor saved successfully!")
