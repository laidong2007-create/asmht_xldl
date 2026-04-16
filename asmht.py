# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import matplotlib.pyplot as plt
import seaborn as sns


# 2. LOAD DATA
df = pd.read_csv("house_data.csv")

# 3. BASIC CLEANING
# Drop duplicates
df = df.drop_duplicates()

# 4. FEATURE ENGINEERING
# ==============================

# ---- DATE FEATURE ----
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter

# ---- TEXT FEATURE ----
if 'description' in df.columns:
    df['description'] = df['description'].fillna('')
    df['desc_length'] = df['description'].apply(len)
    df['has_luxury'] = df['description'].str.contains('luxury', case=False).astype(int)
else:
    df['desc_length'] = 0
    df['has_luxury'] = 0

# ---- INTERACTION FEATURE ----
if 'area' in df.columns and 'num_rooms' in df.columns:
    df['area_room'] = df['area'] * df['num_rooms']

# ---- KPI FEATURES ----
if 'price' in df.columns and 'area' in df.columns:
    df['price_per_m2'] = df['price'] / df['area']

# ==============================
# 5. HANDLE SKEWNESS
# ==============================
num_cols_all = df.select_dtypes(include=np.number).columns

skewness = df[num_cols_all].skew()
skewed_cols = skewness[skewness > 1].index

pt = PowerTransformer(method='yeo-johnson')

for col in skewed_cols:
    if col != 'price':  # không transform target
        df[col] = pt.fit_transform(df[[col]])

# ==============================
# 6. REMOVE OUTLIER (OPTIONAL)
# ==============================
if 'price' in df.columns:
    df = df[df['price'] < df['price'].quantile(0.99)]

# ==============================
# 7. SPLIT DATA
# ==============================
X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 8. PIPELINE PREPROCESSING
# ==============================
num_cols = X.select_dtypes(include=np.number).columns
cat_cols = X.select_dtypes(include='object').columns

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

# ==============================
# 9. MODELS
# ==============================

# Linear Regression
model_lr = Pipeline([
    ('preprocess', preprocessor),
    ('model', LinearRegression())
])

# Random Forest
model_rf = Pipeline([
    ('preprocess', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

# ==============================
# 10. TRAIN
# ==============================
model_lr.fit(X_train, y_train)
model_rf.fit(X_train, y_train)

# ==============================
# 11. EVALUATION FUNCTION
# ==============================
def evaluate(model, name):
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n{name}")
    print("RMSE:", rmse)
    print("MAE :", mae)
    print("R2  :", r2)

# ==============================
# 12. EVALUATE
# ==============================
evaluate(model_lr, "Linear Regression")
evaluate(model_rf, "Random Forest")

# ==============================
# 13. CROSS VALIDATION
# ==============================
cv_scores = cross_val_score(
    model_rf, X, y,
    cv=5,
    scoring='neg_mean_squared_error'
)

print("\nCV RMSE:", np.mean(np.sqrt(-cv_scores)))

# ==============================
# 14. VISUALIZATION
# ==============================

# Price distribution
plt.figure()
sns.histplot(df['price'], kde=True)
plt.title("Price Distribution")
plt.show()

# Price per m2
if 'price_per_m2' in df.columns:
    plt.figure()
    sns.histplot(df['price_per_m2'], kde=True)
    plt.title("Price per m2 Distribution")
    plt.show()

# ==============================
# 15. INSIGHT EXAMPLE
# ==============================
if 'price_per_m2' in df.columns:
    print("\nTop 5 expensive areas (price/m2):")
    print(df.sort_values('price_per_m2', ascending=False).head())

# ==============================
# DONE
# ==============================