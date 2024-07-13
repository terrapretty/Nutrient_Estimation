# run_plsr_with_bg_removal.py

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load spectral data with background removal
spectral_data_with_bg_removal = np.load('/path/to/output/spectral_data_with_bg_removal.npy')

# Load ground truth data
gt_data = pd.read_csv('/path/to/ground_truth.csv')

# Prepare data for PLSR
X = spectral_data_with_bg_removal
y = gt_data.values  # Adjust as needed to match your ground truth format

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform PLSR
pls = PLSRegression(n_components=10)  # Adjust the number of components as needed
pls.fit(X_train, y_train)

# Predict and evaluate
y_pred = pls.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R2 Score: {r2}')
