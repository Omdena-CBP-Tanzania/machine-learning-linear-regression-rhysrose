{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "686f8dbf-8da3-4ac9-8a63-d5a700ac1740",
   "metadata": {},
   "source": [
    "# Task 4: Model Evaluation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "0486ec0f-4da9-4d68-a2b6-7c43b212da3b",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.metrics import mean_squared_error, r2_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "e98ff2e8-320d-4aad-80e4-f2d40d3951a1",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Import Model and Test data\n",
    "import joblib\n",
    "modelpath= \"/Users/rosepeterfunja/Tanzania_KIC/Assignment/machine-learning-linear-regression-rhysrose/models/linear_regression_model.pkl\"\n",
    "\n",
    "model = joblib.load(modelpath)\n",
    "\n",
    "X_data_path = r\"/Users/rosepeterfunja/Tanzania_KIC/Assignment/machine-learning-linear-regression-rhysrose/data/X_test.csv\"\n",
    "y_data_path = r\"/Users/rosepeterfunja/Tanzania_KIC/Assignment/machine-learning-linear-regression-rhysrose/data/y_test.csv\"\n",
    "\n",
    "X_test_df = pd.read_csv(X_data_path)\n",
    "y_test_df = pd.read_csv(y_data_path)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "4d30d1af-e26f-428b-a364-5a9168123a15",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mean Squared Error: 0.5081\n",
      "R² Score: 0.5002\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Applications/anaconda3/lib/python3.7/site-packages/sklearn/base.py:444: UserWarning: X has feature names, but LinearRegression was fitted without feature names\n",
      "  f\"X has feature names, but {self.__class__.__name__} was fitted without\"\n"
     ]
    }
   ],
   "source": [
    "# Predict\n",
    "y_pred = model.predict(X_test_df)\n",
    "\n",
    "# Metrics\n",
    "mse = mean_squared_error(y_test_df, y_pred)\n",
    "r2 = r2_score(y_test_df, y_pred)\n",
    "\n",
    "print(f\"Mean Squared Error: {mse:.4f}\")\n",
    "print(f\"R² Score: {r2:.4f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "89f33913-e9ad-4f06-9e67-4adc0e256027",
   "metadata": {},
   "source": [
    "Moderate Model: The model is moderately effective, with the ability to explain about half of the variability in the target variable, but there's clearly room for improvement\n",
    "0.5 R² and MSE of 0.5081 indicate a model that's somewhat accurate but could be improved."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2920aa5e-07f8-42f0-8ca6-9dfb9aa471dd",
   "metadata": {},
   "source": [
    "## Plotting residuals "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "cf348545-90cf-4cdf-b7ea-89bc1f421c69",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 576x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Residual plot\n",
    "\n",
    "# Convert y_test_df to a 1D array/series for the subtraction\n",
    "y_test_series = y_test_df.values.flatten()  # Convert y_test_df to a 1D array\n",
    "\n",
    "# Calculate residuals (errors between predicted and actual values)\n",
    "residuals = y_test_series - y_pred\n",
    "\n",
    "# Plot the residuals\n",
    "plt.figure(figsize=(8, 5))\n",
    "plt.scatter(y_pred, residuals)\n",
    "plt.axhline(y=0, color='r', linestyle='--')\n",
    "plt.xlabel(\"Predicted values\")\n",
    "plt.ylabel(\"Residuals\")\n",
    "plt.title(\"Residuals Plot\")\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a4d52b87-5284-464a-9968-b84adb5729ba",
   "metadata": {},
   "source": [
    "Ideal residual plot: A cloud of points around the horizontal line at zero, with no discernible pattern."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "010c94f2-99af-4987-871c-762c3a857174",
   "metadata": {},
   "source": [
    "# Comparing model performance with different feature sets or preprocessing steps"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "eb3e04bf-cfc5-4c64-a39a-5dcf0224b576",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                                MSE        R2\n",
      "No Preprocessing          24.291119  0.668759\n",
      "With Scaling              24.291119  0.668759\n",
      "With Imputation           24.291119  0.668759\n",
      "With Polynomial Features  14.183558  0.806589\n"
     ]
    }
   ],
   "source": [
    "from sklearn.model_selection import train_test_split, cross_val_score\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.preprocessing import StandardScaler, PolynomialFeatures\n",
    "from sklearn.impute import SimpleImputer\n",
    "from sklearn.metrics import mean_squared_error, r2_score\n",
    "\n",
    "#load data\n",
    "# Load dataset\n",
    "filepath = r\"/Users/rosepeterfunja/Tanzania_KIC/Assignment/machine-learning-linear-regression-rhysrose/BostonHousing.csv\"\n",
    "df = pd.read_csv(filepath)\n",
    "\n",
    "\n",
    "# Split the data\n",
    "X = df.drop(columns=[\"medv\"])\n",
    "y = df[\"medv\"]\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "# 1. Model with no preprocessing\n",
    "model = LinearRegression()\n",
    "model.fit(X_train, y_train)\n",
    "y_pred = model.predict(X_test)\n",
    "mse_no_preprocessing = mean_squared_error(y_test, y_pred)\n",
    "r2_no_preprocessing = r2_score(y_test, y_pred)\n",
    "\n",
    "# 2. Model with scaling (Standardization)\n",
    "scaler = StandardScaler()\n",
    "X_train_scaled = scaler.fit_transform(X_train)\n",
    "X_test_scaled = scaler.transform(X_test)\n",
    "\n",
    "model.fit(X_train_scaled, y_train)\n",
    "y_pred_scaled = model.predict(X_test_scaled)\n",
    "mse_scaled = mean_squared_error(y_test, y_pred_scaled)\n",
    "r2_scaled = r2_score(y_test, y_pred_scaled)\n",
    "\n",
    "# 3. Model with missing value imputation\n",
    "imputer = SimpleImputer(strategy='mean')  # Mean imputation for missing values\n",
    "X_train_imputed = imputer.fit_transform(X_train)\n",
    "X_test_imputed = imputer.transform(X_test)\n",
    "\n",
    "model.fit(X_train_imputed, y_train)\n",
    "y_pred_imputed = model.predict(X_test_imputed)\n",
    "mse_imputed = mean_squared_error(y_test, y_pred_imputed)\n",
    "r2_imputed = r2_score(y_test, y_pred_imputed)\n",
    "\n",
    "# 4. Model with polynomial features (degree 2)\n",
    "poly = PolynomialFeatures(degree=2)\n",
    "X_train_poly = poly.fit_transform(X_train)\n",
    "X_test_poly = poly.transform(X_test)\n",
    "\n",
    "model.fit(X_train_poly, y_train)\n",
    "y_pred_poly = model.predict(X_test_poly)\n",
    "mse_poly = mean_squared_error(y_test, y_pred_poly)\n",
    "r2_poly = r2_score(y_test, y_pred_poly)\n",
    "\n",
    "# Compare results\n",
    "results = {\n",
    "    \"No Preprocessing\": {\"MSE\": mse_no_preprocessing, \"R2\": r2_no_preprocessing},\n",
    "    \"With Scaling\": {\"MSE\": mse_scaled, \"R2\": r2_scaled},\n",
    "    \"With Imputation\": {\"MSE\": mse_imputed, \"R2\": r2_imputed},\n",
    "    \"With Polynomial Features\": {\"MSE\": mse_poly, \"R2\": r2_poly}\n",
    "}\n",
    "\n",
    "# Display results\n",
    "\n",
    "results_df = pd.DataFrame(results).T\n",
    "print(results_df)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "58ca9c0a-c8c9-425e-b7d3-fb9b76f17eb6",
   "metadata": {},
   "source": [
    "Key Takeaways:\n",
    "Polynomial Features significantly improved the model performance, both reducing MSE and increasing R².\n",
    "\n",
    "Scaling and Imputation didn’t show any impact on the model, suggesting that either the data didn’t require scaling or imputation, or other more important transformations (like polynomial feature engineering) should have been considered."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "00156510-f490-4f04-9646-bdad9fe01940",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:anaconda3]",
   "language": "python",
   "name": "conda-env-anaconda3-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
