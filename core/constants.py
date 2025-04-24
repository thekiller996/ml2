"""
Constant definitions for the ML Platform.
"""

# Application pages
PAGES = {
    "Project Setup": "project_setup",
    "Data Upload": "data_upload",
    "Data Preprocessing": "data_preprocessing",
    "Exploratory Analysis": "exploratory_analysis",
    "Feature Engineering": "feature_engineering",
    "Model Training": "model_training",
    "Model Evaluation": "model_evaluation",
    "Prediction": "prediction"
}

# Data types
DATA_TYPES = {
    "Numerical": "numerical",
    "Categorical": "categorical",
    "DateTime": "datetime",
    "Text": "text",
    "Image": "image",
    "Audio": "audio",
    "Video": "video",
    "Geospatial": "geospatial"
}

# ML problem types
MODEL_TYPES = {
    "Classification": "classification",
    "Regression": "regression",
    "Clustering": "clustering",
    "Dimensionality Reduction": "dimensionality_reduction",
    "Time Series": "time_series"
}

# Preprocessing methods
PREPROCESSING_METHODS = {
    "Missing Values": {
        "Drop": "drop",
        "Mean": "mean",
        "Median": "median",
        "Mode": "mode",
        "Constant": "constant",
        "KNN": "knn",
        "Linear Regression": "linear_regression"
    },
    "Scaling": {
        "StandardScaler": "standard",
        "MinMaxScaler": "minmax",
        "RobustScaler": "robust",
        "Normalizer": "normalizer",
        "MaxAbsScaler": "maxabs"
    },
    "Encoding": {
        "OneHotEncoder": "onehot",
        "LabelEncoder": "label",
        "OrdinalEncoder": "ordinal",
        "TargetEncoder": "target",
        "BinaryEncoder": "binary",
        "HashingEncoder": "hashing"
    },
    "Outliers": {
        "Z-Score": "zscore",
        "IQR": "iqr",
        "Isolation Forest": "isolation_forest",
        "Local Outlier Factor": "local_outlier_factor",
        "One-Class SVM": "one_class_svm"
    }
}

# Classification models
CLASSIFICATION_MODELS = {
    "Logistic Regression": "logistic_regression",
    "Random Forest": "random_forest",
    "Gradient Boosting": "gradient_boosting",
    "Support Vector Machine": "svm",
    "K-Nearest Neighbors": "knn",
    "Decision Tree": "decision_tree",
    "Naive Bayes": "naive_bayes",
    "XGBoost": "xgboost",
    "LightGBM": "lightgbm",
    "Neural Network": "neural_network"
}

# Regression models
REGRESSION_MODELS = {
    "Linear Regression": "linear_regression",
    "Ridge Regression": "ridge",
    "Lasso Regression": "lasso",
    "ElasticNet": "elasticnet",
    "Random Forest": "random_forest",
    "Gradient Boosting": "gradient_boosting",
    "Support Vector Regression": "svr",
    "XGBoost": "xgboost",
    "LightGBM": "lightgbm",
    "Neural Network": "neural_network"
}

# Clustering models
CLUSTERING_MODELS = {
    "K-Means": "kmeans",
    "DBSCAN": "dbscan",
    "Hierarchical": "hierarchical",
    "Gaussian Mixture": "gaussian_mixture",
    "Agglomerative": "agglomerative",
    "BIRCH": "birch",
    "Mean Shift": "mean_shift"
}

# Evaluation metrics
CLASSIFICATION_METRICS = {
    "Accuracy": "accuracy",
    "Precision": "precision",
    "Recall": "recall",
    "F1 Score": "f1",
    "ROC AUC": "roc_auc",
    "Confusion Matrix": "confusion_matrix",
    "Precision-Recall Curve": "precision_recall",
    "Log Loss": "log_loss"
}

REGRESSION_METRICS = {
    "R2 Score": "r2",
    "Mean Absolute Error": "mae",
    "Mean Squared Error": "mse",
    "Root Mean Squared Error": "rmse",
    "Mean Absolute Percentage Error": "mape",
    "Median Absolute Error": "median_absolute_error",
    "Explained Variance": "explained_variance"
}

CLUSTERING_METRICS = {
    "Silhouette Score": "silhouette_score",
    "Calinski-Harabasz Index": "calinski_harabasz",
    "Davies-Bouldin Index": "davies_bouldin",
    "Inertia": "inertia"
}

# Feature selection methods
FEATURE_SELECTION_METHODS = {
    "Variance Threshold": "variance_threshold",
    "Select K Best": "select_k_best",
    "Select Percentile": "select_percentile",
    "Recursive Feature Elimination": "rfe",
    "Sequential Feature Selection": "sequential",
    "Feature Importance": "feature_importance",
    "Correlation": "correlation",
    "Mutual Information": "mutual_info"
}

# Hyperparameter tuning methods
TUNING_METHODS = {
    "Grid Search": "grid_search",
    "Random Search": "random_search",
    "Bayesian Optimization": "bayesian",
    "Optuna": "optuna",
    "Genetic Algorithm": "genetic",
    "HyperOpt": "hyperopt"
}

# Chart types
CHART_TYPES = {
    "Bar Chart": "bar",
    "Line Chart": "line",
    "Scatter Plot": "scatter",
    "Histogram": "histogram",
    "Box Plot": "box",
    "Violin Plot": "violin",
    "Heatmap": "heatmap",
    "Correlation Matrix": "correlation",
    "Pair Plot": "pair",
    "3D Scatter": "scatter_3d",
    "Contour Plot": "contour",
    "Surface Plot": "surface"
}