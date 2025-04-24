"""
Constants used throughout the ML Platform.
"""

# Navigation constants
PAGES = [
    "Project Setup",
    "Data Upload",
    "Exploratory Analysis",
    "Data Preprocessing",
    "Feature Engineering",
    "Model Training",
    "Model Evaluation",
    "Prediction"
]

# ML Task types
ML_TASKS = [
    "Classification", 
    "Regression", 
    "Clustering", 
    "Dimensionality Reduction",
    "Time Series",
    "Image Classification",
    "Object Detection",
    "Text Classification"
]

# Classification metrics
CLASSIFICATION_METRICS = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC AUC",
    "Log Loss"
]

# Regression metrics
REGRESSION_METRICS = [
    "Mean Absolute Error",
    "Mean Squared Error",
    "Root Mean Squared Error",
    "R² Score",
    "Mean Absolute Percentage Error"
]

# Clustering metrics
CLUSTERING_METRICS = [
    "Silhouette Score",
    "Davies-Bouldin Index",
    "Calinski-Harabasz Index"
]

# Preprocessing methods
ENCODING_METHODS = [
    "One-Hot Encoding",
    "Label Encoding",
    "Ordinal Encoding",
    "Target Encoding",
    "Binary Encoding",
    "Frequency Encoding"
]

SCALING_METHODS = [
    "Standard Scaler",
    "Min-Max Scaler",
    "Robust Scaler",
    "Normalizer"
]

MISSING_VALUE_METHODS = [
    "Remove Rows",
    "Remove Columns",
    "Fill with Mean",
    "Fill with Median",
    "Fill with Mode",
    "Fill with Constant",
    "Interpolation",
    "KNN Imputation"
]

OUTLIER_DETECTION_METHODS = [
    "Z-Score",
    "IQR Method",
    "Isolation Forest",
    "Local Outlier Factor"
]

# Classification algorithms
CLASSIFICATION_ALGORITHMS = {
    "Logistic Regression": "sklearn.linear_model.LogisticRegression",
    "Decision Tree": "sklearn.tree.DecisionTreeClassifier",
    "Random Forest": "sklearn.ensemble.RandomForestClassifier",
    "Gradient Boosting": "sklearn.ensemble.GradientBoostingClassifier",
    "XGBoost": "xgboost.XGBClassifier",
    "Support Vector Machine": "sklearn.svm.SVC",
    "K-Nearest Neighbors": "sklearn.neighbors.KNeighborsClassifier",
    "Naive Bayes": "sklearn.naive_bayes.GaussianNB"
}

# Regression algorithms
REGRESSION_ALGORITHMS = {
    "Linear Regression": "sklearn.linear_model.LinearRegression",
    "Ridge Regression": "sklearn.linear_model.Ridge",
    "Lasso Regression": "sklearn.linear_model.Lasso",
    "Decision Tree": "sklearn.tree.DecisionTreeRegressor",
    "Random Forest": "sklearn.ensemble.RandomForestRegressor",
    "Gradient Boosting": "sklearn.ensemble.GradientBoostingRegressor",
    "XGBoost": "xgboost.XGBRegressor",
    "Support Vector Regression": "sklearn.svm.SVR"
}

# Clustering algorithms
CLUSTERING_ALGORITHMS = {
    "K-Means": "sklearn.cluster.KMeans",
    "DBSCAN": "sklearn.cluster.DBSCAN",
    "Hierarchical Clustering": "sklearn.cluster.AgglomerativeClustering",
    "Mean Shift": "sklearn.cluster.MeanShift",
    "Gaussian Mixture": "sklearn.mixture.GaussianMixture"
}

# Dimensionality reduction algorithms
DIMENSIONALITY_REDUCTION_ALGORITHMS = {
    "PCA": "sklearn.decomposition.PCA",
    "t-SNE": "sklearn.manifold.TSNE",
    "UMAP": "umap.UMAP",
    "LDA": "sklearn.discriminant_analysis.LinearDiscriminantAnalysis",
    "Kernel PCA": "sklearn.decomposition.KernelPCA"
}

# Feature selection methods
FEATURE_SELECTION_METHODS = {
    "Variance Threshold": "sklearn.feature_selection.VarianceThreshold",
    "SelectKBest": "sklearn.feature_selection.SelectKBest",
    "RFE": "sklearn.feature_selection.RFE",
    "SelectFromModel": "sklearn.feature_selection.SelectFromModel"
}

# Cross-validation methods
CV_METHODS = {
    "K-Fold": "sklearn.model_selection.KFold",
    "Stratified K-Fold": "sklearn.model_selection.StratifiedKFold",
    "Time Series Split": "sklearn.model_selection.TimeSeriesSplit",
    "Leave One Out": "sklearn.model_selection.LeaveOneOut"
}

# Hyperparameter tuning methods
HYPERPARAMETER_TUNING_METHODS = {
    "Grid Search": "sklearn.model_selection.GridSearchCV",
    "Random Search": "sklearn.model_selection.RandomizedSearchCV",
    "Bayesian Optimization": "skopt.BayesSearchCV"
}
