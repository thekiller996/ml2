import os

structure = {
    "ml_platform": [
        "app.py",
        "config.py",
        ("core", ["__init__.py", "session.py", "constants.py"]),
        ("ui", ["__init__.py", "sidebar.py", "common.py", "styles.py"]),
        ("data", ["__init__.py", "loader.py", "explorer.py", "exporter.py"]),
        ("preprocessing", [
            "__init__.py", "missing_values.py", "outliers.py", "encoding.py", "scaling.py", "image_preprocessing.py"
        ]),
        ("feature_engineering", [
            "__init__.py", "feature_selection.py", "feature_creation.py", "dim_reduction.py", "feature_transform.py"
        ]),
        ("models", [
            "__init__.py", "classifier.py", "regressor.py", "clusterer.py", "evaluation.py", "tuning.py"
        ]),
        ("pages", [
            "__init__.py", "project_setup.py", "data_upload.py", "exploratory_analysis.py", "data_preprocessing.py",
            "feature_engineering.py", "model_training.py", "model_evaluation.py", "prediction.py"
        ]),
        ("plugins", [
            "__init__.py", "plugin_manager.py", "plugin_base.py", "hooks.py", "registry.py", "utils.py",
            ("examples", ["__init__.py", "sample_plugin.py"])
        ]),
        ("utils", ["__init__.py", "file_ops.py", "visualizations.py", "stats.py", "misc.py"]),
    ]
}

def create_structure(base_path, structure):
    for item in structure:
        if isinstance(item, str):
            file_path = os.path.join(base_path, item)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w'): pass
        elif isinstance(item, tuple):
            folder, contents = item
            folder_path = os.path.join(base_path, folder)
            os.makedirs(folder_path, exist_ok=True)
            create_structure(folder_path, contents)

if __name__ == "__main__":
    create_structure(".", structure["ml_platform"])
    print("Project structure created successfully.")
