import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import yaml
from pathlib import Path

# Set up project root and import custom modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)
from src.data_loader.data_loader import DataLoader
from src.models.xgboost_model import XGBoostPipeline
from src.models.random_forest_model import RandomForestPipeline
from src.models.logistic_regression_model import LogisticRegressionPipeline
from src.models.svm_model import SVMPipeline
from src.models.catboost_model import CatBoostPipeline


class App:
    def __init__(self):
        # Initialize data loader and state variables
        self.dataloader = DataLoader()
        self.processed_data = None
        self.preprocessed_data = None
        self.label_map = None
        self.X_train, self.X_test, self.y_train, self.y_test, self.label_map = (
            None,
            None,
            None,
            None,
            None,
        )

        # Load configuration from YAML file
        with open(os.path.join(project_root, "conf.yaml"), "r") as file:
            self.config = yaml.safe_load(file)

        self.Init_Streamlit_Page()

    def load_data(self):
        # Load data using DataLoader and config
        csv_path = (
            Path(project_root)
            / "data"
            / self.config.get("data_file", "breast_cancer.csv")
        )
        X_train, X_test, y_train, y_test, le = self.dataloader.load_data(csv_path)
        return X_train, X_test, y_train, y_test, le

    def Init_Streamlit_Page(self):
        # Set up Streamlit page and sidebar
        st.set_page_config(
            page_title="Machine Learning Model Comparison", layout="wide"
        )
        st.title("Machine Learning Model Comparison")
        st.sidebar.header("User Input Features")
        self.data_load_error = False
        self.data_load_error_msg = None

        # Try to load data and handle errors
        try:
            self.X_train, self.X_test, self.y_train, self.y_test, self.label_map = (
                self.load_data()
            )
        except Exception as e:
            self.data_load_error = True
            self.data_load_error_msg = f"Error loading data: {e}"
            self.X_train, self.X_test, self.y_train, self.y_test, self.label_map = (
                None,
                None,
                None,
                None,
                None,
            )

        # Layout: left for parameters, right for instructions and dataset
        col1, col2 = st.columns(2)
        with col1:
            self.add_parameter_ui()
        with col2:
            if self.data_load_error:
                st.error(self.data_load_error_msg)
                st.info(
                    "Please check and correct the target_col and csv_name settings in the sidebar, then save."
                )
            else:
                # Show usage instructions and dataset button
                st.markdown("""
### How to Use This Page

1. **Configure Data and Target:**
   - In the sidebar, set the target column and CSV file name for your dataset.
   - Click the 💾 icon to save your data settings.
2. **Select and Configure Model:**
   - Choose a machine learning model from the sidebar.
   - Adjust model parameters and feature extraction options as needed.
   - Save your model settings with the 💾 icon.
3. **Train the Model:**
   - Click ⚡ to train the selected model with the current settings.
   - Training metrics and evaluation plots will be displayed below.
4. **Save the Trained Model:**
   - Use the 💾 button in the sidebar to save the trained model as a .pkl file for future use.

_Note: You can always reload or change your data and model settings from the sidebar._
                """)
                # Toggle logic for Show Dataset
                if 'show_dataset' not in st.session_state:
                    st.session_state['show_dataset'] = False
                if st.button(
                    "Show Dataset", help="Display the loaded dataset below.", icon="📊", key="show_dataset_btn"
                ):
                    st.session_state['show_dataset'] = not st.session_state['show_dataset']
                if st.session_state['show_dataset']:
                    self.show_datasets()

                # Toggle logic for Show conf.yaml
                if 'show_conf_yaml' not in st.session_state:
                    st.session_state['show_conf_yaml'] = False
                if st.button(
                    "Show conf.yaml", help="Display the conf.yaml configuration file below.", key="show_conf_yaml_btn", icon="🛠️"
                ):
                    st.session_state['show_conf_yaml'] = not st.session_state['show_conf_yaml']
                if st.session_state['show_conf_yaml']:
                    self.show_conf_yaml()

    def add_parameter_ui(self):
        # Sidebar UI for data and model parameter configuration
        st.sidebar.subheader("Data and Target Settings")
        conf_path = os.path.join(project_root, "conf.yaml")
        with open(conf_path, "r", encoding="utf-8") as f:
            conf = yaml.safe_load(f)
        target_col = st.sidebar.text_input(
            "Target Column (target_col)",
            value=conf.get("target_col", ""),
            key="target_col_input",
        )
        csv_name = st.sidebar.text_input(
            "CSV File Name (csv_name)",
            value=conf.get("csv_name", ""),
            key="csv_name_input",
        )
        if st.sidebar.button(
            "Save Data Settings",
            key="save_data_conf",
            help="Save data and target column settings",
            type="primary",
            icon="💾",
        ):
            conf["target_col"] = target_col
            conf["csv_name"] = csv_name
            with open(conf_path, "w", encoding="utf-8") as f:
                yaml.dump(conf, f, allow_unicode=True, sort_keys=False)
            st.sidebar.success("Data settings saved to conf.yaml.", icon="✅")
            st.session_state["show_yaml_viewer"] = True

        st.sidebar.subheader("Model Selection")
        self.model_type = st.sidebar.selectbox(
            "Select Model",
            ["SVM", "CatBoost", "XGBoost", "Random Forest", "Logistic Regression"],
            help="Choose the machine learning model to train.",
        )

        conf_path = os.path.join(project_root, "conf.yaml")
        with open(conf_path, "r", encoding="utf-8") as f:
            conf = yaml.safe_load(f)
        models_conf = conf.get("models", {})
        model_key = self.model_type
        if model_key == "Random Forest":
            model_key = "RandomForestClassifier"
        model_conf = models_conf.get(model_key, {})

        # Sidebar for model hyperparameters
        st.sidebar.subheader("Model Parameters")
        model_params = {}
        test_size = float(conf.get("test_size", 0.3))
        test_size_min = 0.1
        test_size_max = 0.9
        # Model-specific parameter UI
        if self.model_type == "SVM":
            self.kernel = st.sidebar.selectbox(
                "Kernel",
                ["linear", "poly", "rbf", "sigmoid"],
                index=["linear", "poly", "rbf", "sigmoid"].index(
                    model_conf.get("kernel", "rbf")
                ),
                help="SVM kernel type",
            )
            self.C = st.sidebar.slider(
                "C (Regularization)",
                0.01,
                10.0,
                float(model_conf.get("C", 1.0)),
                key="svm_c",
                help="Regularization parameter for SVM",
            )
            self.test_size = st.sidebar.slider(
                "Test Size",
                test_size_min,
                test_size_max,
                test_size,
                step=0.01,
                key="svm_test_size",
                help="Test set size for splitting data",
            )
            model_params = {"kernel": self.kernel, "C": float(self.C)}
        elif self.model_type == "CatBoost":
            self.iterations = st.sidebar.slider(
                "Iterations",
                50,
                500,
                int(model_conf.get("iterations", 100)),
                key="catboost_iter",
                help="Number of boosting iterations",
            )
            self.learning_rate = st.sidebar.slider(
                "Learning Rate",
                0.01,
                0.5,
                float(model_conf.get("learning_rate", 0.1)),
                key="catboost_lr",
                help="Learning rate for CatBoost",
            )
            self.depth = st.sidebar.slider(
                "Depth",
                1,
                10,
                int(model_conf.get("depth", 6)),
                key="catboost_depth",
                help="Tree depth for CatBoost",
            )
            self.test_size = st.sidebar.slider(
                "Test Size",
                test_size_min,
                test_size_max,
                test_size,
                step=0.01,
                key="catboost_test_size",
                help="Test set size for splitting data",
            )
            model_params = {
                "iterations": int(self.iterations),
                "learning_rate": float(self.learning_rate),
                "depth": int(self.depth),
            }
        elif self.model_type == "XGBoost":
            self.n_estimators = st.sidebar.slider(
                "Number of Estimators",
                50,
                500,
                int(model_conf.get("n_estimators", 100)),
                key="xgb_nest",
                help="Number of trees for XGBoost",
            )
            self.max_depth = st.sidebar.slider(
                "Max Depth",
                1,
                10,
                int(model_conf.get("max_depth", 3)),
                key="xgb_depth",
                help="Maximum tree depth for XGBoost",
            )
            self.learning_rate = st.sidebar.slider(
                "Learning Rate",
                0.01,
                0.5,
                float(model_conf.get("learning_rate", 0.1)),
                key="xgb_lr",
                help="Learning rate for XGBoost",
            )
            self.test_size = st.sidebar.slider(
                "Test Size",
                test_size_min,
                test_size_max,
                test_size,
                step=0.01,
                key="xgb_test_size",
                help="Test set size for splitting data",
            )
            model_params = {
                "n_estimators": int(self.n_estimators),
                "max_depth": int(self.max_depth),
                "learning_rate": float(self.learning_rate),
            }
        elif self.model_type == "Random Forest":
            self.n_estimators_rf = st.sidebar.slider(
                "Number of Estimators",
                50,
                500,
                int(model_conf.get("n_estimators", 100)),
                key="rf_nest",
                help="Number of trees for Random Forest",
            )
            self.max_depth_rf = st.sidebar.slider(
                "Max Depth",
                1,
                10,
                int(model_conf.get("max_depth", 3)),
                key="rf_depth",
                help="Maximum tree depth for Random Forest",
            )
            self.test_size = st.sidebar.slider(
                "Test Size",
                test_size_min,
                test_size_max,
                test_size,
                step=0.01,
                key="rf_test_size",
                help="Test set size for splitting data",
            )
            model_params = {
                "n_estimators": int(self.n_estimators_rf),
                "max_depth": int(self.max_depth_rf),
            }
        elif self.model_type == "Logistic Regression":
            self.penalty = st.sidebar.selectbox(
                "Penalty",
                ["l2", "l1"],
                index=["l2", "l1"].index(model_conf.get("penalty", "l2")),
                key="lr_penalty",
                help="Penalty type for Logistic Regression",
            )
            self.C_lr = st.sidebar.slider(
                "C (Regularization)",
                0.01,
                10.0,
                float(model_conf.get("C", 1.0)),
                key="lr_c",
                help="Regularization parameter for Logistic Regression",
            )
            self.test_size = st.sidebar.slider(
                "Test Size",
                test_size_min,
                test_size_max,
                test_size,
                step=0.01,
                key="lr_test_size",
                help="Test set size for splitting data",
            )
            model_params = {"penalty": self.penalty, "C": float(self.C_lr)}

        # Global feature extraction and selection parameters
        selectkbest_val = int(conf.get("selectkbest", 300))
        feature_extraction_val = bool(conf.get("feature_extraction", True))
        self.feature_extraction = st.sidebar.checkbox(
            "Feature Extraction (PolynomialFeatures + SelectKBest)",
            value=feature_extraction_val,
            key="feature_extraction_checkbox",
            help="Enable feature extraction with polynomial features and SelectKBest.",
        )
        self.selectkbest = st.sidebar.slider(
            "SelectKBest (Number of Features)",
            1,
            500,
            selectkbest_val,
            step=1,
            key="selectkbest_slider",
            disabled=not self.feature_extraction,
            help="Number of features to select with SelectKBest.",
        )
        # Save model settings button
        if st.sidebar.button(
            "Save Model Settings",
            help="Save model and parameter settings to conf.yaml",
            icon="💾",
        ):
            with open(conf_path, "r", encoding="utf-8") as f:
                conf = yaml.safe_load(f)
            model_key_save = self.model_type
            if model_key_save == "Random Forest":
                model_key_save = "RandomForestClassifier"
            elif model_key_save == "Logistic Regression":
                model_key_save = "LogisticRegression"
            conf["model"] = model_key_save
            conf["selectkbest"] = int(self.selectkbest)
            conf["feature_extraction"] = bool(self.feature_extraction)
            conf["test_size"] = float(self.test_size)
            if "models" in conf and model_key_save in conf["models"]:
                conf["models"][model_key_save].update(model_params)
            else:
                if "models" not in conf:
                    conf["models"] = {}
                conf["models"][model_key_save] = model_params
            with open(conf_path, "w", encoding="utf-8") as f:
                yaml.dump(conf, f, allow_unicode=True, sort_keys=False)
            st.sidebar.success("Model and parameters saved to conf.yaml.", icon="✅")
            st.session_state["show_yaml_viewer"] = True
        # Train model button
        if st.button(
            "Train Model",
            help="Train the selected model with current settings",
            icon="⚡",
        ):
            self.train_model()
        # Save trained model button
        if st.sidebar.button(
            "Save Trained Model (.pkl)",
            help="Save the trained pipeline model as a .pkl file",
            icon="💾",
        ):
            save_path = self.save_model()
            if save_path:
                st.sidebar.success(f"Model successfully saved: {save_path}", icon="💾")

    def save_model(self):
        # Save the trained pipeline model to disk
        conf_path = os.path.join(project_root, "conf.yaml")
        with open(conf_path, "r", encoding="utf-8") as f:
            conf = yaml.safe_load(f)
        model_type = conf.get("model")
        models_conf = conf.get("models", {})
        model_conf = models_conf.get(model_type, {})
        if model_type == "SVM":
            model = SVMPipeline(**model_conf)
        elif model_type == "CatBoost":
            model = CatBoostPipeline(**model_conf)
        elif model_type == "XGBoost":
            model = XGBoostPipeline(**model_conf)
        elif model_type == "RandomForestClassifier":
            model = RandomForestPipeline(**model_conf)
        elif model_type == "LogisticRegression":
            model = LogisticRegressionPipeline(**model_conf)
        else:
            st.error(f"Unsupported model type: {model_type}")
            return None
        if self.X_train is None or self.y_train is None:
            st.error("Train data not available. Please reload the data.")
            return None
        model.fit(self.X_train, self.y_train)
        save_path = model.save_model()
        return save_path

    def _get_df_summary(self, df, target_col=None):
        # Generate a summary of the dataframe
        summary = []
        shape = df.shape
        summary.append(f"**Shape:** {shape[0]} rows, {shape[1]} columns  ")
        if target_col and target_col in df.columns:
            class_dist = df[target_col].value_counts().to_dict()
            summary.append(f"**Class Distribution:** {class_dist}")
        missing = df.isnull().sum().sum()
        summary.append(f"**Missing Values:** {missing}")
        dtype_counts = df.dtypes.value_counts().to_dict()
        summary.append(f"**Dtype Counts:** {dtype_counts}")
        return summary

    def show_conf_yaml(self):
        # Display the contents of conf.yaml in a code block
        conf_path = os.path.join(project_root, "conf.yaml")
        try:
            with open(conf_path, "r", encoding="utf-8") as f:
                conf_content = f.read()
            st.markdown("### conf.yaml configuration file")
            st.code(conf_content, language="yaml")
        except Exception as e:
            st.warning(f"Could not load conf.yaml: {e}")

    def show_datasets(self):
        # Display the loaded dataset and its summary
        csv_path = (
            Path(project_root)
            / "data"
            / self.config.get("data_file", "breast_cancer.csv")
        )
        try:
            self.preprocessed_data = pd.read_csv(csv_path)
        except Exception as e:
            self.preprocessed_data = None
            st.warning(
                f"{self.config.get('data_file', 'breast_cancer.csv')} dataset could not be loaded: {e}"
            )

        st.markdown(f"### {self.config.get('data_file', 'breast_cancer.csv')} dataset")
        st.dataframe(self.preprocessed_data)
        if self.preprocessed_data is not None:
            summary = self._get_df_summary(
                self.preprocessed_data, self.config.get("target_col")
            )
            for s in summary:
                st.markdown(s)
       

    def train_model(self):
        # Train the selected model and display evaluation metrics and plots
        conf_path = os.path.join(project_root, "conf.yaml")
        with open(conf_path, "r", encoding="utf-8") as f:
            conf = yaml.safe_load(f)
        self.config = conf
        model_type = self.config.get("model")
        models_conf = self.config.get("models", {})
        model_conf = models_conf.get(model_type, {})
        if model_type == "SVM":
            model = SVMPipeline(**model_conf)
        elif model_type == "CatBoost":
            model = CatBoostPipeline(**model_conf)
        elif model_type == "XGBoost":
            model = XGBoostPipeline(**model_conf)
        elif model_type == "RandomForestClassifier":
            model = RandomForestPipeline(**model_conf)
        elif model_type == "LogisticRegression":
            model = LogisticRegressionPipeline(**model_conf)
        else:
            st.error(f"Unsupported model type: {model_type}")
            return
        if (
            self.X_train is None
            or self.X_test is None
            or self.y_train is None
            or self.y_test is None
        ):
            st.error("Train/test split not available. Please reload the data.")
            return
        model = model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            confusion_matrix,
        )

        acc = accuracy_score(self.y_test, y_pred)
        prec = precision_score(self.y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(self.y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(self.y_test, y_pred, average="weighted", zero_division=0)
        st.success(f"Model trained successfully!", icon="⚡")
        st.markdown(f"**Accuracy:** {acc:.6f}")
        st.markdown(f"**Precision (weighted):** {prec:.6f}")
        st.markdown(f"**Recall (weighted):** {rec:.6f}")
        st.markdown(f"**F1 Score (weighted):** {f1:.6f}")
        from src.streamlit_app.plot_utils import (
            plot_confusion_matrix,
            plot_roc_curve,
            plot_precision_recall_curve,
            plot_calibration_curve,
        )
        import matplotlib.pyplot as plt

        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        label_map = self.label_map
        if label_map is not None and isinstance(label_map, dict):
            sorted_items = sorted(label_map.items())
            labels = [v for k, v in sorted_items]
        else:
            labels = sorted(np.unique(self.y_test))
        fig = plot_confusion_matrix(cm, labels)
        st.pyplot(fig)
        # ROC, PR, Calibration curves
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(self.X_test)
            if y_score.shape[1] == 2:
                y_score_roc = y_score[:, 1]
            else:
                y_score_roc = y_score[:, 1]
            fig_roc = plot_roc_curve(self.y_test, y_score_roc, pos_label=1)
            st.pyplot(fig_roc)
            fig_pr = plot_precision_recall_curve(self.y_test, y_score_roc, pos_label=1)
            st.pyplot(fig_pr)
            fig_cal = plot_calibration_curve(self.y_test, y_score_roc)
            st.pyplot(fig_cal)
        else:
            st.info(
                "The selected model does not support probability prediction, so ROC, Precision-Recall, and Calibration curves cannot be displayed."
            )
