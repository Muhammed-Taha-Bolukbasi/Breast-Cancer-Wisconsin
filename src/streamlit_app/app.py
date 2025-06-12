import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import yaml
from sklearn.metrics import accuracy_score

# Add project root directory to sys.path so that modules in src can be imported
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)
from src.data_loader.data_loader import DataLoader
from src.data_preprocessing.data_preprocessor import DataPreprocessor
from src.models.svm_model import SVMModel
from src.models.catboost_model import CatBoostModel
from src.models.xgboost_model import XGBoost
from src.models.random_forest_model import RandomForest
from src.models.logistic_regression_model import LogisticRegressionModel
from sklearn.model_selection import train_test_split

class App:
    def __init__(self):
        self.dataloader = DataLoader()
        self.preprocessor = DataPreprocessor()
        self.processed_data = None
        self.preprocessed_data = None
        self.label_map = None
        with open(os.path.join(project_root, 'conf.yaml'), 'r') as file:
            self.config = yaml.safe_load(file)
        self.Init_Streamlit_Page()    

    def run(self):
        self.processed_data, self.preprocessed_data, self.label_map = self.load_data(os.path.join(project_root, "data", self.config.get("csv_name", "breast_cancer.csv")))

        #self.generate()

    def load_data(self, csv_path):
        """Load and preprocess data from a CSV file."""
        df_preprocessed = pd.read_csv(csv_path)
        df, df_target = self.dataloader.load_data(csv_path)
        df_processed, label_map = self.preprocessor.fit_transform(df, df_target, return_mapping=True)
        return df_processed, df_preprocessed, label_map


    def Init_Streamlit_Page(self):
        # Set the title and layout of the Streamlit app
        st.set_page_config(page_title="Machine Learning Model Comparison", layout="wide")
        st.title("Machine Learning Model Comparison")
        st.sidebar.header("User Input Features")
        self.processed_data, self.preprocessed_data, self.label_map = self.load_data(os.path.join(project_root, "data", self.config.get("csv_name", "breast_cancer.csv")))

        # --- Sidebar Buttons kaldırıldı, ana içerik doğrudan gösterilecek ---
        col1, col2 = st.columns(2)
        with col1:
            self.add_parameter_ui()
        with col2:
            self.show_datasets()

    def add_parameter_ui(self):
        """Add UI for model parameters and training."""
        # Sidebar for model selection
        st.sidebar.subheader("Model Selection")
        self.model_type = st.sidebar.selectbox("Select Model", ["SVM", "CatBoost", "XGBoost", "Random Forest", "Logistic Regression"])

        # Get initial values from conf.yaml
        conf_path = os.path.join(project_root, 'conf.yaml')
        with open(conf_path, 'r', encoding='utf-8') as f:
            conf = yaml.safe_load(f)
        models_conf = conf.get('models', {})
        # Model key mapping
        model_key = self.model_type
        if model_key == "Random Forest":
            model_key = "RandomForestClassifier"
        model_conf = models_conf.get(model_key, {})

        # Sidebar for model parameters (with defaults from conf.yaml)
        st.sidebar.subheader("Model Parameters")
        model_params = {}
        # Test size default
        test_size = float(model_conf.get("test_size", conf.get("test_size", 0.3)))
        test_size_min = 0.1
        test_size_max = 0.9
        if self.model_type == "SVM":
            self.kernel = st.sidebar.selectbox(
                "Kernel", ["linear", "poly", "rbf", "sigmoid"],
                index=["linear", "poly", "rbf", "sigmoid"].index(model_conf.get("kernel", "rbf"))
            )
            self.C = st.sidebar.slider(
                "C (Regularization)", 0.01, 10.0, float(model_conf.get("C", 1.0)), key="svm_c"
            )
            self.test_size = st.sidebar.slider(
                "Test Size", test_size_min, test_size_max, test_size, step=0.01, key="svm_test_size"
            )
            model_params = {"kernel": self.kernel, "C": float(self.C), "test_size": float(self.test_size)}
        elif self.model_type == "CatBoost":
            self.iterations = st.sidebar.slider(
                "Iterations", 50, 500, int(model_conf.get("iterations", 100)), key="catboost_iter"
            )
            self.learning_rate = st.sidebar.slider(
                "Learning Rate", 0.01, 0.5, float(model_conf.get("learning_rate", 0.1)), key="catboost_lr"
            )
            self.depth = st.sidebar.slider(
                "Depth", 1, 10, int(model_conf.get("depth", 6)), key="catboost_depth"
            )
            self.test_size = st.sidebar.slider(
                "Test Size", test_size_min, test_size_max, test_size, step=0.01, key="catboost_test_size"
            )
            model_params = {"iterations": int(self.iterations), "learning_rate": float(self.learning_rate), "depth": int(self.depth), "test_size": float(self.test_size)}
        elif self.model_type == "XGBoost":
            self.n_estimators = st.sidebar.slider(
                "Number of Estimators", 50, 500, int(model_conf.get("n_estimators", 100)), key="xgb_nest"
            )
            self.max_depth = st.sidebar.slider(
                "Max Depth", 1, 10, int(model_conf.get("max_depth", 3)), key="xgb_depth"
            )
            self.learning_rate = st.sidebar.slider(
                "Learning Rate", 0.01, 0.5, float(model_conf.get("learning_rate", 0.1)), key="xgb_lr"
            )
            self.test_size = st.sidebar.slider(
                "Test Size", test_size_min, test_size_max, test_size, step=0.01, key="xgb_test_size"
            )
            model_params = {"n_estimators": int(self.n_estimators), "max_depth": int(self.max_depth), "learning_rate": float(self.learning_rate), "test_size": float(self.test_size)}
        elif self.model_type == "Random Forest":
            self.n_estimators_rf = st.sidebar.slider(
                "Number of Estimators", 50, 500, int(model_conf.get("n_estimators", 100)), key="rf_nest"
            )
            self.max_depth_rf = st.sidebar.slider(
                "Max Depth", 1, 10, int(model_conf.get("max_depth", 3)), key="rf_depth"
            )
            self.test_size = st.sidebar.slider(
                "Test Size", test_size_min, test_size_max, test_size, step=0.01, key="rf_test_size"
            )
            model_params = {"n_estimators": int(self.n_estimators_rf), "max_depth": int(self.max_depth_rf), "test_size": float(self.test_size)}
        elif self.model_type == "Logistic Regression":
            self.penalty = st.sidebar.selectbox(
                "Penalty", ["l2", "l1"],
                index=["l2", "l1"].index(model_conf.get("penalty", "l2")), key="lr_penalty"
            )
            self.C_lr = st.sidebar.slider(
                "C (Regularization)", 0.01, 10.0, float(model_conf.get("C", 1.0)), key="lr_c"
            )
            self.test_size = st.sidebar.slider(
                "Test Size", test_size_min, test_size_max, test_size, step=0.01, key="lr_test_size"
            )
            model_params = {"penalty": self.penalty, "C": float(self.C_lr), "test_size": float(self.test_size)}

        # SelectKBest parametresi (global, modelden bağımsız)
        selectkbest_val = int(conf.get("selectkbest", 300))
        self.selectkbest = st.sidebar.slider(
            "SelectKBest (Özellik Sayısı)", 1, 500, selectkbest_val, step=1, key="selectkbest_slider"
        )

        # Save to conf.yaml button
        if st.sidebar.button("Model Ayarlarını Kaydet (conf.yaml)"):
            with open(conf_path, 'r', encoding='utf-8') as f:
                conf = yaml.safe_load(f)
            # Update only the selected model's parameters under 'models:' and the 'model' field
            model_key_save = self.model_type
            if model_key_save == "Random Forest":
                model_key_save = "RandomForestClassifier"
            elif model_key_save == "Logistic Regression":
                model_key_save = "LogisticRegression"
            conf['model'] = model_key_save
            conf['selectkbest'] = int(self.selectkbest)
            if 'models' in conf and model_key_save in conf['models']:
                conf['models'][model_key_save].update(model_params)
            else:
                if 'models' not in conf:
                    conf['models'] = {}
                conf['models'][model_key_save] = model_params
            with open(conf_path, 'w', encoding='utf-8') as f:
                yaml.dump(conf, f, allow_unicode=True, sort_keys=False)
            st.sidebar.success("Model ve parametreler conf.yaml dosyasına kaydedildi.")
            st.session_state['show_yaml_viewer'] = True
        # Train Model button
        if st.button("Train Model"):
            self.train_model()

    def show_datasets(self):
        st.markdown("### Processed Data")
        st.dataframe(self.processed_data)
        # --- Processed Data Summary ---
        if self.processed_data is not None:
            shape = self.processed_data.shape
            target_col = 'Target_Label' if 'Target_Label' in self.processed_data.columns else None
            if target_col:
                class_dist = self.processed_data[target_col].value_counts().to_dict() 
            else:
                class_dist = None
            missing = self.processed_data.isnull().sum().sum() 
            st.markdown(f"**Shape:** {shape[0]} rows, {shape[1]} columns  ")
            if class_dist:
                st.markdown(f"**Class Distribution:** {class_dist}")
            st.markdown(f"**Missing Values:** {missing}")

        st.markdown("### Preprocessed Data")
        st.dataframe(self.preprocessed_data)
        # --- Preprocessed Data Summary ---
        if self.preprocessed_data is not None:
            shape = self.preprocessed_data.shape
            # Try to find a target/diagnosis column
            diag_col = None
            for col in ['Diagnosis', 'diagnosis', 'Target_Label', 'target', 'label']:
                if col in self.preprocessed_data.columns:
                    diag_col = col
                    break
            if diag_col:
                class_dist = self.preprocessed_data[diag_col].value_counts().to_dict()
            else:
                class_dist = None
            missing = self.preprocessed_data.isnull().sum().sum()
            st.markdown(f"**Shape:** {shape[0]} rows, {shape[1]} columns  ")
            if class_dist:
                st.markdown(f"**Class Distribution:** {class_dist}")
            st.markdown(f"**Missing Values:** {missing}")


    def train_model(self):
        # Reload config to get latest parameters
        conf_path = os.path.join(project_root, 'conf.yaml')
        with open(conf_path, 'r', encoding='utf-8') as f:
            conf = yaml.safe_load(f)
        self.config = conf
        model = None
        model_type = self.config.get("model")
        models_conf = self.config.get("models", {})
        model_conf = models_conf.get(model_type, {})
        # Model selection with correct param extraction
        if model_type == "SVM":
            kernel = model_conf.get("kernel", "rbf")
            C = float(model_conf.get("C", 1.0))
            test_size = float(model_conf.get("test_size", conf.get("test_size", 0.3)))
            model = SVMModel(kernel=kernel, C=C)
        elif model_type == "CatBoost":
            test_size = float(model_conf.get("test_size", conf.get("test_size", 0.3)))
            model = CatBoostModel(
                iterations=int(model_conf.get("iterations", 100)),
                learning_rate=float(model_conf.get("learning_rate", 0.1)),
                depth=int(model_conf.get("depth", 6))
            )
        elif model_type == "XGBoost":
            test_size = float(model_conf.get("test_size", conf.get("test_size", 0.3)))
            model = XGBoost(
                n_estimators=int(model_conf.get("n_estimators", 100)),
                max_depth=int(model_conf.get("max_depth", 3)),
                learning_rate=float(model_conf.get("learning_rate", 0.1))
            )
        elif model_type == "RandomForestClassifier":
            test_size = float(model_conf.get("test_size", conf.get("test_size", 0.3)))
            model = RandomForest(
                n_estimators=int(model_conf.get("n_estimators", 100)),
                max_depth=int(model_conf.get("max_depth", 3))
            )
        elif model_type == "LogisticRegression":
            penalty = model_conf.get("penalty", "l2")
            C = float(model_conf.get("C", 1.0))
            test_size = float(model_conf.get("test_size", conf.get("test_size", 0.3)))
            # Solver seçimi penalty'ye göre otomatik
            if penalty == "l1":
                solver = "liblinear"
            else:
                solver = "lbfgs"
            model = LogisticRegressionModel(penalty=penalty, C=C, solver=solver)
        else:
            st.error(f"Unsupported model type: {model_type}")
            return
        # Ensure processed_data is loaded
        if self.processed_data is None:
            self.processed_data, _ , _= self.load_data(os.path.join(project_root, "data", self.config.get("csv_name", "breast_cancer.csv")))
        X_train, X_test, y_train, y_test = train_test_split(
            self.processed_data.drop(columns=["Target_Label"]), 
            self.processed_data["Target_Label"],  
            test_size=test_size, random_state=42) 
        model.fit(X_train, y_train) 
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.success(f"Model trained successfully! Accuracy: {acc:.6f}",icon="✅")

        # Confusion Matrix plot
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        cm = confusion_matrix(y_test, y_pred)
        # --- Label names for axes ---
        label_map = self.label_map
        if label_map is not None and isinstance(label_map, dict):
            # Inverse mapping: encoded value -> original label
            # Sort by encoded value (key)
            sorted_items = sorted(label_map.items())
            labels = [v for k, v in sorted_items]
        else:
            # Fallback: show unique values in y_test
            labels = sorted(np.unique(y_test))
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=labels, yticklabels=labels)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
