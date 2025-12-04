import pandas as pd
import numpy as np
import pickle
import io
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, RFE
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, mean_squared_error, r2_score,
    mean_absolute_error
)
import optuna
from flask import current_app

# Import all the algorithms from original streamlit app
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier,
    GradientBoostingClassifier, BaggingClassifier, IsolationForest
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier
from sklearn.linear_model import (
    LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet,
    PassiveAggressiveClassifier, Perceptron, SGDClassifier, RidgeClassifier
)
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

from app.services.s3_service import S3Service

class MLService:
    def __init__(self):
        self.s3_service = S3Service()
        self.supported_algorithms = self._get_supported_algorithms()
        
    def _get_supported_algorithms(self):
        """Return dictionary of supported algorithms"""
        return {
            # Classification algorithms
            'Random Forest Classifier': {
                'class': RandomForestClassifier,
                'type': 'classification',
                'default_params': {'n_estimators': 100, 'random_state': 42}
            },
            'SVM': {
                'class': SVC,
                'type': 'classification',
                'default_params': {'random_state': 42, 'probability': True}
            },
            'Decision Tree Classifier': {
                'class': DecisionTreeClassifier,
                'type': 'classification',
                'default_params': {'random_state': 42}
            },
            'Logistic Regression': {
                'class': LogisticRegression,
                'type': 'classification',
                'default_params': {'random_state': 42, 'max_iter': 1000}
            },
            'AdaBoost Classifier': {
                'class': AdaBoostClassifier,
                'type': 'classification',
                'default_params': {'random_state': 42}
            },
            'Extra Trees Classifier': {
                'class': ExtraTreeClassifier,
                'type': 'classification',
                'default_params': {'random_state': 42}
            },
            'Gradient Boosting Classifier': {
                'class': GradientBoostingClassifier,
                'type': 'classification',
                'default_params': {'random_state': 42}
            },
            'K-Nearest Neighbors Classifier': {
                'class': KNeighborsClassifier,
                'type': 'classification',
                'default_params': {}
            },
            'Gaussian Naive Bayes Classifier': {
                'class': GaussianNB,
                'type': 'classification',
                'default_params': {}
            },
            'Bernoulli Naive Bayes Classifier': {
                'class': BernoulliNB,
                'type': 'classification',
                'default_params': {}
            },
            'Multinomial Naive Bayes Classifier': {
                'class': MultinomialNB,
                'type': 'classification',
                'default_params': {}
            },
            'XGBoost Classifier': {
                'class': XGBClassifier,
                'type': 'classification',
                'default_params': {'random_state': 42, 'eval_metric': 'logloss'}
            },
            'LightGBM Classifier': {
                'class': LGBMClassifier,
                'type': 'classification',
                'default_params': {'random_state': 42, 'verbose': -1}
            },
            'CatBoost Classifier': {
                'class': CatBoostClassifier,
                'type': 'classification',
                'default_params': {'random_state': 42, 'verbose': False}
            },
            'MLP Classifier': {
                'class': MLPClassifier,
                'type': 'classification',
                'default_params': {'random_state': 42, 'max_iter': 1000}
            },
            
            # Regression algorithms
            'Random Forest Regressor': {
                'class': RandomForestRegressor,
                'type': 'regression',
                'default_params': {'n_estimators': 100, 'random_state': 42}
            },
            'Linear Regression': {
                'class': LinearRegression,
                'type': 'regression',
                'default_params': {}
            },
            'Ridge Regression': {
                'class': Ridge,
                'type': 'regression',
                'default_params': {'random_state': 42}
            },
            'Lasso Regression': {
                'class': Lasso,
                'type': 'regression',
                'default_params': {'random_state': 42}
            },
            'ElasticNet Regression': {
                'class': ElasticNet,
                'type': 'regression',
                'default_params': {'random_state': 42}
            },
            'Decision Tree Regressor': {
                'class': DecisionTreeRegressor,
                'type': 'regression',
                'default_params': {'random_state': 42}
            },
            'SVR': {
                'class': SVR,
                'type': 'regression',
                'default_params': {}
            },
            'K-Nearest Neighbors Regressor': {
                'class': KNeighborsRegressor,
                'type': 'regression',
                'default_params': {}
            },
            'XGBoost Regressor': {
                'class': XGBRegressor,
                'type': 'regression',
                'default_params': {'random_state': 42}
            },
            'LightGBM Regressor': {
                'class': LGBMRegressor,
                'type': 'regression',
                'default_params': {'random_state': 42, 'verbose': -1}
            },
            'CatBoost Regressor': {
                'class': CatBoostRegressor,
                'type': 'regression',
                'default_params': {'random_state': 42, 'verbose': False}
            },
            'MLP Regressor': {
                'class': MLPRegressor,
                'type': 'regression',
                'default_params': {'random_state': 42, 'max_iter': 1000}
            }
        }
    
    def get_supported_algorithms(self):
        """Get supported algorithms info"""
        return {name: {
            'type': info['type'],
            'default_params': info['default_params']
        } for name, info in self.supported_algorithms.items()}
    
    def preprocess_data(self, df, preprocessing_config, target_column=None, fit_preprocessors=True):
        """Apply preprocessing to the data"""
        try:
            processed_df = df.copy()
            preprocessors = {}
            
            if target_column and target_column in processed_df.columns:
                X = processed_df.drop(columns=[target_column])
                y = processed_df[target_column]
            else:
                X = processed_df
                y = None
            
            # Handle missing values
            missing_strategy = preprocessing_config.get('missing_values', 'mean')
            if missing_strategy != 'none':
                if fit_preprocessors:
                    imputer = SimpleImputer(strategy=missing_strategy)
                    X_numeric = X.select_dtypes(include=[np.number])
                    if not X_numeric.empty:
                        X[X_numeric.columns] = imputer.fit_transform(X_numeric)
                        preprocessors['numeric_imputer'] = imputer
                    
                    # Handle categorical missing values
                    X_categorical = X.select_dtypes(include=['object'])
                    if not X_categorical.empty:
                        cat_imputer = SimpleImputer(strategy='most_frequent')
                        X[X_categorical.columns] = cat_imputer.fit_transform(X_categorical)
                        preprocessors['categorical_imputer'] = cat_imputer
            
            # Encode categorical variables
            encoding_method = preprocessing_config.get('encoding', 'label')
            categorical_columns = X.select_dtypes(include=['object']).columns
            
            if len(categorical_columns) > 0:
                if encoding_method == 'label':
                    label_encoders = {}
                    for col in categorical_columns:
                        if fit_preprocessors:
                            le = LabelEncoder()
                            X[col] = le.fit_transform(X[col])
                            label_encoders[col] = le
                        else:
                            # Use existing encoders
                            le = preprocessors.get('label_encoders', {}).get(col)
                            if le:
                                X[col] = le.transform(X[col])
                    
                    if fit_preprocessors:
                        preprocessors['label_encoders'] = label_encoders
                
                elif encoding_method == 'onehot':
                    if fit_preprocessors:
                        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                        encoded_features = encoder.fit_transform(X[categorical_columns])
                        feature_names = encoder.get_feature_names_out(categorical_columns)
                        
                        # Replace categorical columns with encoded ones
                        X = X.drop(columns=categorical_columns)
                        encoded_df = pd.DataFrame(encoded_features, columns=feature_names, index=X.index)
                        X = pd.concat([X, encoded_df], axis=1)
                        
                        preprocessors['onehot_encoder'] = encoder
                        preprocessors['categorical_columns'] = categorical_columns.tolist()
            
            # Scale features
            scaling_method = preprocessing_config.get('scaling', 'none')
            if scaling_method != 'none':
                numeric_columns = X.select_dtypes(include=[np.number]).columns
                
                if len(numeric_columns) > 0:
                    if scaling_method == 'standard':
                        scaler = StandardScaler()
                    elif scaling_method == 'minmax':
                        scaler = MinMaxScaler()
                    elif scaling_method == 'robust':
                        scaler = RobustScaler()
                    else:
                        scaler = StandardScaler()
                    
                    if fit_preprocessors:
                        X[numeric_columns] = scaler.fit_transform(X[numeric_columns])
                        preprocessors['scaler'] = scaler
                        preprocessors['numeric_columns'] = numeric_columns.tolist()
            
            # Feature selection
            feature_selection = preprocessing_config.get('feature_selection', 'none')
            if feature_selection != 'none' and y is not None and fit_preprocessors:
                if feature_selection == 'variance_threshold':
                    selector = VarianceThreshold(threshold=0.01)
                    X_selected = selector.fit_transform(X)
                    selected_features = X.columns[selector.get_support()].tolist()
                    X = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
                    preprocessors['feature_selector'] = selector
                    preprocessors['selected_features'] = selected_features
            
            return X, y, preprocessors
            
        except Exception as e:
            current_app.logger.error(f"Preprocessing error: {str(e)}")
            return None, None, {}
    
    def train_model(self, dataset, experiment, target_column, preprocessing_config, training_config):
        """Train a machine learning model"""
        try:
            # Load dataset
            s3_service = S3Service()
            file_content = s3_service.download_file(dataset.path)
            
            if dataset.file_type == 'csv':
                df = pd.read_csv(io.BytesIO(file_content))
            elif dataset.file_type in ['xlsx', 'xls']:
                df = pd.read_excel(io.BytesIO(file_content))
            else:
                return None
            
            # Preprocess data
            X, y, preprocessors = self.preprocess_data(df, preprocessing_config, target_column)
            
            if X is None or y is None:
                return None
            
            # Split data
            test_size = training_config.get('test_size', 0.2)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Get model
            model_config = self.supported_algorithms[experiment.model_type]
            model_class = model_config['class']
            default_params = model_config['default_params'].copy()
            
            # Update with custom parameters
            params = experiment.params or {}
            default_params.update(params)
            
            model = model_class(**default_params)
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = self._calculate_metrics(model_config['type'], y_test, y_pred, model, X_test)
            
            # Cross validation
            cv_scores = cross_val_score(model, X, y, cv=5)
            metrics['cross_validation'] = {
                'scores': cv_scores.tolist(),
                'mean': cv_scores.mean(),
                'std': cv_scores.std()
            }
            
            # Save model and preprocessors
            model_data = {
                'model': model,
                'preprocessors': preprocessors,
                'feature_columns': X.columns.tolist(),
                'target_column': target_column,
                'model_type': experiment.model_type
            }
            
            model_path = f"models/experiment_{experiment.id}_model.pkl"
            model_buffer = io.BytesIO()
            pickle.dump(model_data, model_buffer)
            model_buffer.seek(0)
            
            if not s3_service.upload_file(model_buffer, model_path):
                return None
            
            return {
                'model_path': model_path,
                'metrics': metrics,
                'feature_count': len(X.columns),
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
        except Exception as e:
            current_app.logger.error(f"Model training error: {str(e)}")
            return None
    
    def _calculate_metrics(self, problem_type, y_true, y_pred, model, X_test):
        """Calculate evaluation metrics"""
        metrics = {}
        
        try:
            if problem_type == 'classification':
                metrics['accuracy'] = accuracy_score(y_true, y_pred)
                metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
                metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
                metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted')
                
                # ROC AUC for binary classification
                if len(np.unique(y_true)) == 2 and hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)[:, 1]
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
                
                # Confusion matrix
                cm = confusion_matrix(y_true, y_pred)
                metrics['confusion_matrix'] = cm.tolist()
                
                # Classification report
                report = classification_report(y_true, y_pred, output_dict=True)
                metrics['classification_report'] = report
                
            else:  # regression
                metrics['mae'] = mean_absolute_error(y_true, y_pred)
                metrics['mse'] = mean_squared_error(y_true, y_pred)
                metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
                metrics['r2_score'] = r2_score(y_true, y_pred)
                
                # Calculate MAPE if no zeros in y_true
                if not np.any(y_true == 0):
                    metrics['mape'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                
        except Exception as e:
            current_app.logger.error(f"Metrics calculation error: {str(e)}")
            
        return metrics
    
    def predict_single(self, model_record, features):
        """Make single prediction"""
        try:
            # Load model from S3
            model_data = self._load_model(model_record)
            if not model_data:
                return None
            
            model = model_data['model']
            preprocessors = model_data['preprocessors']
            feature_columns = model_data['feature_columns']
            
            # Prepare input data
            input_df = pd.DataFrame([features], columns=feature_columns)
            
            # Apply preprocessing (without fitting)
            processed_input, _, _ = self.preprocess_data(
                input_df, {}, fit_preprocessors=False
            )
            
            if processed_input is None:
                return None
            
            # Make prediction
            prediction = model.predict(processed_input)[0]
            result = {'prediction': prediction}
            
            # Add probability for classification models
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(processed_input)[0]
                result['probability'] = probabilities.tolist()
                result['confidence'] = np.max(probabilities)
            
            return result
            
        except Exception as e:
            current_app.logger.error(f"Single prediction error: {str(e)}")
            return None
    
    def predict_batch(self, model_record, df):
        """Make batch predictions"""
        try:
            # Load model from S3
            model_data = self._load_model(model_record)
            if not model_data:
                return None
            
            model = model_data['model']
            preprocessors = model_data['preprocessors']
            feature_columns = model_data['feature_columns']
            
            # Ensure input has same columns as training data
            missing_columns = set(feature_columns) - set(df.columns)
            extra_columns = set(df.columns) - set(feature_columns)
            
            if missing_columns:
                current_app.logger.warning(f"Missing columns: {missing_columns}")
                # Add missing columns with default values
                for col in missing_columns:
                    df[col] = 0
            
            if extra_columns:
                current_app.logger.warning(f"Extra columns will be ignored: {extra_columns}")
                df = df[feature_columns]
            
            # Apply preprocessing
            processed_df, _, _ = self.preprocess_data(df, {}, fit_preprocessors=False)
            
            if processed_df is None:
                return None
            
            # Make predictions
            predictions = model.predict(processed_df)
            result = {'predictions': predictions.tolist()}
            
            # Add probabilities for classification models
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(processed_df)
                result['probabilities'] = probabilities.tolist()
            
            return result
            
        except Exception as e:
            current_app.logger.error(f"Batch prediction error: {str(e)}")
            return None
    
    def _load_model(self, model_record):
        """Load model from S3"""
        try:
            model_content = self.s3_service.download_file(model_record.s3_path)
            if not model_content:
                return None
            
            model_data = pickle.load(io.BytesIO(model_content))
            return model_data
            
        except Exception as e:
            current_app.logger.error(f"Model loading error: {str(e)}")
            return None
    
    def get_model_info(self, model_record):
        """Get detailed model information"""
        try:
            model_data = self._load_model(model_record)
            if not model_data:
                return None
            
            model = model_data['model']
            
            info = {
                'model_type': model_data['model_type'],
                'feature_count': len(model_data['feature_columns']),
                'feature_columns': model_data['feature_columns'],
                'target_column': model_data['target_column']
            }
            
            # Add model-specific information
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_importance = dict(zip(model_data['feature_columns'], importances))
                info['feature_importance'] = feature_importance
            
            if hasattr(model, 'coef_'):
                info['coefficients'] = model.coef_.tolist()
            
            return info
            
        except Exception as e:
            current_app.logger.error(f"Get model info error: {str(e)}")
            return None