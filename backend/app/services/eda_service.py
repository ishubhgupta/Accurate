import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
import plotly.express as px
import plotly.graph_objects as go
import json
import io
from app.services.s3_service import S3Service
from flask import current_app

class EDAService:
    def __init__(self):
        self.s3_service = S3Service()
    
    def load_dataset(self, dataset):
        """Load dataset from S3"""
        try:
            file_content = self.s3_service.download_file(dataset.path)
            if not file_content:
                return None
            
            if dataset.file_type == 'csv':
                df = pd.read_csv(io.BytesIO(file_content))
            elif dataset.file_type in ['xlsx', 'xls']:
                df = pd.read_excel(io.BytesIO(file_content))
            else:
                return None
            
            return df
        except Exception as e:
            current_app.logger.error(f"Error loading dataset: {str(e)}")
            return None
    
    def get_quick_summary(self, dataset):
        """Generate quick EDA summary without full profiling"""
        df = self.load_dataset(dataset)
        if df is None:
            return None
        
        try:
            summary = {
                'basic_info': {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'memory_usage': df.memory_usage(deep=True).sum(),
                    'dtypes': df.dtypes.value_counts().to_dict()
                },
                'missing_values': {
                    'total_missing': df.isnull().sum().sum(),
                    'missing_by_column': df.isnull().sum().to_dict(),
                    'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict()
                },
                'numerical_summary': {},
                'categorical_summary': {},
                'correlation_data': None
            }
            
            # Numerical columns summary
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                summary['numerical_summary'] = {
                    'count': len(numerical_cols),
                    'columns': numerical_cols.tolist(),
                    'statistics': df[numerical_cols].describe().to_dict()
                }
                
                # Correlation matrix for numerical columns
                if len(numerical_cols) > 1:
                    corr_matrix = df[numerical_cols].corr()
                    summary['correlation_data'] = corr_matrix.to_dict()
            
            # Categorical columns summary
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                cat_summary = {}
                for col in categorical_cols:
                    cat_summary[col] = {
                        'unique_count': df[col].nunique(),
                        'top_values': df[col].value_counts().head(10).to_dict()
                    }
                summary['categorical_summary'] = {
                    'count': len(categorical_cols),
                    'columns': categorical_cols.tolist(),
                    'details': cat_summary
                }
            
            return summary
            
        except Exception as e:
            current_app.logger.error(f"Error generating quick summary: {str(e)}")
            return None
    
    def generate_full_report(self, dataset, job_id):
        """Generate comprehensive EDA report using ydata-profiling"""
        df = self.load_dataset(dataset)
        if df is None:
            return False
        
        try:
            # Configure profiling
            profile = ProfileReport(
                df,
                title=f"EDA Report - {dataset.name}",
                explorative=True,
                minimal=False
            )
            
            # Generate HTML report
            html_report = profile.to_html()
            
            # Upload report to S3
            report_path = f"eda_reports/{dataset.id}_{job_id}.html"
            html_buffer = io.BytesIO(html_report.encode('utf-8'))
            
            if not self.s3_service.upload_file(html_buffer, report_path):
                return False
            
            # Generate JSON summary for API
            json_summary = profile.to_json()
            json_path = f"eda_reports/{dataset.id}_{job_id}.json"
            json_buffer = io.BytesIO(json_summary.encode('utf-8'))
            
            if not self.s3_service.upload_file(json_buffer, json_path):
                return False
            
            return {
                'html_report_path': report_path,
                'json_summary_path': json_path
            }
            
        except Exception as e:
            current_app.logger.error(f"Error generating full report: {str(e)}")
            return False
    
    def get_report_url(self, dataset):
        """Get presigned URL for EDA report"""
        # Find the most recent EDA report for this dataset
        report_path = f"eda_reports/{dataset.id}_*.html"
        # In a real implementation, you'd query the database for the latest report path
        # For now, we'll assume a simple naming convention
        
        try:
            # This is a simplified approach - in production, store report paths in database
            latest_report_path = f"eda_reports/{dataset.id}_latest.html"
            return self.s3_service.get_file_url(latest_report_path)
        except Exception as e:
            current_app.logger.error(f"Error getting report URL: {str(e)}")
            return None
    
    def generate_charts(self, dataset):
        """Generate interactive charts for the dashboard"""
        df = self.load_dataset(dataset)
        if df is None:
            return None
        
        try:
            charts = {}
            
            # Distribution plots for numerical columns
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            for col in numerical_cols[:5]:  # Limit to first 5 columns
                fig = px.histogram(df, x=col, title=f'Distribution of {col}')
                charts[f'dist_{col}'] = fig.to_json()
            
            # Correlation heatmap
            if len(numerical_cols) > 1:
                corr_matrix = df[numerical_cols].corr()
                fig = px.imshow(
                    corr_matrix,
                    title='Correlation Matrix',
                    color_continuous_scale='RdBu',
                    aspect='auto'
                )
                charts['correlation_heatmap'] = fig.to_json()
            
            # Missing values chart
            missing_data = df.isnull().sum()
            if missing_data.sum() > 0:
                fig = px.bar(
                    x=missing_data.index,
                    y=missing_data.values,
                    title='Missing Values by Column'
                )
                charts['missing_values'] = fig.to_json()
            
            return charts
            
        except Exception as e:
            current_app.logger.error(f"Error generating charts: {str(e)}")
            return None