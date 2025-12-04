from app import db
from datetime import datetime

class Dataset(db.Model):
    __tablename__ = 'datasets'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    path = db.Column(db.String(500), nullable=False)  # S3 path
    size = db.Column(db.BigInteger, nullable=False)  # Size in bytes
    upload_time = db.Column(db.DateTime, default=datetime.utcnow)
    file_type = db.Column(db.String(50), nullable=False)
    columns = db.Column(db.JSON)  # Store column information
    rows = db.Column(db.Integer)
    status = db.Column(db.String(50), default='uploaded')  # uploaded, processed, error
    
    # Relationships
    experiments = db.relationship('Experiment', backref='dataset', lazy=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'path': self.path,
            'size': self.size,
            'upload_time': self.upload_time.isoformat() if self.upload_time else None,
            'file_type': self.file_type,
            'columns': self.columns,
            'rows': self.rows,
            'status': self.status
        }

class Experiment(db.Model):
    __tablename__ = 'experiments'
    
    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey('datasets.id'), nullable=False)
    model_type = db.Column(db.String(100), nullable=False)
    params = db.Column(db.JSON)  # Model parameters
    metrics = db.Column(db.JSON)  # Evaluation metrics
    status = db.Column(db.String(50), default='pending')  # pending, running, completed, failed
    created_time = db.Column(db.DateTime, default=datetime.utcnow)
    completed_time = db.Column(db.DateTime)
    preprocessing_config = db.Column(db.JSON)  # Feature engineering configuration
    
    # Relationships
    models = db.relationship('Model', backref='experiment', lazy=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'dataset_id': self.dataset_id,
            'model_type': self.model_type,
            'params': self.params,
            'metrics': self.metrics,
            'status': self.status,
            'created_time': self.created_time.isoformat() if self.created_time else None,
            'completed_time': self.completed_time.isoformat() if self.completed_time else None,
            'preprocessing_config': self.preprocessing_config
        }

class Model(db.Model):
    __tablename__ = 'models'
    
    id = db.Column(db.Integer, primary_key=True)
    experiment_id = db.Column(db.Integer, db.ForeignKey('experiments.id'), nullable=False)
    s3_path = db.Column(db.String(500), nullable=False)  # S3 path to model file
    status = db.Column(db.String(50), default='training')  # training, completed, deployed, error
    model_version = db.Column(db.String(50))
    created_time = db.Column(db.DateTime, default=datetime.utcnow)
    file_size = db.Column(db.BigInteger)
    
    def to_dict(self):
        return {
            'id': self.id,
            'experiment_id': self.experiment_id,
            's3_path': self.s3_path,
            'status': self.status,
            'model_version': self.model_version,
            'created_time': self.created_time.isoformat() if self.created_time else None,
            'file_size': self.file_size
        }

class Job(db.Model):
    __tablename__ = 'jobs'
    
    id = db.Column(db.String(255), primary_key=True)  # Celery task ID
    job_type = db.Column(db.String(50), nullable=False)  # eda, training, tuning, prediction
    status = db.Column(db.String(50), default='pending')  # pending, running, completed, failed
    progress = db.Column(db.Integer, default=0)  # Progress percentage
    result = db.Column(db.JSON)  # Job result data
    error_message = db.Column(db.Text)
    created_time = db.Column(db.DateTime, default=datetime.utcnow)
    started_time = db.Column(db.DateTime)
    completed_time = db.Column(db.DateTime)
    
    def to_dict(self):
        return {
            'id': self.id,
            'job_type': self.job_type,
            'status': self.status,
            'progress': self.progress,
            'result': self.result,
            'error_message': self.error_message,
            'created_time': self.created_time.isoformat() if self.created_time else None,
            'started_time': self.started_time.isoformat() if self.started_time else None,
            'completed_time': self.completed_time.isoformat() if self.completed_time else None
        }