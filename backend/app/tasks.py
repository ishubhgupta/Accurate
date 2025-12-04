from celery import Celery
from app.models import Job, Dataset, Experiment, Model, db
from app.services.eda_service import EDAService
from app.services.ml_service import MLService
from app import create_app
from datetime import datetime

# Initialize Celery
celery = Celery(__name__)

@celery.task
def generate_eda_report(job_id, dataset_id):
    """Generate EDA report task"""
    app = create_app()
    
    with app.app_context():
        try:
            # Update job status
            job = Job.query.get(job_id)
            job.status = 'running'
            job.started_time = datetime.utcnow()
            job.progress = 10
            db.session.commit()
            
            # Load dataset
            dataset = Dataset.query.get(dataset_id)
            eda_service = EDAService()
            
            # Update progress
            job.progress = 30
            db.session.commit()
            
            # Generate report
            report_result = eda_service.generate_full_report(dataset, job_id)
            
            if report_result:
                # Update progress
                job.progress = 80
                db.session.commit()
                
                # Generate quick summary
                summary = eda_service.get_quick_summary(dataset)
                
                # Complete job
                job.status = 'completed'
                job.progress = 100
                job.completed_time = datetime.utcnow()
                job.result = {
                    'html_report_path': report_result['html_report_path'],
                    'json_summary_path': report_result['json_summary_path'],
                    'summary': summary
                }
                db.session.commit()
                
                return {'status': 'success', 'job_id': job_id}
            else:
                job.status = 'failed'
                job.error_message = 'Failed to generate EDA report'
                db.session.commit()
                return {'status': 'failed', 'job_id': job_id}
                
        except Exception as e:
            job.status = 'failed'
            job.error_message = str(e)
            db.session.commit()
            return {'status': 'failed', 'error': str(e), 'job_id': job_id}

@celery.task
def train_model(job_id, experiment_id, target_column, preprocessing_config, training_config):
    """Train model task"""
    app = create_app()
    
    with app.app_context():
        try:
            # Update job status
            job = Job.query.get(job_id)
            job.status = 'running'
            job.started_time = datetime.utcnow()
            job.progress = 10
            db.session.commit()
            
            # Get experiment and dataset
            experiment = Experiment.query.get(experiment_id)
            dataset = experiment.dataset
            
            # Update experiment status
            experiment.status = 'running'
            db.session.commit()
            
            # Update progress
            job.progress = 20
            db.session.commit()
            
            # Train model
            ml_service = MLService()
            training_result = ml_service.train_model(
                dataset, experiment, target_column, 
                preprocessing_config, training_config
            )
            
            if training_result:
                # Update progress
                job.progress = 80
                db.session.commit()
                
                # Create model record
                model = Model(
                    experiment_id=experiment_id,
                    s3_path=training_result['model_path'],
                    status='completed',
                    model_version='1.0'
                )
                db.session.add(model)
                
                # Update experiment
                experiment.status = 'completed'
                experiment.completed_time = datetime.utcnow()
                experiment.metrics = training_result['metrics']
                
                # Complete job
                job.status = 'completed'
                job.progress = 100
                job.completed_time = datetime.utcnow()
                job.result = {
                    'model_id': model.id,
                    'metrics': training_result['metrics'],
                    'feature_count': training_result['feature_count']
                }
                
                db.session.commit()
                
                return {'status': 'success', 'job_id': job_id, 'model_id': model.id}
            else:
                experiment.status = 'failed'
                job.status = 'failed'
                job.error_message = 'Model training failed'
                db.session.commit()
                return {'status': 'failed', 'job_id': job_id}
                
        except Exception as e:
            experiment.status = 'failed'
            job.status = 'failed'
            job.error_message = str(e)
            db.session.commit()
            return {'status': 'failed', 'error': str(e), 'job_id': job_id}

# Additional task for hyperparameter tuning (placeholder for future implementation)
@celery.task
def tune_hyperparameters(job_id, experiment_id, tuning_method, param_grid, tuning_config):
    """Hyperparameter tuning task"""
    app = create_app()
    
    with app.app_context():
        try:
            # Update job status
            job = Job.query.get(job_id)
            job.status = 'running'
            job.started_time = datetime.utcnow()
            job.progress = 10
            db.session.commit()
            
            # Implementation for hyperparameter tuning would go here
            # This is a placeholder for the detailed implementation
            
            # For now, mark as completed
            job.status = 'completed'
            job.progress = 100
            job.completed_time = datetime.utcnow()
            job.result = {'message': 'Hyperparameter tuning completed (placeholder)'}
            db.session.commit()
            
            return {'status': 'success', 'job_id': job_id}
            
        except Exception as e:
            job.status = 'failed'
            job.error_message = str(e)
            db.session.commit()
            return {'status': 'failed', 'error': str(e), 'job_id': job_id}