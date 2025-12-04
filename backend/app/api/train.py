from flask import Blueprint, request, jsonify, current_app
from app.models import Dataset, Experiment, Job, db
from app.services.ml_service import MLService
from app.tasks import train_model
import uuid

train_bp = Blueprint('train', __name__)

@train_bp.route('/train', methods=['POST'])
def start_training():
    """Start model training"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['dataset_id', 'model_types', 'target_column']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        dataset_id = data['dataset_id']
        model_types = data['model_types']  # List of algorithms to train
        target_column = data['target_column']
        preprocessing_config = data.get('preprocessing_config', {})
        training_config = data.get('training_config', {})
        
        # Validate dataset exists
        dataset = Dataset.query.get_or_404(dataset_id)
        
        if dataset.status != 'uploaded':
            return jsonify({'error': 'Dataset is not ready for training'}), 400
        
        # Validate model types
        ml_service = MLService()
        supported_models = ml_service.get_supported_algorithms()
        
        invalid_models = [model for model in model_types if model not in supported_models]
        if invalid_models:
            return jsonify({
                'error': f'Unsupported model types: {invalid_models}',
                'supported_models': list(supported_models.keys())
            }), 400
        
        experiments = []
        
        # Create experiments for each model type
        for model_type in model_types:
            job_id = str(uuid.uuid4())
            
            # Create experiment record
            experiment = Experiment(
                dataset_id=dataset_id,
                model_type=model_type,
                params=training_config.get(model_type, {}),
                preprocessing_config=preprocessing_config,
                status='pending'
            )
            
            db.session.add(experiment)
            db.session.flush()  # Get the experiment ID
            
            # Create job record
            job = Job(
                id=job_id,
                job_type='training',
                status='pending'
            )
            
            db.session.add(job)
            
            experiments.append({
                'experiment_id': experiment.id,
                'job_id': job_id,
                'model_type': model_type
            })
        
        db.session.commit()
        
        # Start training tasks
        for exp in experiments:
            train_model.delay(
                exp['job_id'],
                exp['experiment_id'],
                target_column,
                preprocessing_config,
                training_config.get(exp['model_type'], {})
            )
        
        return jsonify({
            'message': 'Training started',
            'experiments': experiments
        }), 202
        
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Training start error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@train_bp.route('/experiments', methods=['GET'])
def list_experiments():
    """List all experiments"""
    try:
        dataset_id = request.args.get('dataset_id', type=int)
        
        query = Experiment.query
        if dataset_id:
            query = query.filter_by(dataset_id=dataset_id)
        
        experiments = query.order_by(Experiment.created_time.desc()).all()
        
        return jsonify({
            'experiments': [exp.to_dict() for exp in experiments]
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"List experiments error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@train_bp.route('/experiments/<int:experiment_id>', methods=['GET'])
def get_experiment(experiment_id):
    """Get experiment details"""
    try:
        experiment = Experiment.query.get_or_404(experiment_id)
        return jsonify({'experiment': experiment.to_dict()}), 200
        
    except Exception as e:
        current_app.logger.error(f"Get experiment error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@train_bp.route('/algorithms', methods=['GET'])
def get_supported_algorithms():
    """Get list of supported ML algorithms"""
    try:
        ml_service = MLService()
        algorithms = ml_service.get_supported_algorithms()
        
        return jsonify({
            'algorithms': algorithms
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Get algorithms error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@train_bp.route('/hyperparameter-tuning', methods=['POST'])
def start_hyperparameter_tuning():
    """Start hyperparameter tuning"""
    try:
        data = request.get_json()
        
        required_fields = ['experiment_id', 'tuning_method', 'param_grid']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        experiment_id = data['experiment_id']
        tuning_method = data['tuning_method']  # grid_search, random_search, bayesian
        param_grid = data['param_grid']
        tuning_config = data.get('tuning_config', {})
        
        # Validate experiment exists
        experiment = Experiment.query.get_or_404(experiment_id)
        
        if experiment.status != 'completed':
            return jsonify({'error': 'Base experiment must be completed first'}), 400
        
        # Create tuning job
        job_id = str(uuid.uuid4())
        job = Job(
            id=job_id,
            job_type='tuning',
            status='pending'
        )
        
        db.session.add(job)
        db.session.commit()
        
        # Start tuning task (to be implemented)
        # tune_hyperparameters.delay(job_id, experiment_id, tuning_method, param_grid, tuning_config)
        
        return jsonify({
            'message': 'Hyperparameter tuning started',
            'job_id': job_id
        }), 202
        
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Hyperparameter tuning error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500