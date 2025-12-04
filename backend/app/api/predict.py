from flask import Blueprint, request, jsonify, current_app
from app.models import Model, Experiment, Dataset, Job, db
from app.services.ml_service import MLService
from app.services.s3_service import S3Service
import uuid
import pandas as pd
import io

predict_bp = Blueprint('predict', __name__)

@predict_bp.route('/predict/<int:model_id>', methods=['POST'])
def predict_single(model_id):
    """Make single prediction"""
    try:
        model = Model.query.get_or_404(model_id)
        
        if model.status != 'completed':
            return jsonify({'error': 'Model is not ready for predictions'}), 400
        
        data = request.get_json()
        
        if 'features' not in data:
            return jsonify({'error': 'Missing features data'}), 400
        
        features = data['features']
        
        # Load model and make prediction
        ml_service = MLService()
        prediction_result = ml_service.predict_single(model, features)
        
        if prediction_result is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        return jsonify({
            'model_id': model_id,
            'prediction': prediction_result['prediction'],
            'probability': prediction_result.get('probability'),
            'confidence': prediction_result.get('confidence')
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Single prediction error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@predict_bp.route('/predict/<int:model_id>/batch', methods=['POST'])
def predict_batch(model_id):
    """Make batch predictions"""
    try:
        model = Model.query.get_or_404(model_id)
        
        if model.status != 'completed':
            return jsonify({'error': 'Model is not ready for predictions'}), 400
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read uploaded file
        try:
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.filename.endswith('.xlsx') or file.filename.endswith('.xls'):
                df = pd.read_excel(file)
            else:
                return jsonify({'error': 'Unsupported file format. Please use CSV or Excel.'}), 400
        except Exception as e:
            return jsonify({'error': f'Error reading file: {str(e)}'}), 400
        
        # Create prediction job
        job_id = str(uuid.uuid4())
        job = Job(
            id=job_id,
            job_type='prediction',
            status='pending'
        )
        
        db.session.add(job)
        db.session.commit()
        
        # Load model and make predictions
        ml_service = MLService()
        predictions = ml_service.predict_batch(model, df)
        
        if predictions is None:
            job.status = 'failed'
            job.error_message = 'Batch prediction failed'
            db.session.commit()
            return jsonify({'error': 'Batch prediction failed'}), 500
        
        # Add predictions to dataframe
        df['predictions'] = predictions['predictions']
        if 'probabilities' in predictions:
            df['probabilities'] = predictions['probabilities']
        
        # Save results to S3
        s3_service = S3Service()
        result_path = f"predictions/{job_id}_results.csv"
        
        # Convert dataframe to CSV buffer
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_bytes = io.BytesIO(csv_buffer.getvalue().encode('utf-8'))
        
        if not s3_service.upload_file(csv_bytes, result_path):
            job.status = 'failed'
            job.error_message = 'Failed to save results'
            db.session.commit()
            return jsonify({'error': 'Failed to save prediction results'}), 500
        
        # Update job status
        job.status = 'completed'
        job.result = {
            'predictions_count': len(predictions['predictions']),
            'result_path': result_path
        }
        db.session.commit()
        
        # Generate download URL
        download_url = s3_service.get_file_url(result_path)
        
        return jsonify({
            'message': 'Batch prediction completed',
            'job_id': job_id,
            'predictions_count': len(predictions['predictions']),
            'download_url': download_url
        }), 200
        
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@predict_bp.route('/models', methods=['GET'])
def list_models():
    """List available models for prediction"""
    try:
        experiment_id = request.args.get('experiment_id', type=int)
        
        query = Model.query.filter_by(status='completed')
        if experiment_id:
            query = query.filter_by(experiment_id=experiment_id)
        
        models = query.order_by(Model.created_time.desc()).all()
        
        # Include experiment and dataset info
        results = []
        for model in models:
            model_dict = model.to_dict()
            model_dict['experiment'] = model.experiment.to_dict()
            model_dict['dataset'] = model.experiment.dataset.to_dict()
            results.append(model_dict)
        
        return jsonify({
            'models': results
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"List models error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@predict_bp.route('/models/<int:model_id>/info', methods=['GET'])
def get_model_info(model_id):
    """Get detailed model information"""
    try:
        model = Model.query.get_or_404(model_id)
        
        ml_service = MLService()
        model_info = ml_service.get_model_info(model)
        
        return jsonify({
            'model': model.to_dict(),
            'experiment': model.experiment.to_dict(),
            'dataset': model.experiment.dataset.to_dict(),
            'model_details': model_info
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Get model info error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500