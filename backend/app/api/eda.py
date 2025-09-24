from flask import Blueprint, request, jsonify, current_app
from app.models import Dataset, Job, db
from app.services.eda_service import EDAService
from app.tasks import generate_eda_report
import uuid

eda_bp = Blueprint('eda', __name__)

@eda_bp.route('/eda/<int:dataset_id>', methods=['POST'])
def generate_eda(dataset_id):
    """Trigger EDA report generation for a dataset"""
    try:
        dataset = Dataset.query.get_or_404(dataset_id)
        
        if dataset.status != 'uploaded':
            return jsonify({'error': 'Dataset is not ready for EDA'}), 400
        
        # Create job record
        job_id = str(uuid.uuid4())
        job = Job(
            id=job_id,
            job_type='eda',
            status='pending'
        )
        
        db.session.add(job)
        db.session.commit()
        
        # Start async task
        generate_eda_report.delay(job_id, dataset_id)
        
        return jsonify({
            'message': 'EDA generation started',
            'job_id': job_id
        }), 202
        
    except Exception as e:
        current_app.logger.error(f"EDA generation error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@eda_bp.route('/eda/<int:dataset_id>/quick', methods=['GET'])
def quick_eda(dataset_id):
    """Get quick EDA summary without full report generation"""
    try:
        dataset = Dataset.query.get_or_404(dataset_id)
        eda_service = EDAService()
        
        summary = eda_service.get_quick_summary(dataset)
        
        return jsonify({
            'dataset_id': dataset_id,
            'summary': summary
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Quick EDA error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@eda_bp.route('/eda/<int:dataset_id>/report', methods=['GET'])
def get_eda_report(dataset_id):
    """Get the generated EDA report URL"""
    try:
        dataset = Dataset.query.get_or_404(dataset_id)
        eda_service = EDAService()
        
        report_url = eda_service.get_report_url(dataset)
        
        if not report_url:
            return jsonify({'error': 'EDA report not found. Please generate it first.'}), 404
        
        return jsonify({
            'dataset_id': dataset_id,
            'report_url': report_url
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Get EDA report error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@eda_bp.route('/jobs/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Get job status"""
    try:
        job = Job.query.get_or_404(job_id)
        return jsonify({'job': job.to_dict()}), 200
        
    except Exception as e:
        current_app.logger.error(f"Get job status error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500