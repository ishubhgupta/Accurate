from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import os
import pandas as pd
from app.models import Dataset, db
from app.services.s3_service import S3Service
import uuid
from datetime import datetime

upload_bp = Blueprint('upload', __name__)

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@upload_bp.route('/upload', methods=['POST'])
def upload_dataset():
    """Upload dataset to S3 and register in MySQL"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed. Please upload CSV or Excel files.'}), 400
        
        # Validate file size (max 100MB)
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        
        if file_size > 100 * 1024 * 1024:  # 100MB limit
            return jsonify({'error': 'File too large. Maximum size is 100MB.'}), 400
        
        # Generate unique filename
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        
        # Read and validate the file
        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(file)
                file.seek(0)  # Reset file pointer for S3 upload
            else:  # Excel files
                df = pd.read_excel(file)
                file.seek(0)  # Reset file pointer for S3 upload
        except Exception as e:
            return jsonify({'error': f'Error reading file: {str(e)}'}), 400
        
        if df.empty:
            return jsonify({'error': 'File is empty'}), 400
        
        # Upload to S3
        s3_service = S3Service()
        s3_path = f"datasets/{unique_filename}"
        
        if not s3_service.upload_file(file, s3_path):
            return jsonify({'error': 'Failed to upload file to S3'}), 500
        
        # Store metadata in database
        dataset = Dataset(
            name=filename,
            path=s3_path,
            size=file_size,
            file_type=filename.rsplit('.', 1)[1].lower(),
            columns=df.columns.tolist(),
            rows=len(df),
            status='uploaded'
        )
        
        db.session.add(dataset)
        db.session.commit()
        
        return jsonify({
            'message': 'File uploaded successfully',
            'dataset': dataset.to_dict()
        }), 201
        
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@upload_bp.route('/datasets', methods=['GET'])
def list_datasets():
    """List all uploaded datasets"""
    try:
        datasets = Dataset.query.order_by(Dataset.upload_time.desc()).all()
        return jsonify({
            'datasets': [dataset.to_dict() for dataset in datasets]
        }), 200
    except Exception as e:
        current_app.logger.error(f"List datasets error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@upload_bp.route('/datasets/<int:dataset_id>', methods=['GET'])
def get_dataset(dataset_id):
    """Get dataset details"""
    try:
        dataset = Dataset.query.get_or_404(dataset_id)
        return jsonify({'dataset': dataset.to_dict()}), 200
    except Exception as e:
        current_app.logger.error(f"Get dataset error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@upload_bp.route('/datasets/<int:dataset_id>', methods=['DELETE'])
def delete_dataset(dataset_id):
    """Delete dataset from S3 and database"""
    try:
        dataset = Dataset.query.get_or_404(dataset_id)
        
        # Delete from S3
        s3_service = S3Service()
        s3_service.delete_file(dataset.path)
        
        # Delete from database
        db.session.delete(dataset)
        db.session.commit()
        
        return jsonify({'message': 'Dataset deleted successfully'}), 200
        
    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Delete dataset error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500