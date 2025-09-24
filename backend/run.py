from app import create_app, db

app = create_app()

@app.before_first_request
def create_tables():
    """Create database tables on first request"""
    db.create_all()

@app.route('/')
def index():
    return {
        'message': 'Accurate ML Backend API',
        'version': '1.0.0',
        'endpoints': {
            'upload': '/api/upload',
            'datasets': '/api/datasets',
            'eda': '/api/eda/<dataset_id>',
            'train': '/api/train',
            'predict': '/api/predict/<model_id>',
            'algorithms': '/api/algorithms'
        }
    }

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return {'status': 'healthy', 'timestamp': '2024-01-01T00:00:00Z'}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)