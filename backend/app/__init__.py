from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from celery import Celery
import os
from dotenv import load_dotenv

load_dotenv()

db = SQLAlchemy()
celery = Celery(__name__)

def create_app(config=None):
    app = Flask(__name__)
    
    # Configuration
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'mysql+pymysql://root:password@localhost/accurate_db')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['CELERY_BROKER_URL'] = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    app.config['CELERY_RESULT_BACKEND'] = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    
    # AWS Configuration
    app.config['AWS_ACCESS_KEY_ID'] = os.getenv('AWS_ACCESS_KEY_ID')
    app.config['AWS_SECRET_ACCESS_KEY'] = os.getenv('AWS_SECRET_ACCESS_KEY')
    app.config['AWS_S3_BUCKET'] = os.getenv('AWS_S3_BUCKET', 'accurate-ml-bucket')
    app.config['AWS_REGION'] = os.getenv('AWS_REGION', 'us-east-1')
    
    if config:
        app.config.update(config)
    
    # Initialize extensions
    db.init_app(app)
    CORS(app)
    
    # Configure Celery
    celery.conf.update(app.config)
    
    # Register blueprints
    from app.api import upload_bp, eda_bp, train_bp, predict_bp
    app.register_blueprint(upload_bp, url_prefix='/api')
    app.register_blueprint(eda_bp, url_prefix='/api')
    app.register_blueprint(train_bp, url_prefix='/api')
    app.register_blueprint(predict_bp, url_prefix='/api')
    
    return app