from flask import Blueprint

# Import all blueprints
from .upload import upload_bp
from .eda import eda_bp
from .train import train_bp
from .predict import predict_bp

__all__ = ['upload_bp', 'eda_bp', 'train_bp', 'predict_bp']