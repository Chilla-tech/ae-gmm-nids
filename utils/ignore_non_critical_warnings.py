
import warnings
import tensorflow as tf
import os

def suppress_non_critical_warnings():
    """Configure environment to suppress non-critical warnings for production"""
    
    # Suppress specific TensorFlow optimizer warnings
    warnings.filterwarnings('ignore', message='.*Skipping variable loading for optimizer.*')
    warnings.filterwarnings('ignore', message='.*optimizer.*variables.*')
    
    # Set TensorFlow logging to only show errors
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF info/warning messages