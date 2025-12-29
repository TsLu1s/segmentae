"""
SegmentAE: A Python Library for Anomaly Detection Optimization

SegmentAE enhances anomaly detection performance through the optimization of
reconstruction error by integrating clustering methods with tabular autoencoders.

Key Components:
    - Preprocessing: Data preparation with encoding, scaling, and imputation
    - Clustering: Multiple clustering algorithms (KMeans, GMM, Agglomerative)
    - SegmentAE: Main pipeline integrating autoencoders and clustering
    - Autoencoders: Dense, BatchNorm, and Ensemble implementations
    - Optimizer: Grid search for optimal configuration
"""

__version__ = "2.0.0"
__author__ = "Luís Fernando da Silva Santos"

# Core components
from segmentae.core import (
    PhaseType,
    ClusterModel,
    ThresholdMetric,
    EncoderType,
    ScalerType,
    ImputerType
)

# Preprocessing
from segmentae.processing.preprocessing import Preprocessing

# Clustering
from segmentae.clusters import Clustering

# Pipeline
from segmentae.pipeline import SegmentAE

# Optimization
from segmentae.optimization import SegmentAE_Optimizer

# Metrics
from segmentae.metrics import metrics_classification, metrics_regression

# Data Sources
from segmentae.data_sources import load_dataset

# Note: Autoencoders are kept in their original location
# They should be imported directly when available:
from segmentae.autoencoders.dense import DenseAutoencoder
from segmentae.autoencoders.batch_norm import BatchNormAutoencoder
from segmentae.autoencoders.ensemble import EnsembleAutoencoder

__all__ = [
    # Version info
    '__version__',
    '__author__',
    
    # Core enums
    'PhaseType',
    'ClusterModel',
    'ThresholdMetric',
    'EncoderType',
    'ScalerType',
    'ImputerType',
    
    # Main classes
    'Preprocessing',
    'Clustering',
    'SegmentAE',
    'SegmentAE_Optimizer',
    
    # Metrics
    'metrics_classification',
    'metrics_regression',
    
    # Data
    'load_dataset',
    
    # Autoencoders
    'DenseAutoencoder',
    'BatchNormAutoencoder',
    'EnsembleAutoencoder',
]