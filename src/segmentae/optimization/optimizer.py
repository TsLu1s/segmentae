import pandas as pd

from itertools import product
from typing import List, Optional, Any, Dict

from pydantic import BaseModel, field_serializer

from segmentae.core.constants import ClusterModel
from segmentae.core.exceptions import ValidationError, ConfigurationError
from segmentae.pipeline.segmentae import SegmentAE
from segmentae.clusters.clustering import Clustering


class OptimizerConfig(BaseModel):
    """Configuration for grid search optimizer."""
    
    n_clusters_list: List[int] = [1, 2, 3, 4]
    cluster_models: List[str] = ["KMeans", "MiniBatchKMeans", "GMM"]
    threshold_ratios: List[float] = [0.75, 1, 1.5, 2, 3, 4]
    performance_metric: str = 'f1_score'
    
    @field_serializer('cluster_models')
    def convert_cluster_models(cls, v):
        """Validate cluster models are valid strings."""
        valid_models = [m.value for m in ClusterModel]
        for model in v:
            if model not in valid_models:
                raise ValueError(
                    f"Invalid cluster model: '{model}'. "
                    f"Valid options: {valid_models}"
                )
        return v
    
    @field_serializer('n_clusters_list')
    def validate_n_clusters(cls, v):
        """Validate all n_clusters values are positive."""
        if any(n < 1 for n in v):
            raise ValueError("All n_clusters values must be >= 1")
        return v
    
    @field_serializer('threshold_ratios')
    def validate_ratios(cls, v):
        """Validate all threshold ratios are positive."""
        if any(r <= 0 for r in v):
            raise ValueError("All threshold_ratios must be positive")
        return v
    
    @field_serializer('performance_metric')
    def validate_metric(cls, v):
        """Validate performance metric name."""
        valid_metrics = [
            'accuracy', 'precision', 'recall', 'f1_score',
            'Accuracy', 'Precision', 'Recall', 'F1 Score'
        ]
        if v not in valid_metrics:
            raise ValueError(
                f"Invalid performance metric: '{v}'. "
                f"Valid options: {valid_metrics}"
            )
        return v


class SegmentAE_Optimizer:
    """
    Grid search optimizer for SegmentAE configurations.
    
    Systematically evaluates combinations of autoencoders, clustering algorithms,
    cluster numbers, and threshold ratios to identify optimal configuration for
    anomaly detection performance.
    """
    
    def __init__(self,
                 autoencoder_models: list,
                 n_clusters_list: List[int] = [1, 2, 3, 4],
                 cluster_models: List[str] = ["KMeans", "MiniBatchKMeans", "GMM"],
                 threshold_ratios: List[float] = [0.75, 1, 1.5, 2, 3, 4],
                 performance_metric: str = 'f1_score'):
        """
        Initialize grid search optimizer.
        """
        # Validate configuration
        self.config = OptimizerConfig(
            n_clusters_list=n_clusters_list,
            cluster_models=cluster_models,
            threshold_ratios=threshold_ratios,
            performance_metric=performance_metric
        )
        
        # Store autoencoder models
        self.autoencoder_models = autoencoder_models
        self.performance_metric = performance_metric
        
        # Validate autoencoders
        self._validate_autoencoders()
        
        # Results storage
        self.optimal_segmentae: Optional[SegmentAE] = None
        self.best_threshold_ratio: Optional[float] = None
        self.best_n_clusters: Optional[int] = None
        self.best_performance: float = float('-inf')
        self.leaderboard: Optional[pd.DataFrame] = None
    
    def optimize(self,
                 X_train: pd.DataFrame,
                 X_test: pd.DataFrame,
                 y_test: pd.Series) -> SegmentAE:
        """
        Execute grid search optimization.
        
        Evaluates all combinations of autoencoders, clustering algorithms,
        cluster numbers, and threshold ratios to find optimal configuration.
        """
        self._validate_inputs(X_train, X_test, y_test)
        
        results = []
        iteration = 1
        
        # Calculate total configurations
        total_configs = (
            len(self.config.n_clusters_list) *
            len(self.config.cluster_models) *
            len(self.autoencoder_models)
        )
        
        print(f"\n{'='*60}")
        print("Starting Grid Search Optimization")
        print(f"Total Configurations: {total_configs}")
        print(f"Performance Metric: {self.performance_metric}")
        print(f"{'='*60}\n")
        
        # Grid search over all combinations
        for n_clusters, cluster_model, autoencoder in product(
            self.config.n_clusters_list,
            self.config.cluster_models,
            self.autoencoder_models
        ):
            print(f"Iteration {iteration}/{total_configs}")
            print(f"Cluster Model: {cluster_model}")
            print(f"Number of Clusters: {n_clusters}")
            print(f"Autoencoder: {type(autoencoder).__name__}")
            print("")
            
            # Evaluate configuration
            config_results = self._evaluate_configuration(
                autoencoder=autoencoder,
                cluster_model=cluster_model,
                n_clusters=n_clusters,
                X_train=X_train,
                X_test=X_test,
                y_test=y_test
            )
            
            results.extend(config_results)
            iteration += 1
        
        # Create leaderboard
        self.leaderboard = self._create_leaderboard(results)
        self._print_optimization_summary()
        
        return self.optimal_segmentae
    
    # Private methods
    
    def _evaluate_configuration(self,
                               autoencoder: Any,
                               cluster_model: str,
                               n_clusters: int,
                               X_train: pd.DataFrame,
                               X_test: pd.DataFrame,
                               y_test: pd.Series) -> List[Dict]:
        """Evaluate a single configuration across all threshold ratios."""
        # Create and fit clustering
        clustering = Clustering(
            cluster_model=[cluster_model],
            n_clusters=n_clusters
        )
        clustering.clustering_fit(X=X_train)
        
        # Create SegmentAE and fit reconstruction
        sg = SegmentAE(ae_model=autoencoder, cl_model=clustering)
        sg.reconstruction(input_data=X_train, threshold_metric='mse')
        
        # Evaluate across threshold ratios
        config_results = []
        for threshold_ratio in self.config.threshold_ratios:
            result = self._evaluate_single_threshold(
                sg=sg,
                autoencoder=autoencoder,
                cluster_model=cluster_model,
                n_clusters=n_clusters,
                threshold_ratio=threshold_ratio,
                X_test=X_test,
                y_test=y_test
            )
            config_results.append(result)
        
        return config_results
    
    def _evaluate_single_threshold(self,
                                  sg: SegmentAE,
                                  autoencoder: Any,
                                  cluster_model: str,
                                  n_clusters: int,
                                  threshold_ratio: float,
                                  X_test: pd.DataFrame,
                                  y_test: pd.Series) -> Dict:
        """Evaluate a single threshold ratio."""
        # Run evaluation
        evaluation = sg.evaluation(
            input_data=X_test,
            target_col=y_test,
            threshold_ratio=threshold_ratio
        )
        
        global_metrics = evaluation["global metrics"].copy()
        
        # Extract performance score
        performance = self._extract_performance_score(global_metrics)
        
        # Update best model if necessary
        if performance > self.best_performance:
            self.best_performance = performance
            self.optimal_segmentae = sg
            self.best_threshold_ratio = threshold_ratio
            self.best_n_clusters = n_clusters
        
        # Add configuration info to metrics
        global_metrics["Autoencoder"] = type(autoencoder).__name__
        global_metrics["Cluster"] = cluster_model
        global_metrics["N_Clusters"] = n_clusters
        
        return global_metrics
    
    def _extract_performance_score(self, metrics: pd.DataFrame) -> float:
        """Extract performance score from metrics DataFrame."""
        metric_name = self.performance_metric
        
        # Try exact match first
        if metric_name in metrics.columns:
            return metrics[metric_name].iloc[0]
        
        # Try case-insensitive match
        for col in metrics.columns:
            if col.lower() == metric_name.lower():
                return metrics[col].iloc[0]
        
        # Try common variations
        metric_map = {
            'f1_score': 'F1 Score',
            'accuracy': 'Accuracy',
            'precision': 'Precision',
            'recall': 'Recall'
        }
        
        if metric_name in metric_map:
            mapped_name = metric_map[metric_name]
            if mapped_name in metrics.columns:
                return metrics[mapped_name].iloc[0]
        
        raise ConfigurationError(
            f"Performance metric '{metric_name}' not found in results. "
            f"Available metrics: {list(metrics.columns)}"
        )
    
    def _create_leaderboard(self, results: List[Dict]) -> pd.DataFrame:
        """Create sorted leaderboard from results."""
        leaderboard = pd.concat(results, axis=0, ignore_index=True)
        
        # Find the correct column name for sorting
        sort_column = self._find_sort_column(leaderboard)
        
        return leaderboard.sort_values(
            by=sort_column,
            ascending=False
        ).reset_index(drop=True)
    
    def _find_sort_column(self, df: pd.DataFrame) -> str:
        """Find the correct column name for the performance metric."""
        metric_name = self.performance_metric
        
        # Try exact match
        if metric_name in df.columns:
            return metric_name
        
        # Try case-insensitive match
        for col in df.columns:
            if col.lower() == metric_name.lower():
                return col
        
        # Try common variations
        metric_map = {
            'f1_score': 'F1 Score',
            'accuracy': 'Accuracy',
            'precision': 'Precision',
            'recall': 'Recall'
        }
        
        if metric_name in metric_map and metric_map[metric_name] in df.columns:
            return metric_map[metric_name]
        
        return metric_name  # Fall back to original
    
    def _print_optimization_summary(self) -> None:
        """Print optimization summary."""
        print(f"\n{'='*60}")
        print("OPTIMIZATION COMPLETE")
        print(f"{'='*60}")
        print(f"Best Performance ({self.performance_metric}): {round(self.best_performance, 6)}")
        
        if len(self.autoencoder_models) > 1:
            best_ae = type(self.optimal_segmentae.ae_model).__name__
            print(f"Best Autoencoder: {best_ae}")
        
        if len(self.config.cluster_models) > 1:
            print(f"Best Cluster Model: {self.optimal_segmentae.cl_model.cluster_model[0]}")
        
        if len(self.config.n_clusters_list) > 1:
            print(f"Best Number of Clusters: {self.best_n_clusters}")
        
        if len(self.config.threshold_ratios) > 1:
            print(f"Best Threshold Ratio: {self.best_threshold_ratio}")
        
        print(f"{'='*60}\n")
    
    # Validation methods
    
    def _validate_autoencoders(self) -> None:
        """Validate autoencoder models."""
        if not self.autoencoder_models:
            raise ConfigurationError(
                "autoencoder_models list cannot be empty",
                valid_options=["Provide at least one trained autoencoder"]
            )
        
        for ae in self.autoencoder_models:
            if not hasattr(ae, 'predict'):
                raise ConfigurationError(
                    f"Autoencoder {type(ae).__name__} must have a 'predict' method. "
                    f"Ensure all autoencoders are properly trained."
                )
    
    def _validate_inputs(self,
                        X_train: pd.DataFrame,
                        X_test: pd.DataFrame,
                        y_test: pd.Series) -> None:
        """Validate optimization inputs."""
        if not isinstance(X_train, pd.DataFrame):
            raise ValidationError(
                f"X_train must be a pandas DataFrame, got {type(X_train).__name__}"
            )
        
        if not isinstance(X_test, pd.DataFrame):
            raise ValidationError(
                f"X_test must be a pandas DataFrame, got {type(X_test).__name__}"
            )
        
        if not isinstance(y_test, pd.Series):
            raise ValidationError(
                f"y_test must be a pandas Series, got {type(y_test).__name__}"
            )
        
        if len(X_test) != len(y_test):
            raise ValidationError(
                f"X_test length ({len(X_test)}) must match "
                f"y_test length ({len(y_test)})"
            )
    
    def __repr__(self) -> str:
        """String representation of optimizer."""
        return (
            f"SegmentAE_Optimizer("
            f"n_autoencoders={len(self.autoencoder_models)}, "
            f"n_clusters={self.config.n_clusters_list}, "
            f"clusters={self.config.cluster_models}, "
            f"metric={self.performance_metric})"
        )