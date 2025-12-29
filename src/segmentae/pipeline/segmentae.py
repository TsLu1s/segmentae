import pandas as pd

from typing import Dict, Optional, Union, Any

from sklearn.metrics import confusion_matrix

from segmentae.core.constants import (
    PhaseType,
    ThresholdMetric,
    get_metric_column_name,
)
from segmentae.core.exceptions import (
    ValidationError,
    ModelNotFittedError,
    ConfigurationError,
    AutoencoderError
)
from segmentae.clusters.clustering import Clustering
from segmentae.metrics.performance_metrics import metrics_classification
from segmentae.pipeline.reconstruction import (
    ReconstructionConfig,
    EvaluationConfig,
    compute_reconstruction_errors,
    compute_column_metrics,
    create_metrics_dataframe,
    compute_threshold,
    detect_anomalies
)

import warnings
warnings.filterwarnings("ignore", category=Warning)

class SegmentAE:
    """
    SegmentAE integrates autoencoders with clustering for anomaly detection.
    
    This class orchestrates the reconstruction, evaluation, and detection pipeline
    by combining autoencoder reconstruction errors with cluster-specific thresholding
    to optimize anomaly detection performance.
    
    The workflow consists of three phases:
    1. Reconstruction: Compute reconstruction errors on training data
    2. Evaluation: Test anomaly detection on labeled test data
    3. Detection: Predict anomalies on unlabeled data
    """
    
    def __init__(self, ae_model: Any, cl_model: Clustering):
        """
        Initialize SegmentAE pipeline.
        """
        # Validate inputs
        self._validate_autoencoder(ae_model)
        self._validate_clustering(cl_model)
        
        # Store models
        self.ae_model = ae_model
        self.cl_model = cl_model
        
        # State management
        self._phase: PhaseType = PhaseType.EVALUATION
        self._threshold: Optional[float] = None
        self._threshold_metric: Optional[ThresholdMetric] = None
        self._metric_column: Optional[str] = None
        
        # Results storage (for backward compatibility)
        self.preds_train: Dict[int, Dict] = {}
        self.preds_test: Dict[int, Dict] = {}
        self.reconstruction_eval: Dict[int, Dict] = {}
        self.reconstruction_test: Dict[int, Dict] = {}
        self.results: Dict = {}
        
        # Internal state
        self._is_reconstruction_fitted: bool = False
    
    def reconstruction(self,
                      input_data: pd.DataFrame,
                      target_col: Optional[pd.Series] = None,
                      threshold_metric: str = "mse") -> Union['SegmentAE', tuple]:
        """
        Reconstruct input data and compute reconstruction errors per cluster.
        
        This method segments data by cluster, generates autoencoder reconstructions,
        and computes reconstruction errors using the specified metric.
        """
        # Validate and configure
        self._validate_reconstruction_input(input_data, target_col)
        config = ReconstructionConfig(threshold_metric=threshold_metric)
        self._threshold_metric = config.threshold_metric
        self._metric_column = get_metric_column_name(self._threshold_metric)
        
        # Get cluster assignments
        cluster_predictions = self._get_cluster_predictions(input_data)
        
        # Process each cluster
        cluster_results = self._process_all_clusters(
            input_data=input_data,
            target_col=target_col,
            cluster_predictions=cluster_predictions
        )
        
        # Store results based on phase
        self._store_reconstruction_results(cluster_results)
        
        self._is_reconstruction_fitted = True
        return self._return_based_on_phase()
    
    def evaluation(self,
                   input_data: pd.DataFrame,
                   target_col: pd.Series,
                   threshold_ratio: float = 1.0) -> Dict[str, Any]: 
        """
        Evaluate anomaly detection performance on labeled test data.
        
        Computes cluster-specific thresholds and evaluates detection performance
        against ground truth labels, providing both global and cluster-level metrics.
        """
        self._validate_fitted_for_evaluation()
        self._validate_evaluation_input(input_data, target_col)
        
        config = EvaluationConfig(threshold_ratio=threshold_ratio) # Threshold Ratio Adjusts the anomaly detection sensitivity relative to baseline threshold
        
        # Set phase and run reconstruction
        self._phase = PhaseType.TESTING
        self.preds_test, self.reconstruction_test = self.reconstruction(
            input_data=input_data,
            target_col=target_col,
            threshold_metric=self._threshold_metric.value
        )
        
        # Evaluate each cluster
        cluster_results = self._evaluate_all_clusters(config.threshold_ratio)
        
        # Aggregate global results
        global_results = self._aggregate_evaluation_results(
            cluster_results,
            config.threshold_ratio
        )
        
        self.results = global_results
        self._phase = PhaseType.EVALUATION  # Reset phase
        
        return self.results
    
    def detections(self,
                   input_data: pd.DataFrame,
                   threshold_ratio: float = 1.0) -> pd.DataFrame:
        """
        Perform anomaly detection on unlabeled data.
        
        Uses trained cluster-specific thresholds to detect anomalies in new data
        without requiring ground truth labels.
        """
        self._validate_fitted_for_detection()
        self._validate_input(input_data, "Input data")
        
        # Set phase and run reconstruction
        self._phase = PhaseType.PREDICTION
        self.reconstruction(
            input_data=input_data,
            target_col=None,
            threshold_metric=self._threshold_metric.value
        )
        
        # Detect anomalies per cluster
        predictions = self._detect_anomalies_all_clusters(threshold_ratio)
        
        self._phase = PhaseType.EVALUATION  # Reset phase
        return predictions
    
    def _process_all_clusters(self,
                              input_data: pd.DataFrame,
                              target_col: Optional[pd.Series],
                              cluster_predictions: pd.DataFrame) -> Dict[int, Dict]:
        """Process reconstruction for all clusters."""
        cluster_model_name = self.cl_model.cluster_model[0]
        results = {}
        
        for cluster_id in cluster_predictions[cluster_model_name].unique():
            cluster_data = self._extract_cluster_data(
                input_data,
                target_col,
                cluster_predictions,
                cluster_model_name,
                cluster_id
            )
            
            # Generate reconstructions and compute errors
            reconstructions = self._reconstruct_cluster(cluster_data['X'])
            errors = self._compute_cluster_errors(
                cluster_data['X'],
                reconstructions
            )
            
            # Store cluster results
            results[cluster_id] = {
                "cluster": cluster_id,
                "real": cluster_data['X'],
                "y_true": cluster_data['y'],
                "predictions": reconstructions,
                "indexs": cluster_data['indices'],
                "errors": errors
            }
        
        return results
    
    def _extract_cluster_data(self,
                              input_data: pd.DataFrame,
                              target_col: Optional[pd.Series],
                              cluster_predictions: pd.DataFrame,
                              cluster_model_name: str,
                              cluster_id: int) -> Dict[str, Any]:
        """Extract data for a specific cluster."""
        cluster_indices = cluster_predictions.index[
            cluster_predictions[cluster_model_name] == cluster_id
        ].tolist()
        
        X_cluster = input_data.loc[cluster_indices]
        y_cluster = target_col.loc[cluster_indices] if target_col is not None else None
        
        return {
            'X': X_cluster,
            'y': y_cluster,
            'indices': cluster_indices
        }
    
    def _reconstruct_cluster(self, X_cluster: pd.DataFrame) -> pd.DataFrame:
        """Generate autoencoder reconstructions for cluster data."""
        try:
            predictions = self.ae_model.predict(X_cluster)
            return pd.DataFrame(predictions, columns=X_cluster.columns).astype(float)
        except Exception as e:
            raise AutoencoderError(
                f"Failed to generate reconstructions: {str(e)}"
            )
    
    def _compute_cluster_errors(self,
                                real_values: pd.DataFrame,
                                predictions: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Compute reconstruction errors for cluster."""
        real_np = real_values.values
        pred_np = predictions.values
        
        # Compute per-row errors
        mse, mae, rmse, max_err = compute_reconstruction_errors(real_np, pred_np)
        metrics_df = create_metrics_dataframe(mse, mae, rmse, max_err)
        
        # Compute per-column metrics
        col_metrics = compute_column_metrics(
            real_np,
            pred_np,
            list(real_values.columns),
            0  # cluster_id will be set by caller
        )
        
        return {
            'metrics': metrics_df,
            'column_metrics': col_metrics
        }
    
    def _evaluate_all_clusters(self, threshold_ratio: float) -> list:
        """Evaluate anomaly detection for all clusters."""
        cluster_results = []
        
        for cluster_id in self.reconstruction_eval.keys():
            if cluster_id not in self.reconstruction_test:
                print(f"Warning: Cluster {cluster_id} not found in test data")
                continue
            
            result = self._evaluate_single_cluster(cluster_id, threshold_ratio)
            cluster_results.append(result)
        
        return cluster_results
    
    def _evaluate_single_cluster(self,
                                 cluster_id: int,
                                 threshold_ratio: float) -> Dict[str, Any]:
        """Evaluate a single cluster."""
        # Compute threshold
        threshold = self._compute_cluster_threshold(cluster_id, threshold_ratio)
        
        # Get test data
        metrics_test = self.reconstruction_test[cluster_id]["metrics"]
        predictions = self.preds_test[cluster_id]["predictions"].copy()
        y_test = self.preds_test[cluster_id]["y_true"].reset_index(drop=True)
        indices = self.preds_test[cluster_id]["indexs"]
        
        # Classify anomalies
        predictions['Predicted Anomalies'] = detect_anomalies(
            metrics_test[self._metric_column],
            threshold
        )
        
        # Compute metrics
        cm = confusion_matrix(y_test, predictions['Predicted Anomalies'])
        metrics = metrics_classification(y_test, predictions['Predicted Anomalies'])
        metrics["N_Cluster"] = cluster_id
        metrics["Threshold Metric"] = self._threshold_metric.value.upper()
        metrics["Threshold Value"] = round(threshold, 6)
        
        return {
            "cluster_id": cluster_id,
            "metrics": metrics,
            "confusion_matrix": cm,
            "predictions": predictions,
            "indices": indices,
            "y_test": y_test
        }
    
    def _compute_cluster_threshold(self,
                                   cluster_id: int,
                                   threshold_ratio: float) -> float:
        """Compute reconstruction threshold for a cluster."""
        rec_errors = self.reconstruction_eval[cluster_id]["metrics"][self._metric_column]
        threshold = compute_threshold(rec_errors, threshold_ratio)
        print(f"Cluster {cluster_id} || Reconstruction Threshold: {round(threshold, 5)}")
        
        # Print empty line after last cluster
        if cluster_id == len(self.reconstruction_eval) - 1:
            print("")
        
        return threshold
    
    def _detect_anomalies_all_clusters(self,
                                       threshold_ratio: float) -> pd.DataFrame:
        """Detect anomalies across all clusters."""
        all_predictions = []
        
        for cluster_id in self.reconstruction_eval.keys():
            if cluster_id not in self.reconstruction_final:
                continue
            
            cluster_preds = self._detect_cluster_anomalies(
                cluster_id,
                threshold_ratio
            )
            all_predictions.append(cluster_preds)
        
        # Aggregate and sort by original index
        if not all_predictions:
            raise ValidationError("No cluster predictions available")
        
        final_predictions = pd.concat(all_predictions, ignore_index=True)
        final_predictions = final_predictions.sort_values(by='_index')
        final_predictions = final_predictions.reset_index(drop=True)
        
        return final_predictions.drop('_index', axis=1)
    
    def _detect_cluster_anomalies(self,
                                  cluster_id: int,
                                  threshold_ratio: float) -> pd.DataFrame:
        """Detect anomalies for a single cluster."""
        # Compute threshold (without printing)
        rec_errors = self.reconstruction_eval[cluster_id]["metrics"][self._metric_column]
        threshold = compute_threshold(rec_errors, threshold_ratio)
        
        # Get predictions
        recons_metrics = self.reconstruction_final[cluster_id]["metrics"]
        predictions = self.preds_final[cluster_id]["predictions"].copy()
        indices = self.preds_final[cluster_id]["indexs"]
        
        # Detect anomalies
        predictions['Predicted Anomalies'] = detect_anomalies(
            recons_metrics[self._metric_column],
            threshold
        )
        predictions['_index'] = indices
        
        return predictions
    
    def _aggregate_evaluation_results(self,
                                     cluster_results: list,
                                     threshold_ratio: float) -> Dict[str, Any]:
        """Aggregate cluster evaluation results into global metrics."""
        # Cluster-level metrics
        cluster_metrics = pd.concat(
            [result["metrics"] for result in cluster_results],
            ignore_index=True
        )
        
        # Confusion matrices
        confusion_matrices = {
            result["cluster_id"]: {
                f"CM_{result['cluster_id']}": result["confusion_matrix"]
            }
            for result in cluster_results
        }
        
        # Global predictions
        all_predictions = []
        for result in cluster_results:
            pred_df = pd.DataFrame({
                'index': result["indices"],
                'y_test': result["y_test"],
                'Predicted Anomalies': result["predictions"]['Predicted Anomalies']
            })
            all_predictions.append(pred_df)
        
        ytpred = pd.concat(all_predictions, ignore_index=True)
        ytpred = ytpred.sort_values(by='index').set_index('index')
        
        # Global metrics
        global_metrics = metrics_classification(
            ytpred['y_test'],
            ytpred['Predicted Anomalies']
        )
        global_metrics["Threshold Metric"] = self._threshold_metric.value.upper()
        global_metrics["Threshold Ratio"] = threshold_ratio
        
        return {
            "global metrics": global_metrics,
            "clusters metrics": cluster_metrics,
            "confusion matrix": confusion_matrices,
            "y_true vs y_pred": ytpred
        }
    
    # Storage and retrieval methods
    
    def _store_reconstruction_results(self, cluster_results: Dict) -> None:
        """Store reconstruction results based on current phase."""
        # Convert cluster results to metrics format
        metrics_results = {}
        preds_results = {}
        
        for cluster_id, result in cluster_results.items():
            metrics_results[cluster_id] = {
                "cluster": result["cluster"],
                "metrics": result["errors"]["metrics"],
                "column_metrics": result["errors"]["column_metrics"],
                "indexs": result["indexs"]
            }
            
            preds_results[cluster_id] = {
                "cluster": result["cluster"],
                "real": result["real"],
                "y_true": result["y_true"],
                "predictions": result["predictions"],
                "indexs": result["indexs"]
            }
        
        if self._phase == PhaseType.EVALUATION:
            self.preds_train = preds_results
            self.reconstruction_eval = metrics_results
        elif self._phase == PhaseType.TESTING:
            self.preds_test = preds_results
            self.reconstruction_test = metrics_results
        elif self._phase == PhaseType.PREDICTION:
            self.preds_final = preds_results
            self.reconstruction_final = metrics_results
    
    def _return_based_on_phase(self) -> Union['SegmentAE', tuple]:
        """Return appropriate results based on phase."""
        if self._phase == PhaseType.EVALUATION:
            return self
        elif self._phase == PhaseType.TESTING:
            return self.preds_test, self.reconstruction_test
        elif self._phase == PhaseType.PREDICTION:
            return self.preds_final, self.reconstruction_final
    
    # Helper methods
    
    def _get_cluster_predictions(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Get cluster assignments for input data."""
        return self.cl_model.cluster_prediction(X=input_data)
    
    def _validate_autoencoder(self, ae_model: Any) -> None:
        """Validate that autoencoder has required interface."""
        if not hasattr(ae_model, 'predict'):
            raise ConfigurationError(
                "Autoencoder must have a 'predict' method. "
                "Ensure you're passing a trained Keras model or built-in autoencoder."
            )
    
    def _validate_clustering(self, cl_model: Clustering) -> None:
        """Validate clustering model."""
        if not isinstance(cl_model, Clustering):
            raise ConfigurationError(
                "cl_model must be an instance of Clustering",
                valid_options=["Clustering"]
            )
        if not cl_model._is_fitted:
            raise ModelNotFittedError(
                component="Clustering",
                message="Clustering model must be fitted before use. "
                        "Call clustering_fit(X) method first."
            )
    
    def _validate_reconstruction_input(self,
                                      input_data: pd.DataFrame,
                                      target_col: Optional[pd.Series]) -> None:
        """Validate reconstruction inputs."""
        self._validate_input(input_data, "Input data")
        
        if target_col is not None:
            if not isinstance(target_col, pd.Series):
                raise ValidationError(
                    f"target_col must be a pandas Series, got {type(target_col).__name__}",
                    suggestion="Convert to Series using pd.Series() or use DataFrame column"
                )
            if len(target_col) != len(input_data):
                raise ValidationError(
                    f"target_col length ({len(target_col)}) must match "
                    f"input_data length ({len(input_data)})"
                )
    
    def _validate_evaluation_input(self,
                                  input_data: pd.DataFrame,
                                  target_col: pd.Series) -> None:
        """Validate evaluation inputs."""
        self._validate_input(input_data, "Input data")
        
        if not isinstance(target_col, pd.Series):
            raise ValidationError(
                f"target_col must be a pandas Series, got {type(target_col).__name__}",
                suggestion="Use test[target_column] to extract Series"
            )
        
        if len(target_col) != len(input_data):
            raise ValidationError(
                f"target_col length ({len(target_col)}) must match "
                f"input_data length ({len(input_data)})"
            )
    
    def _validate_input(self, X: pd.DataFrame, context: str = "Input") -> None:
        """Validate input DataFrame."""
        if not isinstance(X, pd.DataFrame):
            raise ValidationError(
                f"{context} must be a pandas DataFrame, got {type(X).__name__}",
                suggestion="Convert to DataFrame using pd.DataFrame()"
            )
        
        if X.empty:
            raise ValidationError(
                f"{context} DataFrame is empty",
                suggestion="Ensure your dataset contains data"
            )
    
    def _validate_fitted_for_evaluation(self) -> None:
        """Validate that reconstruction has been performed."""
        if not self._is_reconstruction_fitted:
            raise ModelNotFittedError(
                component="SegmentAE",
                message="Must call reconstruction() before evaluation(). "
                        "Example: sg.reconstruction(X_train, threshold_metric='mse')"
            )
    
    def _validate_fitted_for_detection(self) -> None:
        """Validate that reconstruction has been performed."""
        if not self._is_reconstruction_fitted:
            raise ModelNotFittedError(
                component="SegmentAE",
                message="Must call reconstruction() before detections(). "
                        "Example: sg.reconstruction(X_train, threshold_metric='mse')"
            )
    
    def __repr__(self) -> str:
        """String representation of SegmentAE."""
        ae_name = type(self.ae_model).__name__
        return (
            f"SegmentAE("
            f"autoencoder={ae_name}, "
            f"clustering={self.cl_model.cluster_model}, "
            f"fitted={self._is_reconstruction_fitted})"
        )