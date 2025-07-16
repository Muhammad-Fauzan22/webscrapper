#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODEL EVALUATOR AGENT
Version: 2.0.0
Created: 2025-07-17
Author: Quality Assurance Team
"""

import logging
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from .base_agent import BaseAgent

logger = logging.getLogger("ModelEvaluator")

class ModelEvaluator(BaseAgent):
    def __init__(self, primary_config, fallback_config, cost_optimizer):
        super().__init__(primary_config, fallback_config, cost_optimizer)
        self.metrics = {}
        
    async def evaluate_model(self, y_true, y_pred, model_id: str = None) -> dict:
        """Comprehensive model evaluation"""
        try:
            # Calculate standard metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average="weighted")
            recall = recall_score(y_true, y_pred, average="weighted")
            f1 = f1_score(y_true, y_pred, average="weighted")
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Calculate class-wise metrics
            class_report = {}
            unique_classes = np.unique(np.concatenate((y_true, y_pred)))
            for cls in unique_classes:
                cls_precision = precision_score(
                    y_true, y_pred, labels=[cls], average=None, zero_division=0
                )[0]
                cls_recall = recall_score(
                    y_true, y_pred, labels=[cls], average=None, zero_division=0
                )[0]
                cls_f1 = f1_score(
                    y_true, y_pred, labels=[cls], average=None, zero_division=0
                )[0]
                
                class_report[int(cls)] = {
                    "precision": float(cls_precision),
                    "recall": float(cls_recall),
                    "f1": float(cls_f1)
                }
            
            # Create result object
            result = {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "confusion_matrix": cm.tolist(),
                "class_report": class_report
            }
            
            # Store for historical comparison
            if model_id:
                self.metrics[model_id] = result
                
            return result
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            return {
                "error": str(e)
            }
    
    async def compare_models(self, model_id1: str, model_id2: str) -> dict:
        """Compare performance of two models"""
        if model_id1 not in self.metrics or model_id2 not in self.metrics:
            raise ValueError("Metrics not available for one or both models")
            
        metrics1 = self.metrics[model_id1]
        metrics2 = self.metrics[model_id2]
        
        comparison = {
            "accuracy_diff": metrics1["accuracy"] - metrics2["accuracy"],
            "f1_diff": metrics1["f1"] - metrics2["f1"],
            "significant_improvement": self._is_significant_improvement(
                metrics1, metrics2
            )
        }
        
        return comparison
    
    def _is_significant_improvement(self, metrics1: dict, metrics2: dict) -> bool:
        """Determine if improvement is statistically significant"""
        # Simplified significance check
        accuracy_diff = abs(metrics1["accuracy"] - metrics2["accuracy"])
        f1_diff = abs(metrics1["f1"] - metrics2["f1"])
        
        # Consider improvement significant if > 2% on both metrics
        return accuracy_diff > 0.02 and f1_diff > 0.02
    
    async def data_quality_report(self, dataset: pd.DataFrame) -> dict:
        """Generate data quality report"""
        report = {
            "samples": len(dataset),
            "features": len(dataset.columns),
            "missing_values": dataset.isnull().sum().to_dict(),
            "duplicates": dataset.duplicated().sum(),
            "class_distribution": dataset["label"].value_counts().to_dict()
        }
        
        return report
