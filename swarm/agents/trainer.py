#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADAPTIVE MODEL TRAINER
Version: 3.0.0
Created: 2025-07-17
Author: AI Training Team
"""

import os
import logging
import numpy as np
import pandas as pd
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from datasets import Dataset, load_metric
from sklearn.model_selection import train_test_split
from .base_agent import BaseAgent

logger = logging.getLogger("ModelTrainer")

class ModelTrainer(BaseAgent):
    def __init__(self, primary_config, fallback_config, cost_optimizer):
        super().__init__(primary_config, fallback_config, cost_optimizer)
        self.base_model = "distilbert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self.model = None
        self.trained_models = {}
        
    async def train(self, dataset: pd.DataFrame, token_budget: int) -> dict:
        """Train model with adaptive curriculum learning"""
        logger.info("Starting model training")
        
        # Prepare dataset
        hf_dataset = self._prepare_dataset(dataset)
        
        # Split dataset
        train_val = hf_dataset.train_test_split(test_size=0.2, seed=42)
        train_dataset = train_val["train"]
        val_dataset = train_val["test"]
        
        # Initialize model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model, 
            num_labels=3  # Adjust based on your classification needs
        )
        
        # Training configuration
        training_args = TrainingArguments(
            output_dir="./model_checkpoints",
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            learning_rate=2e-5,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            report_to="none",  # Disable external reporting to save tokens
            logging_steps=50,
            save_total_limit=1,
        )
        
        # Create Trainer instance
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self._compute_metrics,
        )
        
        # Execute training
        try:
            train_result = trainer.train()
            metrics = train_result.metrics
            
            # Save model
            model_id = f"model_{int(time.time())}"
            save_path = f"./trained_models/{model_id}"
            trainer.save_model(save_path)
            self.trained_models[model_id] = save_path
            
            # Track token usage (estimate)
            tokens_used = len(train_dataset) * 100  # Approximate tokens per sample
            self._update_token_usage(tokens_used)
            
            logger.info(f"Training completed. Model saved to {save_path}")
            
            return {
                "model_id": model_id,
                "metrics": metrics,
                "tokens": tokens_used,
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "tokens": 0
            }
    
    def _prepare_dataset(self, df: pd.DataFrame) -> Dataset:
        """Prepare dataset for training"""
        # Preprocess data
        df["text"] = df["title"] + " " + df["content"]
        df = df.dropna(subset=["text"])
        
        # Tokenize
        tokenized = df["text"].apply(
            lambda x: self.tokenizer(
                x, 
                padding="max_length", 
                truncation=True, 
                max_length=256
            )
        )
        
        # Create dataset
        dataset = Dataset.from_pandas(df)
        dataset = dataset.map(
            lambda x: self.tokenizer(
                x["text"], 
                padding="max_length", 
                truncation=True, 
                max_length=256
            ),
            batched=True
        )
        
        return dataset
    
    def _compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        metric = load_metric("accuracy")
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    
    async def evaluate(self, model_id: str, test_dataset: Dataset) -> dict:
        """Evaluate model performance"""
        if model_id not in self.trained_models:
            raise ValueError(f"Model {model_id} not found")
        
        model_path = self.trained_models[model_id]
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        trainer = Trainer(
            model=model,
            compute_metrics=self._compute_metrics,
        )
        
        metrics = trainer.evaluate(test_dataset)
        
        # Track token usage (estimate)
        tokens_used = len(test_dataset) * 50  # Approximate tokens per sample
        self._update_token_usage(tokens_used)
        
        return {
            "metrics": metrics,
            "tokens": tokens_used
        }
    
    async def predict(self, model_id: str, text: str) -> dict:
        """Make prediction using trained model"""
        if model_id not in self.trained_models:
            raise ValueError(f"Model {model_id} not found")
        
        model_path = self.trained_models[model_id]
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=256
        )
        
        outputs = model(**inputs)
        probs = outputs.logits.softmax(dim=-1).detach().numpy()[0]
        prediction = probs.argmax()
        
        return {
            "prediction": int(prediction),
            "confidence": float(probs.max()),
            "probabilities": probs.tolist()
        }
