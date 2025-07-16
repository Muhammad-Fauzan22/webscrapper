#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SWARM ORCHESTRATOR CORE
Version: 3.0.0
Created: 2025-07-17
Author: Muhammad-Fauzan22
"""

import asyncio
import os
import time
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
from .agents import (
    MetaPlanner,
    Researcher,
    Scraper,
    DataCleaner,
    Healer,
    ModelTrainer,
    ModelEvaluator
)
from .storage import MongoDBManager, GoogleDriveManager, HFCacheManager
from .utils import Logger, RetryHandler, SecretManager, CostOptimizer
from .cloud import AutoScaler

# Initialize logger
logger = Logger(name="Orchestrator")

class Orchestrator:
    def __init__(self, topic="ASEAN Renewable Energy"):
        load_dotenv()  # Load environment variables
        
        # Initialize components
        self.topic = topic
        self.secret_manager = SecretManager()
        self.cost_optimizer = CostOptimizer()
        self.auto_scaler = AutoScaler()
        self.healer = Healer()
        
        # Initialize agents with token optimization
        self.meta_planner = self._init_agent(MetaPlanner, "CYPHER", "HF")
        self.researcher = self._init_agent(Researcher, "PERPLEXITY", "SERPAPI")
        self.scraper = self._init_agent(Scraper, "DEEPSEEK", "SCRAPEOPS")
        self.cleaner = self._init_agent(DataCleaner, "CLAUDE", "HF")
        self.trainer = self._init_agent(ModelTrainer, "GEMMA", "HF")
        
        # Initialize storage
        self.db = MongoDBManager()
        self.gdrive = GoogleDriveManager()
        self.hf_cache = HFCacheManager()
        
        # State management
        self.last_successful_run = None
        self.error_count = 0
        self.total_tokens_used = 0
        self.token_budget = int(os.getenv("TOKEN_BUDGET", 100000))  # Default 100K tokens/day
        
    def _init_agent(self, agent_class, primary_provider, fallback_provider):
        """Initialize agent with fallback strategy"""
        primary_config = self._get_provider_config(primary_provider)
        fallback_config = self._get_provider_config(fallback_provider)
        
        return agent_class(
            primary_config=primary_config,
            fallback_config=fallback_config,
            cost_optimizer=self.cost_optimizer
        )
    
    def _get_provider_config(self, provider):
        """Get provider configuration with decrypted keys"""
        provider = provider.upper()
        return {
            "endpoint": os.getenv(f"{provider}_ENDPOINT"),
            "model": os.getenv(f"{provider}_MODEL"),
            "key": self.secret_manager.decrypt(os.getenv(f"{provider}_KEY"))
        }
    
    async def run(self):
        """Main execution loop"""
        logger.info(f"Starting Swarm Orchestrator for topic: {self.topic}")
        
        while True:
            try:
                # Check token budget before starting
                if self.total_tokens_used >= self.token_budget:
                    logger.warning("Token budget exhausted. Pausing until reset.")
                    await asyncio.sleep(3600)  # Sleep for 1 hour
                    self.total_tokens_used = 0  # Reset counter
                    continue
                
                start_time = time.time()
                
                # Execute scraping pipeline
                await self.execute_pipeline()
                
                # Run periodic tasks
                await self.run_periodic_tasks()
                
                # Update successful run timestamp
                self.last_successful_run = datetime.now()
                self.error_count = 0
                
                duration = time.time() - start_time
                logger.info(f"Cycle completed in {duration:.2f} seconds. Tokens used: {self.total_tokens_used}/{self.token_budget}")
                
                # Sleep until next cycle (default 6 hours)
                await asyncio.sleep(21600)
                
            except Exception as e:
                self.error_count += 1
                logger.error(f"Critical error in main loop: {str(e)}")
                await self.handle_critical_error(e)
    
    async def execute_pipeline(self):
        """Execute the full scraping pipeline"""
        # Step 1: Planning
        plan_result = await self.execute_with_retry(
            self.meta_planner.generate_scrape_plan,
            self.topic
        )
        self._update_token_usage(plan_result.get("tokens", 0))
        
        # Step 2: Research
        research_result = await self.execute_with_retry(
            self.researcher.research_topic,
            self.topic
        )
        self._update_token_usage(research_result.get("tokens", 0))
        
        # Step 3: Scraping
        scrape_results = []
        for url in plan_result["targets"]:
            result = await self.execute_with_retry(
                self.scraper.scrape_url,
                url
            )
            scrape_results.append(result)
            self._update_token_usage(result.get("tokens", 0))
        
        # Step 4: Cleaning
        cleaned_data = []
        for data in scrape_results:
            result = await self.execute_with_retry(
                self.cleaner.clean,
                data["content"]
            )
            cleaned_data.append(result)
            self._update_token_usage(result.get("tokens", 0))
        
        # Step 5: Storage
        storage_result = await self.execute_with_retry(
            self.store_results,
            cleaned_data
        )
        
        return {
            "plan": plan_result,
            "research": research_result,
            "scrape": scrape_results,
            "cleaned": cleaned_data,
            "storage": storage_result
        }
    
    async def execute_with_retry(self, func, *args, **kwargs):
        """Execute function with smart retry mechanism"""
        retry_handler = RetryHandler(
            max_retries=3,
            backoff_factor=2,
            cost_optimizer=self.cost_optimizer
        )
        
        return await retry_handler.execute(
            func, 
            *args, 
            **kwargs,
            token_budget_remaining=self.token_budget - self.total_tokens_used
        )
    
    async def run_periodic_tasks(self):
        """Execute periodic maintenance tasks"""
        now = datetime.now()
        
        # Daily tasks (run once per day)
        if not self.last_successful_run or (now - self.last_successful_run) >= timedelta(days=1):
            logger.info("Running daily tasks")
            await self.backup_data()
            await self.cleanup_cache()
            
            # Weekly tasks (run on Sundays)
            if now.weekday() == 6:  # Sunday
                logger.info("Running weekly tasks")
                await self.train_model()
                await self.evaluate_model()
    
    async def backup_data(self):
        """Backup data to Google Drive"""
        logger.info("Starting data backup to Google Drive")
        try:
            await self.gdrive.backup_database(
                db_manager=self.db,
                folder_id=os.getenv("GOOGLE_DRIVE_FOLDER_ID")
            )
            logger.info("Data backup completed successfully")
        except Exception as e:
            logger.error(f"Backup failed: {str(e)}")
    
    async def cleanup_cache(self):
        """Clean up cache storage"""
        logger.info("Cleaning up cache")
        try:
            await self.hf_cache.cleanup(
                max_age_days=7,
                max_size_gb=1
            )
            logger.info("Cache cleanup completed")
        except Exception as e:
            logger.error(f"Cache cleanup failed: {str(e)}")
    
    async def train_model(self):
        """Train and update model"""
        logger.info("Starting model training")
        try:
            # Get recent data for training
            training_data = await self.db.get_recent_data(
                collection=self.topic,
                days=30
            )
            
            # Train model
            training_result = await self.trainer.train(
                dataset=training_data,
                token_budget=self.token_budget - self.total_tokens_used
            )
            self._update_token_usage(training_result.get("tokens", 0))
            
            logger.info(f"Model training completed: {training_result['model_id']}")
            return training_result
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            return None
    
    async def evaluate_model(self):
        """Evaluate model performance"""
        logger.info("Evaluating model performance")
        try:
            # Get validation data
            validation_data = await self.db.get_validation_data(
                collection=self.topic,
                sample_size=100
            )
            
            # Evaluate model
            evaluation_result = await self.trainer.evaluate(
                dataset=validation_data,
                token_budget=self.token_budget - self.total_tokens_used
            )
            self._update_token_usage(evaluation_result.get("tokens", 0))
            
            logger.info(f"Model evaluation completed: Accuracy={evaluation_result['accuracy']:.2f}")
            return evaluation_result
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            return None
    
    async def store_results(self, data):
        """Store results in database"""
        logger.info(f"Storing {len(data)} records to database")
        try:
            await self.db.connect()
            result = await self.db.bulk_insert(
                collection=self.topic,
                documents=data
            )
            logger.info(f"Storage completed: {result['inserted_count']} documents inserted")
            return result
        except Exception as e:
            logger.error(f"Storage failed: {str(e)}")
            raise
    
    async def handle_critical_error(self, error):
        """Handle critical errors in the system"""
        logger.critical("Initiating critical error recovery sequence")
        
        # Step 1: Attempt self-healing
        healing_result = await self.healer.diagnose_and_heal(error)
        
        if healing_result["success"]:
            logger.info("Self-healing successful. Resuming operations.")
            return
        
        # Step 2: Scale resources if possible
        logger.warning("Self-healing failed. Attempting to scale resources.")
        scale_result = await self.auto_scaler.scale_up()
        
        if scale_result["success"]:
            logger.info("Resource scaling successful. Retrying operation.")
            return
        
        # Step 3: Final fallback - restart container
        logger.error("All recovery attempts failed. Triggering full restart.")
        await self.trigger_restart()
    
    async def trigger_restart(self):
        """Trigger a full system restart via cloud provider"""
        logger.critical("Initiating system restart")
        try:
            # Implement cloud-specific restart logic
            if os.getenv("CLOUD_PROVIDER") == "AZURE":
                from .cloud import AzureDeployer
                deployer = AzureDeployer()
                await deployer.restart_container()
            else:
                # Generic restart fallback
                os._exit(1)  # Force exit with error code
        except:
            os._exit(1)
    
    def _update_token_usage(self, tokens):
        """Update token usage and check budget"""
        if not tokens:
            return
            
        self.total_tokens_used += tokens
        self.cost_optimizer.track_usage(tokens)
        
        # Log if approaching budget limit
        if self.total_tokens_used > self.token_budget * 0.8:
            logger.warning(
                f"Token usage approaching budget: {self.total_tokens_used}/{self.token_budget}"
            )
    
    def get_status(self):
        """Get current system status"""
        return {
            "topic": self.topic,
            "last_successful_run": self.last_successful_run.isoformat() if self.last_successful_run else None,
            "error_count": self.error_count,
            "token_usage": f"{self.total_tokens_used}/{self.token_budget}",
            "cost_today": self.cost_optimizer.get_daily_cost(),
            "components": {
                "meta_planner": self.meta_planner.get_status(),
                "researcher": self.researcher.get_status(),
                "scraper": self.scraper.get_status(),
                "cleaner": self.cleaner.get_status(),
                "trainer": self.trainer.get_status()
            }
        }

# Entry point for standalone execution
if __name__ == "__main__":
    # Create and run orchestrator
    orchestrator = Orchestrator(topic="ASEAN Renewable Energy Trends")
    
    try:
        asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        logger.info("Orchestrator shutdown by user")
    except Exception as e:
        logger.critical(f"Unrecoverable error: {str(e)}")
        raise
