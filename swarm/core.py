#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CORE SYSTEM - Autonomous Web Scraping Orchestrator
Version: 1.5.2
Created: 2025-07-15
Author: AI Master Programmer
"""

import os
import asyncio
import aiohttp
import json
import logging
import platform
import psutil
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

# Custom module imports
from swarm.agents.meta_planner import MetaPlanner
from swarm.agents.scraper import Scraper
from swarm.agents.healer import Healer
from swarm.agents.cleaner import DataCleaner
from swarm.storage.gdrive import DriveStorage
from swarm.storage.mongodb import MongoDBStorage
from swarm.util.performance import SystemMonitor

# Configure advanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-16s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("SwarmCore")
logger.setLevel(logging.DEBUG if os.getenv("DEBUG_MODE") == "1" else logging.INFO)

# Load AI configuration
def load_ai_config() -> Dict[str, Any]:
    """Load AI configuration with error handling and validation"""
    try:
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            '..', 
            'configs', 
            'ai_config.yaml'
        )
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"AI config not found at {config_path}")
        
        with open(config_path) as f:
            import yaml
            config = yaml.safe_load(f)
            
            # Validate essential keys
            required_services = ["DEEPSEEK", "CLAUDE", "PERPLEXITY", "CYPHER"]
            for service in required_services:
                if service not in config:
                    raise ValueError(f"Missing {service} configuration")
                if "key" not in config[service] or not config[service]["key"]:
                    raise ValueError(f"Invalid API key for {service}")
            
            return config
    except Exception as e:
        logger.critical(f"AI config load failed: {str(e)}")
        # Attempt to load from environment as fallback
        return {
            "DEEPSEEK": {
                "endpoint": os.getenv("DEEPSEEK_ENDPOINT"),
                "model": os.getenv("DEEPSEEK_MODEL"),
                "key": os.getenv("DEEPSEEK_KEY")
            },
            "CLAUDE": {
                "endpoint": os.getenv("CLAUDE_ENDPOINT"),
                "model": os.getenv("CLAUDE_MODEL"),
                "key": os.getenv("CLAUDE_KEY")
            },
            "PERPLEXITY": {
                "endpoint": os.getenv("PERPLEXITY_ENDPOINT"),
                "model": os.getenv("PERPLEXITY_MODEL"),
                "key": os.getenv("PERPLEXITY_KEY")
            },
            "CYPHER": {
                "endpoint": os.getenv("CYPHER_ENDPOINT"),
                "model": os.getenv("CYPHER_MODEL"),
                "key": os.getenv("CYPHER_KEY")
            }
        }

class SuperSwarm:
    def __init__(self, topic: str = "renewable energy in ASEAN"):
        """
        Initialize autonomous scraping swarm
        
        Args:
            topic: Main research topic for the scraping session
        """
        self.topic = topic
        self.ai_config = load_ai_config()
        self.storage = DriveStorage()  # Primary storage
        self.backup_storage = MongoDBStorage()  # Secondary storage
        self.monitor = SystemMonitor()
        self.cycle_count = 0
        self.consecutive_failures = 0
        self.start_time = datetime.now()
        self.last_cycle_time = None
        self.status = "INITIALIZING"
        self.active_workers = 4  # Default worker count
        self.max_workers = 8
        self.min_workers = 2
        self.scaling_threshold = 70  CPU usage percentage
        
        logger.info(f"🌀 Swarm Initialized | Topic: {self.topic}")
        logger.info(f"System: {platform.system()} {platform.release()}")
        logger.info(f"Python: {platform.python_version()}")
        logger.info(f"Processor: {platform.processor()}")
        logger.info(f"AI Models: {', '.join(self.ai_config.keys())}")
        
    async def run_cycle(self) -> bool:
        """
        Execute a full scraping cycle with self-healing capabilities
        
        Returns:
            bool: True if cycle completed successfully, False otherwise
        """
        self.cycle_count += 1
        cycle_start = datetime.now()
        self.status = f"CYCLE_{self.cycle_count}_RUNNING"
        
        logger.info(f"\n{'='*50}")
        logger.info(f"🚀 CYCLE {self.cycle_count} STARTED | Topic: {self.topic}")
        logger.info(f"⏱️  Start Time: {cycle_start.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"👷 Active Workers: {self.active_workers}")
        
        try:
            # Phase 1: System Health Check
            if not await self._health_check():
                logger.warning("System health check failed! Initiating self-healing...")
                await self._self_heal("SYSTEM_HEALTH_CHECK_FAILURE")
                return False

            # Phase 2: AI-Powered Target Planning
            logger.info("🧠 Phase 1: AI Target Planning")
            targets = await self._plan_targets()
            
            if not targets:
                logger.error("Target planning failed! No targets generated")
                return False
                
            logger.info(f"🎯 Targets Identified: {len(targets)} URLs")

            # Phase 3: Parallel Scraping Execution
            logger.info("⚡ Phase 2: Parallel Scraping")
            results = await self._execute_scraping(targets)
            
            successes = [r for r in results if r.get('status') == 'SUCCESS']
            failures = [r for r in results if r.get('status') == 'FAILURE']
            
            logger.info(f"📊 Results: {len(successes)} ✅ | {len(failures)} ❌")

            # Phase 4: Data Processing
            logger.info("🧹 Phase 3: Data Processing")
            processed_data = await self._process_data(successes)
            
            # Phase 5: Storage
            logger.info("💾 Phase 4: Data Storage")
            storage_results = await self._store_data(processed_data)
            
            # Phase 6: Failure Analysis and Healing
            if failures:
                logger.warning(f"⚠️ Failures detected: {len(failures)}")
                await self._analyze_failures(failures)
            
            # Phase 7: Performance Optimization
            await self._optimize_performance()
            
            # Update status and metrics
            self.status = "IDLE"
            cycle_duration = datetime.now() - cycle_start
            self.last_cycle_time = cycle_duration.total_seconds()
            
            logger.info(f"✅ CYCLE COMPLETED | Duration: {str(cycle_duration)}")
            self.consecutive_failures = 0  # Reset failure counter
            return True
            
        except Exception as e:
            self.consecutive_failures += 1
            self.status = f"ERROR_CYCLE_{self.cycle_count}"
            error_id = f"ERR-{time.strftime('%Y%m%d-%H%M%S')}"
            
            logger.critical(f"🚨 CYCLE FAILURE [{error_id}] | Error: {str(e)}")
            logger.exception("Stack trace:")
            
            # Emergency healing for critical failures
            if self.consecutive_failures >= 3:
                logger.error("‼️ CONSECUTIVE FAILURES DETECTED! INITIATING EMERGENCY HEALING")
                await self._emergency_heal(e, error_id)
                
            return False

    async def _health_check(self) -> bool:
        """Perform comprehensive system health check"""
        logger.info("🩺 Running System Health Check")
        
        # Resource monitoring
        cpu_usage = self.monitor.cpu_usage()
        mem_usage = self.monitor.memory_usage()
        disk_usage = self.monitor.disk_usage()
        
        logger.info(f"  CPU Usage: {cpu_usage}%")
        logger.info(f"  Memory Usage: {mem_usage}%")
        logger.info(f"  Disk Usage: {disk_usage}%")
        
        # Threshold checks
        if cpu_usage > 90:
            logger.warning("  High CPU usage detected!")
            return False
        if mem_usage > 85:
            logger.warning("  High memory usage detected!")
            return False
        if disk_usage > 90:
            logger.warning("  High disk usage detected!")
            return False
            
        # AI service connectivity check
        try:
            async with aiohttp.ClientSession() as session:
                for service, config in self.ai_config.items():
                    health_url = config['endpoint'].replace('/completions', '/health')
                    async with session.get(health_url, timeout=10) as response:
                        if response.status != 200:
                            logger.warning(f"  {service} health check failed!")
                            return False
            return True
        except Exception as e:
            logger.warning(f"Health check failed: {str(e)}")
            return False

    async def _plan_targets(self) -> List[str]:
        """Generate scraping targets using AI planner"""
        try:
            planner = MetaPlanner()
            return await planner.generate_scrape_plan(
                self.topic, 
                self.ai_config["CYPHER"]
            )
        except Exception as e:
            logger.error(f"Target planning failed: {str(e)}")
            # Fallback to predefined targets
            return [
                "https://aseanenergy.org",
                "https://www.irena.org/",
                "https://www.worldbank.org/en/region/eap/brief/asean",
                "https://www.adb.org/sectors/energy/renewable-energy",
                "https://www.asean-renewables.org/"
            ]

    async def _execute_scraping(self, targets: List[str]) -> List[Dict[str, Any]]:
        """Execute parallel scraping operations"""
        semaphore = asyncio.Semaphore(self.active_workers)
        
        async def _scrape_with_limit(target: str):
            async with semaphore:
                scraper = Scraper()
                return await scraper.execute(target, self.ai_config["DEEPSEEK"])
        
        tasks = [_scrape_with_limit(target) for target in targets]
        return await asyncio.gather(*tasks, return_exceptions=False)

    async def _process_data(self, scraped_data: List[Dict]) -> List[Dict]:
        """Process and clean scraped data"""
        cleaner = DataCleaner()
        processed = []
        
        for data in scraped_data:
            try:
                # Skip failed items
                if data.get('status') != 'SUCCESS':
                    continue
                    
                # Clean and transform data
                cleaned = await cleaner.clean(
                    data['content'], 
                    self.ai_config["CLAUDE"]
                )
                
                # Add metadata
                processed.append({
                    'metadata': {
                        'source': data['url'],
                        'scrape_time': data['timestamp'],
                        'processing_time': datetime.now().isoformat(),
                        'cycle': self.cycle_count
                    },
                    'content': cleaned
                })
            except Exception as e:
                logger.error(f"Data processing failed: {str(e)}")
                
        return processed

    async def _store_data(self, processed_data: List[Dict]) -> Dict[str, int]:
        """Store processed data with redundancy"""
        results = {
            'primary': 0,
            'backup': 0
        }
        
        # Store in primary storage
        if processed_data:
            try:
                results['primary'] = await self.storage.save_batch(processed_data)
                logger.info(f"📦 Primary storage: Saved {results['primary']} items")
            except Exception as e:
                logger.error(f"Primary storage failed: {str(e)}")
                
            # Backup storage
            try:
                results['backup'] = await self.backup_storage.save_batch(processed_data)
                logger.info(f"💽 Backup storage: Saved {results['backup']} items")
            except Exception as e:
                logger.error(f"Backup storage failed: {str(e)}")
                
        return results

    async def _analyze_failures(self, failures: List[Dict]):
        """Analyze and heal from failures"""
        logger.info("🔍 Analyzing failures...")
        
        healer = Healer()
        try:
            # Perform failure analysis
            diagnosis = await healer.diagnose(
                failures, 
                self.ai_config["CLAUDE"],
                self.ai_config["CYPHER"]
            )
            
            # Apply healing solutions
            if diagnosis and diagnosis.get('solutions'):
                logger.info("💊 Applying healing solutions")
                await healer.apply_solutions(diagnosis['solutions'])
        except Exception as e:
            logger.error(f"Failure analysis failed: {str(e)}")

    async def _optimize_performance(self):
        """Optimize system performance based on metrics"""
        logger.info("⚙️ Optimizing performance...")
        
        # Adjust workers based on CPU usage
        current_cpu = self.monitor.cpu_usage()
        
        if current_cpu > self.scaling_threshold and self.active_workers < self.max_workers:
            new_workers = min(self.active_workers + 2, self.max_workers)
            logger.info(f"⬆️ Scaling UP workers: {self.active_workers} → {new_workers}")
            self.active_workers = new_workers
            
        elif current_cpu < (self.scaling_threshold / 2) and self.active_workers > self.min_workers:
            new_workers = max(self.active_workers - 1, self.min_workers)
            logger.info(f"⬇️ Scaling DOWN workers: {self.active_workers} → {new_workers}")
            self.active_workers = new_workers

    async def _self_heal(self, issue_type: str):
        """Perform self-healing for detected issues"""
        logger.info(f"🛠️ Self-healing initiated for: {issue_type}")
        
        healer = Healer()
        try:
            # Get healing strategy from AI
            solution = await healer.generate_healing_strategy(
                issue_type, 
                self.ai_config["CYPHER"]
            )
            
            # Execute healing strategy
            if solution:
                logger.info(f"🔧 Applying solution: {solution[:100]}...")
                await healer.apply_solution(solution)
        except Exception as e:
            logger.error(f"Self-healing failed: {str(e)}")

    async def _emergency_heal(self, error: Exception, error_id: str):
        """Emergency recovery procedure for critical failures"""
        logger.critical("🚑 EMERGENCY HEALING PROCEDURE ACTIVATED!")
        
        try:
            # Capture system state for diagnosis
            state = {
                'error': str(error),
                'error_id': error_id,
                'timestamp': datetime.now().isoformat(),
                'system': {
                    'os': f"{platform.system()} {platform.release()}",
                    'python': platform.python_version(),
                    'cpu': psutil.cpu_percent(),
                    'memory': psutil.virtual_memory().percent,
                    'disk': psutil.disk_usage('/').percent
                },
                'swarm_state': {
                    'cycle': self.cycle_count,
                    'consecutive_failures': self.consecutive_failures,
                    'active_workers': self.active_workers,
                    'status': self.status
                }
            }
            
            # Send diagnostic data to healing system
            healer = Healer()
            solution = await healer.emergency_heal(state, self.ai_config["CLAUDE"])
            
            # Execute emergency solution
            if solution:
                logger.info("🆘 Applying emergency solution...")
                await healer.apply_solution(solution)
                
                # Reset system state
                self.active_workers = self.min_workers
                self.consecutive_failures = 0
                self.status = "RECOVERY_MODE"
                
                logger.info("♻️ System reset to safe state")
        except Exception as heal_error:
            logger.critical(f"❌ CRITICAL UNRECOVERABLE FAILURE: {str(heal_error)}")
            logger.critical("🛑 SYSTEM SHUTDOWN REQUIRED")
            self.status = "FAILED"
            # Implement graceful shutdown
            await self._graceful_shutdown()

    async def _graceful_shutdown(self):
        """Perform graceful shutdown of the system"""
        logger.info("🛑 Initiating graceful shutdown...")
        # Clean up resources, close connections, etc.
        await self.storage.close()
        await self.backup_storage.close()
        logger.info("🛑 System shutdown complete")
        # Exit the application
        raise SystemExit("Emergency shutdown")

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status report"""
        return {
            'status': self.status,
            'cycle_count': self.cycle_count,
            'consecutive_failures': self.consecutive_failures,
            'active_workers': self.active_workers,
            'start_time': self.start_time.isoformat(),
            'last_cycle_duration': self.last_cycle_time,
            'system_metrics': self.monitor.get_all_metrics(),
            'ai_services': list(self.ai_config.keys())
        }

# Main execution flow
async def main():
    """Entry point for the swarm system"""
    logger.info("🚀 Starting Super AI Scraper System")
    
    # Initialize swarm with topic
    swarm = SuperSwarm(topic="renewable energy in ASEAN")
    
    # Run continuous cycles
    while swarm.status != "FAILED":
        await swarm.run_cycle()
        
        # Add delay between cycles based on performance
        delay = 300  # 5 minutes default
        if swarm.last_cycle_time and swarm.last_cycle_time > 120:
            delay = 600  # 10 minutes if last cycle was long
            
        logger.info(f"⏳ Next cycle in {delay//60} minutes")
        await asyncio.sleep(delay)
    
    logger.error("❌ System has entered FAILED state. Manual intervention required.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 System shutdown requested by user")
    except Exception as e:
        logger.critical(f"‼️ UNHANDLED SYSTEM FAILURE: {str(e)}")
        raise
