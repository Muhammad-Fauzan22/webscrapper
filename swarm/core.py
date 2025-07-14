import os
import asyncio
import aiohttp
import logging
from swarm.agents.meta_planner import MetaPlanner
from swarm.agents.scraper import Scraper
from swarm.storage.gdrive import DriveStorage
from swarm.agents.healer import Healer
from config import CONFIG, AI_CONFIG, SCALING_PROFILE, TIMESTAMP, USER, ALERT_THRESHOLDS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SuperSwarm")

class SuperSwarm:
    def __init__(self):
        self.storage = DriveStorage()
        self.active_workers = CONFIG["workers"]
        self.cycle_count = 0
        self.failed_cycles = 0
        
    async def run_cycle(self):
        """Main execution cycle for the swarm"""
        self.cycle_count += 1
        logger.info(f"\n=== CYCLE {self.cycle_count} ===")
        logger.info(f"Timestamp: {TIMESTAMP} | User: {USER}")
        
        try:
            # 1. AI-Powered Planning Phase
            plan = await MetaPlanner.generate_scrape_plan(
                "renewable energy in ASEAN",
                AI_CONFIG["CYPHER"]
            )
            logger.info(f"Generated {len(plan)} targets for scraping")
            
            # 2. Parallel Execution Phase
            async with aiohttp.ClientSession() as session:
                tasks = [
                    Scraper.execute(session, target, AI_CONFIG["DEEPSEEK"]) 
                    for target in plan[:CONFIG["max_targets"]]
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                successes = [r for r in results if not isinstance(r, Exception)]
                errors = [r for r in results if isinstance(r, Exception)]
                
                logger.info(f"Scraping completed: {len(successes)} success, {len(errors)} errors")
                
                # 3. Intelligent Storage
                if successes:
                    success_count = await self.storage.save_batch(successes)
                    logger.info(f"Saved {success_count} items to storage")
                
            # 4. Auto-Healing Mechanism
            if self.requires_healing(results):
                logger.warning("High error rate detected, initiating healing...")
                await Healer.diagnose_and_fix(
                    errors,
                    AI_CONFIG["CLAUDE"],
                    AI_CONFIG["CYPHER"]
                )
                
            # 5. Dynamic Scaling
            self.adjust_workers()
            
            # Reset failed cycle counter
            self.failed_cycles = 0
            return True
            
        except Exception as e:
            self.failed_cycles += 1
            logger.error(f"Critical error in swarm cycle: {str(e)}")
            
            # Emergency healing after consecutive failures
            if self.failed_cycles >= 3:
                logger.critical("Consecutive failures detected! Initiating emergency healing...")
                await self.emergency_heal(e)
                
            return False
            
    async def emergency_heal(self, error: Exception):
        """Emergency healing for critical failures"""
        from swarm.agents.healer import Healer
        
        try:
            diagnosis = await Healer.diagnose_error(
                str(error),
                AI_CONFIG["CLAUDE"],
                AI_CONFIG["CYPHER"]
            )
            
            if "solution" in diagnosis:
                logger.info(f"Applying emergency solution: {diagnosis['solution'][:100]}...")
                # Implement solution logic here
                
            # Reset workers to minimum
            self.active_workers = SCALING_PROFILE["min_workers"]
            logger.info(f"Reset workers to minimum: {self.active_workers}")
            
        except Exception as heal_error:
            logger.error(f"Emergency healing failed: {str(heal_error)}")
            
    def requires_healing(self, results) -> bool:
        """Determine if healing is needed based on error ratio"""
        if not results:
            return False
            
        error_count = sum(1 for r in results if isinstance(r, Exception))
        error_ratio = error_count / len(results)
        return error_ratio > ALERT_THRESHOLDS["healing_ratio"]
        
    def adjust_workers(self):
        """Adjust worker count based on system load"""
        # Simulated system monitoring
        cpu_load = 65  # Placeholder
        
        if cpu_load > SCALING_PROFILE["scale_up_threshold"]:
            new_workers = min(
                self.active_workers * 2,
                SCALING_PROFILE["max_workers"]
            )
            logger.info(f"Scaling UP workers: {self.active_workers} → {new_workers}")
            self.active_workers = new_workers
            
        elif cpu_load < SCALING_PROFILE["scale_down_threshold"]:
            new_workers = max(
                self.active_workers // 2,
                SCALING_PROFILE["min_workers"]
            )
            logger.info(f"Scaling DOWN workers: {self.active_workers} → {new_workers}")
            self.active_workers = new_workers

if __name__ == "__main__":
    swarm = SuperSwarm()
    asyncio.run(swarm.run_cycle())
