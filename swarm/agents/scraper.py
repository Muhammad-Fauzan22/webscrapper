#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADVANCED INTELLIGENT WEB SCRAPER AGENT
Version: 4.2.1
Created: 2025-07-15
Author: Data Acquisition Master Team
"""

import asyncio
import logging
import base64
import random
import time
import json
import re
import zlib
from datetime import datetime
from urllib.parse import urlparse
from playwright.async_api import async_playwright, TimeoutError
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import aiohttp

logger = logging.getLogger("ScraperPro")

class AdvancedScraper:
    def __init__(self, 
                 timeout=25, 
                 max_retries=3, 
                 headless=True,
                 stealth_mode=True,
                 resource_blacklist=["image", "stylesheet", "font"]):
        """
        Initialize advanced web scraper with enhanced capabilities
        
        Args:
            timeout: Operation timeout in seconds
            max_retries: Maximum retry attempts
            headless: Run browser in headless mode
            stealth_mode: Enable anti-detection measures
            resource_blacklist: Resource types to block
        """
        self.timeout = timeout * 1000  # Convert to ms
        self.max_retries = max_retries
        self.headless = headless
        self.stealth_mode = stealth_mode
        self.resource_blacklist = resource_blacklist
        self.ua = UserAgent()
        self.proxy_pool = self._init_proxy_pool()
        
        # Security and bot protection thresholds
        self.security_params = {
            'max_redirects': 5,
            'captcha_timeout': 15,
            'bot_signatures': [
                'distil', 'incapsula', 'cloudflare', 'akamai',
                'imperva', 'datadome', 'perimeterx', 'recaptcha'
            ]
        }
        
        # Performance metrics
        self.metrics = {
            'total_requests': 0,
            'success_count': 0,
            'failure_count': 0,
            'avg_load_time': 0,
            'bot_detections': 0
        }

    async def execute(self, url: str, ai_config: dict) -> dict:
        """
        Execute advanced scraping operation with AI-enhanced capabilities
        
        Args:
            url: Target URL to scrape
            ai_config: AI service configuration
            
        Returns:
            Dictionary with scraping results and metadata
        """
        result = {
            'status': 'pending',
            'url': url,
            'retries': 0,
            'start_time': datetime.now().isoformat(),
            'security': {}
        }
        
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"🌐 Attempt {attempt}/{self.max_retries} for {url}")
                result['retries'] = attempt
                
                scraped_data = await self._scrape_page(url, attempt)
                
                # Security check
                if await self._detect_bot_protection(scraped_data['content']):
                    result['security']['bot_detected'] = True
                    result['security']['confidence'] = 0.85
                    logger.warning(f"🤖 Bot protection detected on {url}")
                    
                    # Bypass attempt
                    if attempt < self.max_retries:
                        logger.info("🛡️ Attempting bypass...")
                        scraped_data = await self._bypass_protection(url, scraped_data)
                
                # AI-enhanced content analysis
                if scraped_data['content']:
                    analysis = await self._analyze_content(
                        scraped_data['content'], 
                        ai_config,
                        url
                    )
                    scraped_data['analysis'] = analysis
                
                # Success handling
                result.update(scraped_data)
                result['status'] = 'success'
                result['end_time'] = datetime.now().isoformat()
                result['duration'] = time.time() - scraped_data['metrics']['start_time']
                
                # Update metrics
                self.metrics['total_requests'] += 1
                self.metrics['success_count'] += 1
                self.metrics['avg_load_time'] = (
                    (self.metrics['avg_load_time'] * (self.metrics['success_count'] - 1) + 
                    result['duration']
                ) / self.metrics['success_count']
                
                return result
                
            except TimeoutError:
                logger.warning(f"⏱️ Timeout on attempt {attempt} for {url}")
                result['last_error'] = 'timeout'
                await self._adjust_timeout(attempt)
                
            except Exception as e:
                logger.error(f"🚨 Error on attempt {attempt}: {str(e)}")
                result['last_error'] = str(e)
                await asyncio.sleep(self._backoff_time(attempt))
        
        # Final failure handling
        result['status'] = 'failure'
        result['end_time'] = datetime.now().isoformat()
        self.metrics['failure_count'] += 1
        
        # Fallback to basic HTTP request
        if result['status'] == 'failure':
            logger.info("🆘 Using HTTP fallback")
            fallback_data = await self._http_fallback(url)
            if fallback_data:
                result.update(fallback_data)
                result['status'] = 'fallback_success'
        
        return result
        
    async def _scrape_page(self, url: str, attempt: int) -> dict:
        """
        Scrape web page using Playwright with advanced features:
        - Stealth mode
        - Resource blocking
        - Proxy rotation
        - Security evasion
        """
        async with async_playwright() as p:
            # Browser configuration
            launch_options = {
                'headless': self.headless,
                'timeout': self.timeout,
                'proxy': self._get_proxy() if self.proxy_pool else None
            }
            
            browser = await p.chromium.launch(**launch_options)
            context = await browser.new_context(
                user_agent=self._get_user_agent(attempt),
                viewport={'width': 1366, 'height': 768},
                locale='en-US,en',
                timezone_id='Asia/Singapore',
                java_script_enabled=True,
                ignore_https_errors=True,
                bypass_csp=True
            )
            
            # Enable stealth mode
            if self.stealth_mode:
                await self._enable_stealth(context)
            
            # Block unnecessary resources
            await context.route('**/*', self._block_resources)
            
            page = await context.new_page()
            start_time = time.time()
            metrics = {'start_time': start_time}
            
            try:
                # Navigate with enhanced monitoring
                navigation_result = await page.goto(
                    url, 
                    timeout=self.timeout, 
                    wait_until="domcontentloaded"
                )
                
                # Check HTTP status
                status = navigation_result.status
                metrics['http_status'] = status
                
                if status >= 400:
                    raise ConnectionError(f"HTTP error {status}")
                
                # Wait for page completion
                await self._wait_for_page(page, attempt)
                
                # Security checks
                metrics['security'] = await self._check_security(page)
                
                # Capture content and assets
                content = await self._get_page_content(page)
                screenshot = await page.screenshot(type='png', full_page=True)
                html = await page.content()
                
                # Compress HTML for storage
                compressed_html = zlib.compress(html.encode('utf-8'))
                
                return {
                    'content': content,
                    'screenshot': base64.b64encode(screenshot).decode('utf-8'),
                    'html': base64.b64encode(compressed_html).decode('utf-8'),
                    'metrics': metrics
                }
                
            finally:
                # Close browser and update metrics
                await browser.close()
                metrics['load_time'] = time.time() - start_time
    
    async def _enable_stealth(self, context):
        """Inject advanced anti-detection scripts"""
        await context.add_init_script("""
            // Remove webdriver property
            delete navigator.__proto__.webdriver;
            
            // Mock plugins
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5],
                configurable: true
            });
            
            // Mock languages
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en'],
                configurable: true
            });
            
            // Mock permissions
            const originalQuery = navigator.permissions.query;
            navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications' ?
                    Promise.resolve({ state: Notification.permission }) :
                    originalQuery(parameters)
            );
            
            // Mock WebGL
            const getParameter = WebGLRenderingContext.prototype.getParameter;
            WebGLRenderingContext.prototype.getParameter = function(parameter) {
                if (parameter === 37445) return 'Intel Open Source Technology Center';
                if (parameter === 37446) return 'Mesa DRI Intel(R) HD Graphics';
                return getParameter(parameter);
            };
        """)
    
    async def _block_resources(self, route):
        """Block unnecessary resources to improve performance"""
        resource_type = route.request.resource_type
        if resource_type in self.resource_blacklist:
            await route.abort()
        else:
            await route.continue_()
    
    async def _wait_for_page(self, page, attempt):
        """Intelligent wait strategy based on page characteristics"""
        # Initial wait for core content
        await page.wait_for_load_state('domcontentloaded', timeout=self.timeout)
        
        # Dynamic wait based on attempt number
        wait_strategies = [
            {'state': 'networkidle', 'timeout': self.timeout},
            {'selector': 'body', 'timeout': 5000},
            {'function': 'document.readyState === "complete"', 'timeout': 3000}
        ]
        
        for strategy in wait_strategies[:attempt]:
            try:
                if 'state' in strategy:
                    await page.wait_for_load_state(strategy['state'], timeout=strategy['timeout'])
                elif 'selector' in strategy:
                    await page.wait_for_selector(strategy['selector'], timeout=strategy['timeout'])
                elif 'function' in strategy:
                    await page.wait_for_function(strategy['function'], timeout=strategy['timeout'])
            except:
                continue
    
    async def _check_security(self, page):
        """Detect security mechanisms and bot protection"""
        security = {
            'bot_protection': False,
            'captcha_present': False,
            'security_services': []
        }
        
        # Check for common bot protection services
        content = await page.content()
        for service in self.security_params['bot_signatures']:
            if re.search(service, content, re.IGNORECASE):
                security['security_services'].append(service)
                security['bot_protection'] = True
        
        # Detect CAPTCHAs
        captcha_selectors = [
            '.g-recaptcha', 
            '#recaptcha', 
            'iframe[src*="recaptcha"]',
            'div[class*="captcha"]'
        ]
        
        for selector in captcha_selectors:
            if await page.query_selector(selector):
                security['captcha_present'] = True
                break
                
        return security
    
    async def _detect_bot_protection(self, content: str) -> bool:
        """AI-enhanced bot protection detection"""
        # Signature-based detection
        for service in self.security_params['bot_signatures']:
            if service.lower() in content.lower():
                return True
                
        # Content pattern detection
        bot_indicators = [
            "access denied", 
            "you are not human",
            "bot detected",
            "security challenge"
        ]
        
        if any(indicator in content.lower() for indicator in bot_indicators):
            return True
            
        return False
    
    async def _bypass_protection(self, url: str, existing_data: dict) -> dict:
        """Attempt to bypass bot protection mechanisms"""
        bypass_strategies = [
            self._change_user_agent,
            self._use_proxy_rotation,
            self._simulate_human_interaction,
            self._use_headless_browser,
            self._execute_js_evasion
        ]
        
        for strategy in bypass_strategies:
            try:
                logger.info(f"🛠️ Trying bypass: {strategy.__name__}")
                result = await strategy(url, existing_data)
                if result and not await self._detect_bot_protection(result['content']):
                    logger.info("✅ Bypass successful")
                    self.metrics['bot_detections'] += 1
                    return result
            except Exception as e:
                logger.warning(f"Bypass failed: {str(e)}")
                
        return existing_data
    
    async def _change_user_agent(self, url, data):
        """Bypass by changing user agent"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=self.headless)
            context = await browser.new_context(
                user_agent=self._get_user_agent(force_new=True),
                **self._get_context_options()
            )
            page = await context.new_page()
            await page.goto(url, timeout=self.timeout)
            content = await self._get_page_content(page)
            await browser.close()
            return {'content': content}
    
    async def _use_proxy_rotation(self, url, data):
        """Bypass using proxy rotation"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=self.headless,
                proxy=self._get_proxy(force_new=True)
            )
            context = await browser.new_context(**self._get_context_options())
            page = await context.new_page()
            await page.goto(url, timeout=self.timeout)
            content = await self._get_page_content(page)
            await browser.close()
            return {'content': content}
    
    async def _simulate_human_interaction(self, url, data):
        """Bypass by simulating human-like behavior"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=self.headless)
            context = await browser.new_context(**self._get_context_options())
            page = await context.new_page()
            
            await page.goto(url, timeout=self.timeout)
            
            # Simulate human interaction patterns
            await page.mouse.move(100, 100)
            await page.wait_for_timeout(random.randint(200, 800))
            await page.mouse.wheel(0, 500)
            await page.wait_for_timeout(random.randint(500, 1500))
            await page.mouse.click(200, 300, delay=random.randint(50, 200))
            await page.wait_for_timeout(random.randint(1000, 3000))
            
            content = await self._get_page_content(page)
            await browser.close()
            return {'content': content}
    
    async def _use_headless_browser(self, url, data):
        """Bypass by using non-headless browser"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            context = await browser.new_context(**self._get_context_options())
            page = await context.new_page()
            await page.goto(url, timeout=self.timeout)
            content = await self._get_page_content(page)
            await browser.close()
            return {'content': content}
    
    async def _execute_js_evasion(self, url, data):
        """Bypass by executing JavaScript evasion techniques"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=self.headless)
            context = await browser.new_context(**self._get_context_options())
            page = await context.new_page()
            
            # Evasion JavaScript code
            evasion_js = """
            // Delete bot detection variables
            delete window.__webdriver;
            delete window.document.$cdc_asdjflasutopfhvcZLmcfl_;
            delete window.document.documentElement.getAttribute('webdriver');
            
            // Overwrite permission API
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications' ? 
                    Promise.resolve({ state: Notification.permission }) :
                    originalQuery(parameters)
            );
            """
            
            await page.add_init_script(evasion_js)
            await page.goto(url, timeout=self.timeout)
            content = await self._get_page_content(page)
            await browser.close()
            return {'content': content}
    
    async def _get_page_content(self, page) -> str:
        """Extract and clean page content with semantic analysis"""
        try:
            # Get rendered content
            content = await page.evaluate("""
                () => {
                    // Remove non-content elements
                    const removals = [
                        'script', 'style', 'noscript', 'meta', 'link',
                        'footer', 'header', 'nav', 'aside', 'form'
                    ];
                    
                    removals.forEach(tag => {
                        document.querySelectorAll(tag).forEach(el => el.remove());
                    });
                    
                    // Extract semantic content
                    const mainContent = document.querySelector('main') || 
                                       document.querySelector('article') || 
                                       document.body;
                                       
                    return mainContent.innerText;
                }
            """)
            
            # Clean and structure content
            return self._clean_content(content)
        except Exception as e:
            logger.error(f"Content extraction failed: {str(e)}")
            return ""

    def _clean_content(self, content: str) -> str:
        """Advanced content cleaning and normalization"""
        # Remove non-printable characters
        cleaned = re.sub(r'[\x00-\x1F\x7F-\x9F]', ' ', content)
        
        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Remove boilerplate text
        boilerplates = [
            r'privacy\s*policy', r'terms\s*of\s*use', r'cookie\s*policy',
            r'all\s*rights\s*reserved', r'©\s*\d{4}', r'[\w\s]+\.com'
        ]
        
        for pattern in boilerplates:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Extract meaningful paragraphs
        paragraphs = [p.strip() for p in cleaned.split('.') if len(p.split()) > 5]
        return '. '.join(paragraphs[:20])  # Limit to 20 most meaningful paragraphs
    
    async def _analyze_content(self, content: str, ai_config: dict, url: str) -> dict:
        """AI-powered content analysis with domain adaptation"""
        domain = urlparse(url).netloc
        domain_key = domain.replace('.', '_')[:20]
        
        prompt = f"""
        Analyze scraped web content from {domain} and extract structured data.
        Consider domain-specific patterns for {domain_key}.
        
        Content:
        {content[:8000]}{'... [truncated]' if len(content) > 8000 else ''}
        
        Required JSON output:
        {{
            "main_topic": "string (max 3 words)",
            "keywords": ["list", "of", "top", "5", "keywords"],
            "summary": "string (50-100 words)",
            "sentiment": {{
                "overall": "positive/neutral/negative",
                "score": 0.0-1.0
            }},
            "language": "ISO 639-1 code",
            "entities": [
                {{
                    "name": "string",
                    "type": "PERSON/ORG/LOCATION/OTHER",
                    "relevance": 0.0-1.0
                }}
            ],
            "domain_specific": {{
                "{domain_key}": {{
                    "key_metrics": [],
                    "notable_features": []
                }}
            }}
        }}
        """
        
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": ai_config["model"],
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1500,
                "temperature": 0.2,
                "response_format": {"type": "json_object"}
            }
            
            headers = {
                "Authorization": f"Bearer {ai_config['key']}",
                "Content-Type": "application/json"
            }
            
            try:
                async with session.post(
                    ai_config["endpoint"],
                    json=payload,
                    headers=headers,
                    timeout=45
                ) as response:
                    if response.status != 200:
                        error = await response.text()
                        raise RuntimeError(f"AI API error {response.status}: {error}")
                    
                    data = await response.json()
                    return json.loads(data["choices"][0]["message"]["content"])
            except Exception as e:
                logger.error(f"Content analysis failed: {str(e)}")
                return {}
    
    async def _http_fallback(self, url: str) -> dict:
        """Basic HTTP fallback when Playwright fails"""
        try:
            headers = {
                'User-Agent': self._get_user_agent(),
                'Accept': 'text/html,application/xhtml+xml',
                'Accept-Language': 'en-US,en;q=0.9',
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=10) as response:
                    content = await response.text()
                    return {
                        'content': self._clean_content(content),
                        'metrics': {
                            'http_status': response.status,
                            'fallback': True
                        }
                    }
        except Exception as e:
            logger.error(f"HTTP fallback failed: {str(e)}")
            return {}
    
    def _get_user_agent(self, attempt=1, force_new=False) -> str:
        """Get user agent with rotation logic"""
        if force_new or attempt % 2 == 0:
            return self.ua.random
        return random.choice([
            # Popular desktop browsers
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64; rv:126.0) Gecko/20100101 Firefox/126.0",
            
            # Mobile browsers
            "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
            "Mozilla/5.0 (Linux; Android 14; SM-S918B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.6422.147 Mobile Safari/537.36"
        ])
    
    def _init_proxy_pool(self) -> list:
        """Initialize proxy pool from environment or config"""
        proxy_env = os.getenv("PROXY_POOL", "")
        if proxy_env:
            return [p.strip() for p in proxy_env.split(',') if p.strip()]
        return []
    
    def _get_proxy(self, force_new=False) -> dict:
        """Get proxy configuration with rotation"""
        if not self.proxy_pool:
            return None
            
        proxy = random.choice(self.proxy_pool)
        return {
            'server': proxy,
            'username': os.getenv("PROXY_USER", ""),
            'password': os.getenv("PROXY_PASS", "")
        }
    
    def _backoff_time(self, attempt: int) -> float:
        """Calculate exponential backoff with jitter"""
        base = 2
        max_backoff = 10
        backoff = min(max_backoff, base ** attempt)
        jitter = random.uniform(0.5, 1.5)
        return backoff * jitter
    
    def _get_context_options(self) -> dict:
        """Get default context options"""
        return {
            'viewport': {'width': 1366, 'height': 768},
            'locale': 'en-US,en',
            'timezone_id': 'Asia/Singapore',
            'java_script_enabled': True,
            'ignore_https_errors': True,
            'bypass_csp': True
        }
    
    async def _adjust_timeout(self, attempt: int):
        """Dynamically adjust timeout based on attempt"""
        self.timeout = min(120000, self.timeout * 1.5)  # Max 2 minutes
        logger.info(f"⏱️ Adjusted timeout to {self.timeout/1000}s")
        await asyncio.sleep(self._backoff_time(attempt))
    
    def get_metrics(self) -> dict:
        """Get current scraping metrics"""
        return self.metrics
