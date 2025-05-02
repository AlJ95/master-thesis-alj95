import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse, urljoin
import asyncio
import aiohttp
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from multiprocessing import freeze_support
import csv
import os
from pathlib import Path
import signal
import logging
import time
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set a timeout for HTTP requests
TIMEOUT = 30

# Configure session parameters
CONN_TIMEOUT = aiohttp.ClientTimeout(total=TIMEOUT, connect=10)

# Rate limiting configuration
MAX_REQUESTS_PER_SECOND = 5
REQUEST_INTERVAL = 1.0 / MAX_REQUESTS_PER_SECOND

base_urls = {
    # "alluxio": "https://docs.alluxio.io/os/javadoc/2.5/",
    # "django": "https://docs.djangoproject.com/en/4.0/",
    # "etcd": "https://etcd.io/docs/v3.5/",
    # "hbase": "https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-common/",
    # "hdfs": "https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-hdfs/",
    "postgresql": "https://www.postgresql.org/docs/13/",
    # "redis": "https://redis.io/docs/latest/commands/",
    # "yarn": "https://hadoop.apache.org/docs/r3.3.0/",
    # "zookeeper-server": "https://zookeeper.apache.org/doc/r3.7.0/apidocs/zookeeper-server/",
}


class RateLimiter:
    """Rate limiter to control request frequency per domain"""
    def __init__(self, requests_per_second=MAX_REQUESTS_PER_SECOND):
        self.requests_per_second = requests_per_second
        self.interval = 1.0 / requests_per_second
        self.domain_last_request = defaultdict(float)
        self._lock = asyncio.Lock()
    
    async def acquire(self, url):
        """Acquire permission to make a request to the specified domain"""
        domain = urlparse(url).netloc
        
        async with self._lock:
            current_time = time.time()
            time_since_last_request = current_time - self.domain_last_request[domain]
            
            if time_since_last_request < self.interval:
                # Need to wait before making another request
                wait_time = self.interval - time_since_last_request
                await asyncio.sleep(wait_time)
            
            # Update the last request time
            self.domain_last_request[domain] = time.time()


def clean_url(url):
    """Remove hash fragments from URLs"""
    return url.split('#')[0]


def get_safe_filename(url):
    """Convert URL to a safe filename"""
    # Remove protocol and replace special characters
    filename = re.sub(r'^https?://', '', url)
    filename = re.sub(r'[\\/*?:"<>|]', '_', filename)
    # Replace remaining slashes with underscores
    filename = filename.replace('/', '_')
    # Limit filename length
    if len(filename) > 200:
        filename = filename[:200]
    return filename


async def scrape_url_async(url, base_url, visited_urls, system_urls, session, system, rate_limiter):
    """Scrape a single URL asynchronously and return new URLs found"""
    try:
        # Apply rate limiting before making the request
        await rate_limiter.acquire(url)
        
        async with session.get(url, timeout=CONN_TIMEOUT, ssl=False) as response:
            if response.status != 200:
                logger.warning(f"Non-200 status for {url}: {response.status}")
                return []
            
            try:
                html = await response.text()
            except UnicodeDecodeError:
                logger.warning(f"Unicode decode error for {url}, trying with ISO-8859-1")
                html = await response.read()
                html = html.decode('iso-8859-1', errors='ignore')
            
            # Save HTML content to file
            html_dir = Path("raw/html") / system
            html_dir.mkdir(parents=True, exist_ok=True)
            
            filename = get_safe_filename(url)
            filepath = html_dir / f"{filename}.html"
            
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(html)
            except Exception as e:
                logger.error(f"Error saving HTML file for {url}: {e}")
            
            try:
                # Use a more lenient parser
                soup = BeautifulSoup(html, 'html.parser', from_encoding='utf-8')
                new_urls = []
                
                for link in soup.find_all('a', href=True):
                    try:
                        href = link['href']
                        
                        # Handle relative URLs
                        if not href.startswith(('http://', 'https://')):
                            href = urljoin(url, href)
                        
                        # Clean the URL by removing hash fragments
                        href = clean_url(href)
                        
                        # Only consider URLs that start with the base_url
                        if (href not in visited_urls and 
                            href.startswith(base_url) and
                            href not in system_urls):
                            new_urls.append(href)
                            system_urls.append(href)
                            visited_urls.add(href)
                    except Exception as e:
                        logger.warning(f"Error processing link in {url}: {e}")
                
                return new_urls
            except Exception as e:
                logger.error(f"Error parsing HTML for {url}: {e}")
                return []
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logger.warning(f"Connection error for {url}: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error scraping {url}: {e}")
        return []


async def scrape_system_async(system, base_url, max_depth=2, max_urls_per_system=100):
    """Scrape a single system asynchronously"""
    all_urls = []
    
    # Track visited URLs to avoid duplicates
    visited_urls = set()
    visited_urls.add(base_url)
    
    # Initialize with the base URL
    urls_to_visit = [base_url]
    all_urls.append(base_url)
    
    # Set up TCP connector with limits
    connector = aiohttp.TCPConnector(limit=10, ttl_dns_cache=300)
    
    # Create a rate limiter for this system
    rate_limiter = RateLimiter(MAX_REQUESTS_PER_SECOND)
    
    try:
        async with aiohttp.ClientSession(connector=connector, timeout=CONN_TIMEOUT) as session:
            # Process URLs up to max_depth
            for depth in range(max_depth):
                if not urls_to_visit or len(all_urls) >= max_urls_per_system:
                    break
                    
                logger.info(f"Depth {depth+1}: Processing {len(urls_to_visit)} URLs for {system}")
                
                # Process URLs in batches to avoid overwhelming resources
                batch_size = 5
                for i in range(0, len(urls_to_visit), batch_size):
                    batch = urls_to_visit[i:i+batch_size]
                    
                    # Process all URLs at this depth concurrently
                    tasks = [scrape_url_async(url, base_url, visited_urls, all_urls, session, system, rate_limiter) 
                            for url in batch[:max_urls_per_system - len(all_urls)]]
                    
                    if not tasks:
                        break
                        
                    try:
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                        # Filter out exceptions
                        results = [r for r in results if not isinstance(r, Exception) and r is not None]
                        
                        # Flatten the list of lists
                        next_urls = [url for sublist in results for url in sublist]
                        urls_to_visit.extend(next_urls)
                        
                        # No need for additional sleep as rate limiter handles delays
                    except asyncio.CancelledError:
                        logger.warning(f"Tasks cancelled for {system}")
                        break
        
        logger.info(f"Found {len(all_urls)} unique URLs for {system}")
        return system, all_urls
    except Exception as e:
        logger.error(f"Error scraping system {system}: {e}")
        return system, all_urls


def scrape_system_wrapper(args):
    """Wrapper function for multiprocessing"""
    try:
        system, base_url, max_depth, max_urls_per_system = args
        logger.info(f"Starting scrape for {system} from {base_url}")
        
        # Set a loop policy that works better with Windows
        if os.name == 'nt':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            
        result = asyncio.run(scrape_system_async(system, base_url, max_depth, max_urls_per_system))
        return result
    except Exception as e:
        logger.error(f"Error in scrape_system_wrapper for {args[0]}: {e}")
        return args[0], []


def scrape_base_urls(max_depth=2, max_urls_per_system=1000):
    """Scrape all base URLs using multiprocessing"""
    all_urls = {system: [] for system in base_urls}
    
    # Ensure the raw/html directory exists
    Path("raw/html").mkdir(parents=True, exist_ok=True)
    
    # Prepare arguments for multiprocessing
    args_list = [(system, base_url, max_depth, max_urls_per_system) 
                for system, base_url in base_urls.items()]
    
    # Limit number of concurrent processes and reduce max_workers
    max_workers = min(4, multiprocessing.cpu_count())
    logger.info(f"Using {max_workers} workers for parallel scraping")
    
    # Use ProcessPoolExecutor to parallelize across systems
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(scrape_system_wrapper, args_list))
        
        # Collect results
        for system, urls in results:
            all_urls[system] = urls
    except KeyboardInterrupt:
        logger.warning("Scraping interrupted by user")
    except Exception as e:
        logger.error(f"Error in scrape_base_urls: {e}")
    
    return all_urls


def save_results_to_csv(urls_by_system):
    """Save URLs to CSV file"""
    try:
        with open('urls.csv', 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            # Write header
            csv_writer.writerow(['system', 'url'])
            
            # Write URLs for each system
            total_urls = 0
            for system, urls in urls_by_system.items():
                for url in urls:
                    csv_writer.writerow([system, url])
                    total_urls += 1
            
            logger.info(f"Saved {total_urls} URLs to urls.csv")
    except Exception as e:
        logger.error(f"Error saving results to CSV: {e}")


if __name__ == '__main__':
    freeze_support()
    try:
        # Set a reasonable timeout value for the entire script
        start_time = time.time()
        logger.info("Starting web scraping...")
        
        # Use smaller values for testing
        urls_by_system = scrape_base_urls(max_depth=3, max_urls_per_system=50_000)
        
        save_results_to_csv(urls_by_system)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Scraping completed in {elapsed_time:.2f} seconds")
    except KeyboardInterrupt:
        logger.warning("Script interrupted by user")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
