import requests
from bs4 import BeautifulSoup
import time
import os
import re
from urllib.parse import urlparse

# --- Selenium Imports ---
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService # For Chrome
from selenium.webdriver.chrome.options import Options as ChromeOptions # For Chrome
# from selenium.webdriver.firefox.service import Service as FirefoxService # For Firefox
# from selenium.webdriver.firefox.options import Options as FirefoxOptions # For Firefox
from webdriver_manager.chrome import ChromeDriverManager # Automates driver download
# from webdriver_manager.firefox import GeckoDriverManager # Automates driver download
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


# --- Script Version ---
SCRIPT_VERSION = "v6_selenium"
print(f"--- Running Xeno-Canto Scraper Version: {SCRIPT_VERSION} ---")

# --- Configuration ---
NUMBER_OF_RECORDINGS_TO_PROCESS = 3 # Start with a small number for testing Selenium
REQUEST_DELAY = 7 # Increased delay as browser automation is slower
SELENIUM_WAIT_TIMEOUT = 20 # How long Selenium should wait for elements

# IMPORTANT: Update these paths if needed.
DOWNLOAD_DIR = "/media/george-vengrovski/Desk SSD/BirdJEPA/xeno_canto_all/files"
SEEN_RECORDINGS_FILE = "/media/george-vengrovski/Desk SSD/BirdJEPA/xeno_canto_all/seen_xeno_canto_ids.txt"

RANDOM_RECORDING_URL = "https://xeno-canto.org/explore/random"
BASE_DOMAIN = "https://xeno-canto.org"

def setup_selenium_driver():
    """Sets up and returns a Selenium WebDriver instance."""
    print("Setting up Selenium WebDriver...")
    try:
        # --- Chrome Setup ---
        chrome_options = ChromeOptions()
        chrome_options.add_argument("--headless")  # Run headless (no GUI)
        chrome_options.add_argument("--no-sandbox") # Bypass OS security model, REQUIRED for Docker/Linux CI
        chrome_options.add_argument("--disable-dev-shm-usage") # Overcome limited resource problems
        chrome_options.add_argument("--disable-gpu") # Applicable to Windows OS only if no GPU available
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        
        # Use webdriver_manager to automatically download and manage ChromeDriver
        service = ChromeService(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # --- Firefox Setup (Alternative) ---
        # firefox_options = FirefoxOptions()
        # firefox_options.add_argument("--headless")
        # firefox_options.add_argument("--disable-gpu") # Optional
        # firefox_options.set_preference("general.useragent.override", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Firefox/89.0")
        # service = FirefoxService(GeckoDriverManager().install())
        # driver = webdriver.Firefox(service=service, options=firefox_options)

        print("Selenium WebDriver setup complete.")
        return driver
    except Exception as e:
        print(f"Error setting up Selenium WebDriver: {e}")
        print("Please ensure you have Google Chrome (or Firefox) installed and that webdriver_manager can download the driver.")
        print("You might need to install the driver manually if webdriver_manager fails.")
        return None

def load_seen_ids():
    if not os.path.exists(SEEN_RECORDINGS_FILE):
        print(f"Seen IDs file not found at: {SEEN_RECORDINGS_FILE}. Starting with an empty set.")
        return set()
    try:
        with open(SEEN_RECORDINGS_FILE, 'r') as f:
            return set(line.strip() for line in f if line.strip())
    except IOError as e:
        print(f"Warning: Could not read seen IDs file '{SEEN_RECORDINGS_FILE}': {e}")
        return set()

def save_seen_id(recording_id, seen_ids_set):
    seen_ids_set.add(recording_id)
    try:
        seen_dir = os.path.dirname(SEEN_RECORDINGS_FILE)
        if seen_dir and not os.path.exists(seen_dir):
            os.makedirs(seen_dir, exist_ok=True)
        with open(SEEN_RECORDINGS_FILE, 'a') as f:
            f.write(recording_id + '\n')
    except Exception as e:
        print(f"Warning: Could not write/create dir for seen IDs file '{SEEN_RECORDINGS_FILE}': {e}")


def fetch_page_with_selenium(driver, url):
    """Fetches a page using Selenium, allowing JS to render."""
    print(f"Fetching page with Selenium: {url}")
    try:
        driver.get(url)
        # Wait for a known element that indicates a recording page has loaded.
        # The player div is a good candidate. Or the title to change from "Random Recordings".
        # Let's wait for the presence of the catalog number link, which is quite specific.
        # Example: <a class="xc-id" href="/123456">XC123456</a>
        # Or wait for the canonical link to be something other than /explore/random
        WebDriverWait(driver, SELENIUM_WAIT_TIMEOUT).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.jp-player[data-xc-id], link[rel='canonical']"))
        )
        
        # Additional check: ensure canonical link is not pointing to /explore/random anymore
        # This is a bit more complex as it requires JS execution to check the DOM property
        # For now, the presence of jp-player or a canonical link should be a good indicator.
        # We can also add a small fixed delay if needed, but explicit waits are better.
        # time.sleep(3) # Small fixed delay as a fallback if explicit waits are tricky

        current_url = driver.current_url
        page_source = driver.page_source
        print(f"Page fetched with Selenium. Current URL: {current_url}")
        return page_source, current_url
    except Exception as e:
        print(f"Error fetching page with Selenium {url}: {e}")
        # driver.save_screenshot('selenium_error_screenshot.png') # Helpful for debugging
        return None, None

def extract_catalog_number_from_page_content(html_content, page_url):
    if not html_content:
        print(f"Debug: HTML content is empty for page {page_url}.")
        return None
        
    soup = BeautifulSoup(html_content, 'html.parser')

    # Strategy 1: Canonical link (should be populated by JS now)
    canonical_tag = soup.find('link', rel='canonical')
    if canonical_tag and canonical_tag.has_attr('href'):
        href = canonical_tag['href']
        # Ensure it's not the random or explore page itself
        if "/explore/random" not in href and "/explore" not in href :
            match = re.search(r'/(\d+)$', href)
            if match:
                print(f"Debug (Selenium): Found ID via canonical link: XC{match.group(1)}")
                return "XC" + match.group(1)

    # Strategy 2: Data attributes (should be populated by JS)
    player_div = soup.find('div', class_='jp-player', attrs={'data-xc-id': True})
    if player_div and player_div.has_attr('data-xc-id'):
        xc_id = player_div['data-xc-id'].strip().upper()
        if xc_id.startswith("XC") and xc_id[2:].isdigit():
            print(f"Debug (Selenium): Found ID via data-xc-id: {xc_id}")
            return xc_id

    # Strategy 3: Check the page title (should be updated by JS)
    title_tag = soup.find('title')
    if title_tag and title_tag.string:
        if "random recordings" not in title_tag.string.lower(): # Check if title changed
            match = re.search(r'(XC\d+)', title_tag.string, re.IGNORECASE)
            if match:
                print(f"Debug (Selenium): Found ID via title: {match.group(1).upper()}")
                return match.group(1).upper()
            
    # Strategy 4: Look for common display patterns of the ID
    id_link = soup.select_one('a.xc-id[href*="/"]') # e.g., <a class="xc-id" href="/123456">XC123456</a>
    if id_link and id_link.has_attr('href'):
        href = id_link['href']
        match_digits = re.search(r'/(\d+)$', href)
        if match_digits:
             id_text = id_link.get_text(strip=True).upper()
             if id_text.startswith("XC") and id_text[2:].isdigit():
                 print(f"Debug (Selenium): Found ID via a.xc-id text and href: {id_text}")
                 return id_text
             print(f"Debug (Selenium): Found ID via a.xc-id href: XC{match_digits.group(1)}")
             return "XC" + match_digits.group(1)


    # Strategy 5: Fallback to parsing from the page_url (Selenium's driver.current_url)
    # This is more reliable with Selenium as driver.current_url should reflect the actual page.
    if "xeno-canto.org" in page_url and not page_url.endswith(("/random", "/explore/", "/explore")):
        url_match_digits = re.search(r'/(\d+)$', page_url)
        if url_match_digits:
            print(f"Debug (Selenium): Found ID via Selenium current_url (digits): XC{url_match_digits.group(1)}")
            return "XC" + url_match_digits.group(1)
        url_match_xc = re.search(r'/(XC\d+)$', page_url, re.IGNORECASE)
        if url_match_xc:
            print(f"Debug (Selenium): Found ID via Selenium current_url (XC): {url_match_xc.group(1).upper()}")
            return url_match_xc.group(1).upper()

    print(f"Debug (Selenium): Catalog number not found on page {page_url}. Start of HTML content (first 500 chars):\n{html_content[:500]}\n[...HTML truncated...]")
    return None

def get_audio_download_link(html_content, page_url_for_relative_resolution):
    if not html_content:
        print(f"Cannot find download link: HTML content is empty for {page_url_for_relative_resolution}")
        return None
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Primary target: <a title="download sound file" href="...">
    download_link_tag = soup.find('a', title='download sound file')
    if download_link_tag and download_link_tag.has_attr('href'):
        href = download_link_tag['href']
        # Resolve URL (it might be relative or protocol-relative)
        if href.startswith('//'):
            return "https:" + href 
        elif href.startswith('/'):
            parsed_page_url = urlparse(page_url_for_relative_resolution)
            return f"{parsed_page_url.scheme}://{parsed_page_url.netloc}{href}"
        elif href.startswith('http://') or href.startswith('https://'):
            return href
        else: 
            # This case for resolving relative paths like "file.mp3" might need context
            parsed_page_url = urlparse(page_url_for_relative_resolution)
            base_path = parsed_page_url.path
            if not base_path.endswith('/'): # Ensure base_path ends with a slash if it's a directory context
                base_path = base_path.rsplit('/',1)[0] + '/'
            return f"{parsed_page_url.scheme}://{parsed_page_url.netloc}{base_path}{href}"

    # Fallback: Look for <audio> tag source
    # Example: <audio id="player_XC123456_0" ...><source src="...?dl=XC123456.mp3"></audio>
    audio_tag = soup.find('audio', id=re.compile(r'player_XC\d+', re.IGNORECASE))
    if audio_tag:
        source_tag = audio_tag.find('source')
        if source_tag and source_tag.has_attr('src'):
            src = source_tag['src']
            # Often these src attributes are direct download links or can be transformed into them
            if src.startswith('//'):
                return "https:" + src
            elif src.startswith('/'):
                 parsed_page_url = urlparse(page_url_for_relative_resolution)
                 return f"{parsed_page_url.scheme}://{parsed_page_url.netloc}{src}"
            elif src.startswith('http://') or src.startswith('https://'):
                return src
            # Sometimes the download link is embedded as a query parameter like "?dl=XC123456.mp3"
            dl_match = re.search(r'\?dl=([^&]+)', src)
            if dl_match:
                filename = dl_match.group(1)
                # We need to construct the full URL. This might be tricky if src is just a path fragment.
                # Assuming it's relative to the domain if it doesn't start with http
                if not src.startswith('http'):
                    return f"{BASE_DOMAIN}/{filename}" # This is a guess, might need refinement

    print(f"Could not find a suitable audio download link on page: {page_url_for_relative_resolution}")
    return None

def download_audio_file(url, catalog_number, directory):
    if not url:
        print(f"No URL provided for download of {catalog_number}.")
        return False
    if not os.path.exists(directory):
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"Created download directory: {directory}")
        except OSError as e:
            print(f"Error creating directory {directory}: {e}")
            return False

    safe_catalog_number = re.sub(r'[^\w\.-]', '_', catalog_number)
    filename = f"{safe_catalog_number}.mp3"
    
    # Try to get a more specific filename from the URL if it looks like one
    try:
        # Check if URL itself ends with .mp3 or has a clear filename part
        parsed_url = urlparse(url)
        path_filename = os.path.basename(parsed_url.path)
        if path_filename and path_filename.lower().endswith(".mp3"):
            potential_filename = path_filename
        else: # Check query parameters like 'dl='
            query_params = dict(qc.split("=") for qc in parsed_url.query.split("&") if "=" in qc)
            if 'dl' in query_params and query_params['dl'].lower().endswith(".mp3"):
                potential_filename = query_params['dl']
            else:
                potential_filename = None

        if potential_filename:
            # Sanitize potential_filename
            clean_potential_filename = re.sub(r'[^\w\.-]', '_', potential_filename)
            # Ensure it still contains the catalog number for consistency, or is clearly related
            if safe_catalog_number.lower() in clean_potential_filename.lower():
                 filename = clean_potential_filename
            else:
                filename = f"{safe_catalog_number}_{clean_potential_filename}" 
    except Exception:
        pass # Stick to default if parsing URL for filename fails

    filepath = os.path.join(directory, filename)

    try:
        print(f"Downloading {catalog_number} from {url} to {filepath}...")
        # Use requests for downloading, it's generally better for file downloads than Selenium
        audio_response = requests.get(url, stream=True, timeout=60, headers={'User-Agent': 'Mozilla/5.0'}) 
        audio_response.raise_for_status()
        with open(filepath, 'wb') as f:
            for chunk in audio_response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Successfully downloaded {filepath}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading audio file {url}: {e}")
    except IOError as e:
        print(f"Error saving audio file {filepath}: {e}")
    return False

def main():
    for path_to_check in [DOWNLOAD_DIR, os.path.dirname(SEEN_RECORDINGS_FILE)]:
        if path_to_check and not os.path.exists(path_to_check):
            try:
                os.makedirs(path_to_check, exist_ok=True)
                print(f"Ensured directory exists: {path_to_check}")
            except OSError as e:
                print(f"Critical Error: Could not create/access directory {path_to_check}: {e}. Please check permissions and path.")
                return
        elif path_to_check and not os.access(path_to_check, os.W_OK):
             print(f"Critical Error: Directory {path_to_check} is not writable. Please check permissions.")
             return

    seen_ids = load_seen_ids()
    print(f"Loaded {len(seen_ids)} previously seen recording IDs from {SEEN_RECORDINGS_FILE}.")
    
    driver = setup_selenium_driver()
    if not driver:
        print("Failed to initialize Selenium WebDriver. Exiting.")
        return

    downloaded_count = 0
    attempt_count = 0

    print(f"Attempting to process {NUMBER_OF_RECORDINGS_TO_PROCESS} random recordings using Selenium.")

    try:
        for i in range(NUMBER_OF_RECORDINGS_TO_PROCESS):
            attempt_count += 1
            print(f"\n--- Attempt {attempt_count} of {NUMBER_OF_RECORDINGS_TO_PROCESS} ---")
            
            page_html, final_url = fetch_page_with_selenium(driver, RANDOM_RECORDING_URL)

            if not page_html or not final_url:
                print("Failed to fetch page content with Selenium. Skipping this attempt.")
                if i < NUMBER_OF_RECORDINGS_TO_PROCESS - 1:
                     time.sleep(REQUEST_DELAY) # Still delay between attempts
                continue
            
            catalog_number = extract_catalog_number_from_page_content(page_html, final_url)

            if not catalog_number:
                print(f"Skipping attempt for {final_url} as catalog number extraction failed.")
                if i < NUMBER_OF_RECORDINGS_TO_PROCESS - 1:
                     time.sleep(REQUEST_DELAY)
                continue
            
            print(f"Found recording ID: {catalog_number} (URL: {final_url})")

            if catalog_number in seen_ids:
                print(f"Recording {catalog_number} has already been processed/downloaded. Skipping.")
            else:
                print(f"Recording {catalog_number} is new. Attempting to download...")
                audio_download_url = get_audio_download_link(page_html, final_url)

                if audio_download_url:
                    print(f"Found audio download link: {audio_download_url}")
                    if download_audio_file(audio_download_url, catalog_number, DOWNLOAD_DIR):
                        save_seen_id(catalog_number, seen_ids)
                        downloaded_count += 1
                    else:
                        print(f"Failed to download audio for {catalog_number}.")
                else:
                    print(f"Could not find audio download link for {catalog_number} on page {final_url}.")
                    save_seen_id(catalog_number, seen_ids) 

            if i < NUMBER_OF_RECORDINGS_TO_PROCESS - 1:
                print(f"Waiting for {REQUEST_DELAY} seconds before next Selenium fetch...")
                time.sleep(REQUEST_DELAY)
    finally:
        if driver:
            print("Closing Selenium WebDriver.")
            driver.quit()

    print(f"\n--- Processing Complete ---")
    print(f"Script Version: {SCRIPT_VERSION}")
    print(f"Total attempts: {attempt_count}")
    print(f"New recordings downloaded in this session: {downloaded_count}")
    print(f"Total unique recordings processed: {len(seen_ids)}")
    print(f"Audio files are saved in: {os.path.abspath(DOWNLOAD_DIR)}")
    print(f"List of processed IDs is in: {os.path.abspath(SEEN_RECORDINGS_FILE)}")

if __name__ == "__main__":
    main()
