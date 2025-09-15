# Business Outreach Automation Script
#
# This script automates the process of business outreach by:
# 1. Reading a list of companies from an Excel file.
# 2. Asynchronously crawling each company's website to gather text content.
# 3. Leveraging the Gemini AI to perform a qualitative analysis of the website.
# 4. Generating a unique, context-aware reason for outreach based on the analysis.
# 5. Updating the original Excel file with the analysis scores and generated reason.
# 6. Generating a log file with a summary of the script's execution.
#
# Author: Scott Murray
#
# --- How to Use ---
# 1.  **Configuration**: Create and configure `config.json` with necessary parameters
#     (e.g., column names, AI model settings).
# 2.  **Input File**: Place the target Excel file (containing company names and websites)
#     in the `xls` directory.
# 3.  **API Key**: Set your Gemini API key in a `.env` file as `GEMINI_API_KEY`.
# 4.  **Dependencies**: Install required libraries using `pip install -r requirements.txt`.
# 5.  **Execution**: Run the script from your terminal using `python main.py`.
# 6.  **Results**: The script updates the Excel file in place with the new data.
#     A detailed log is saved in the `logs` directory.

import pandas as pd
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
import getpass
import time
import os
from urllib.parse import urljoin, urlparse
import json
import asyncio
import aiohttp
from collections import deque
from dotenv import load_dotenv
import re

# --- Global Configuration ---
# The GEMINI_MODEL_NAME specifies the model to be used for AI-driven analysis.
# This can be updated to any compatible model offered by the Gemini API.

GEMINI_MODEL_NAME = 'gemini-2.5-pro'

# --- Utility Functions ---

def is_valid_url(url: str) -> bool:
    """
    Validates if the given string is a well-formed URL.

    Args:
        url: The string to validate.

    Returns:
        True if the URL is valid, False otherwise.
    """
    if not isinstance(url, str):
        return False
    # Regex to check for a valid URL pattern (simplified for this use case)
    regex = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return re.match(regex, url) is not None

# --- Core Functions ---

async def crawl_website(session: aiohttp.ClientSession, base_url: str) -> tuple[str, int]:
    """
    Asynchronously crawls a website to extract text content from its internal pages.

    This function starts at the provided base URL and systematically navigates through
    internal links, collecting text while avoiding external sites, duplicates, and non-HTML content.
    It respects server load by including a small delay between requests.

    Args:
        session: An active aiohttp.ClientSession for making HTTP requests.
        base_url: The starting URL for the crawl.

    Returns:
        A tuple containing:
        - The aggregated text content from all crawled pages (truncated to 50,000 chars).
        - The total number of unique pages visited.
    """
    if not base_url.startswith(('http://', 'https://')):
        base_url = 'https://' + base_url

    domain_name = urlparse(base_url).netloc
    urls_to_visit = deque([base_url])
    visited_urls = {base_url}
    full_text = ""
    pages_crawled = 0
    max_pages = config.get("max_pages_to_crawl", 15)

    print(f"    - Starting crawl of {base_url} (up to {max_pages} pages)")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

    while urls_to_visit and pages_crawled < max_pages:
        url = urls_to_visit.popleft()
        print(f"      - Crawling: {url}")
        pages_crawled += 1

        try:
            async with session.get(url, headers=headers, timeout=10, allow_redirects=True) as response:
                response.raise_for_status()
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')

                # Extract text
                for script_or_style in soup(['script', 'style', 'nav', 'footer']):
                    script_or_style.decompose()
                
                if soup.body:
                    text = soup.body.get_text(separator=' ', strip=True)
                    full_text += ' '.join(text.split()) + " "

                # Find new internal links to visit
                for link in soup.find_all('a', href=True):
                    absolute_link = urljoin(url, link['href'])
                    parsed_link = urlparse(absolute_link)
                    
                    # Stay on the same domain and avoid mailto, tel, etc.
                    if parsed_link.netloc == domain_name and parsed_link.scheme in ['http', 'https'] and absolute_link not in visited_urls:
                        visited_urls.add(absolute_link)
                        urls_to_visit.append(absolute_link)

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            print(f"        - Could not fetch {url}: {e}")
        except Exception as e:
            print(f"        - Error processing {url}: {e}")
        
        await asyncio.sleep(0.5) # Be respectful to the server

    print(f"    - Crawl finished. Visited {len(visited_urls)} pages.")
    return full_text[:50000], len(visited_urls)

async def get_website_evaluation(website_text: str) -> dict:
    """
    Evaluates website content using the Gemini AI for qualitative analysis.

    This function sends the website's text content to the Gemini API and asks it to
    score the site on clarity, professionalism, and credibility. The AI is instructed
    to return a clean JSON object, which this function parses and returns.

    Args:
        website_text: The aggregated text content from the company's website.

    Returns:
        A dictionary containing the evaluation scores (clarity, professionalism,
        credibility) or an error message if the analysis failed.
    """
    if not website_text:
        return {"error": "No text to analyse."}

    try:
        system_instruction = """You are a website quality analyst. Your task is to evaluate a company's website based on the provided text content. Provide a score from 1-10 for each criterion. Return your response ONLY as a valid JSON object with the keys "clarity", "professionalism", and "credibility"."""
        
        prompt = f"""
        Analyse the text content from the website provided. Based on this text, score the site on the following criteria (1=Poor, 10=Excellent):
        1. Clarity of Value Proposition: How clearly do they explain their products/services and who they are for?
        2. Professionalism & Design Impression: Based on the language and structure, what is the impression of the site's professionalism?
        3. Trust & Credibility: Is there evidence of trust signals like case studies, testimonials, clear history, or specific expertise?

        Website Text:
        ---
        {website_text}
        ---
        """
        model = genai.GenerativeModel(GEMINI_MODEL_NAME, system_instruction=system_instruction)
        response = await model.generate_content_async(prompt)
        
        # Clean the response to ensure it's valid JSON
        cleaned_response = response.text.strip().replace('```json', '').replace('```', '')
        return json.loads(cleaned_response)

    except Exception as e:
        print(f"    - Gemini API Error (Evaluation): {e}")
        return {"clarity": 0, "professionalism": 0, "credibility": 0, "error": str(e)}

async def get_reason_to_be_impressed(website_text: str) -> str:
    """
    Generates a personalized and compelling reason for outreach using the Gemini AI.

    This function provides the AI with the website text and a specific system prompt
    to act as an M&A analyst. It crafts a unique phrase that completes a pre-defined
    sentence, focusing on the company's strengths like market niche, legacy, or quality.
    The output is carefully cleaned to ensure it fits grammatically into the outreach email.

    Args:
        website_text: The aggregated text content from the company's website.

    Returns:
        A concise, AI-generated phrase to be used as the reason for outreach,
        or a default error message if generation fails.
    """
    if not website_text:
        return config["default_error_reason"]

    try:
        system_instruction = """You are an M&A analyst working for Scott Murray, an individual investor looking to buy a great company for the long term. Your task is to find the single most compelling strength of a company based on its website text. The phrase you generate will be used in an email directly to the company's owner. Therefore, you MUST write in the second person (using "your" instead of "their" or "the company's"). Focus on aspects that indicate a stable, well-run business: a strong legacy, a clear market niche, loyal customers, or a unique, high-quality product/service. Ignore generic marketing fluff."""

        # A more direct prompt for the AI
        # Create a dynamic prompt using the sentence template from the config file
        sentence_template = config.get("sentence", "This website stood out to me {{ generated_text }}")
        prompt_sentence = sentence_template.replace("{{ generated_text }}", "[THE PHRASE YOU GENERATE]")

        prompt = f"""
        Analyse the text from the website provided.
        Your task is to generate a concise, professional phrase that grammatically completes the following sentence:
        "{prompt_sentence}"

        Based ONLY on the website text, generate the phrase that replaces "[THE PHRASE YOU GENERATE]".

        IMPORTANT:
        - The phrase MUST be written in the second person (e.g., "your").
        - The final sentence must be grammatically correct and natural-sounding.
        - DO NOT repeat the beginning of the sentence in your response. ONLY provide the replacement phrase.

        Example phrases to generate:
        - "because of your clear commitment to using sustainable, locally-sourced materials."
        - "due to your long-standing reputation serving the community for over 20 years."
        - "for your innovative approach to solving complex engineering challenges."
        - "and the impressive portfolio of client success stories on your site."

        Website Text:
        ---
        {website_text}
        ---
        """
        model = genai.GenerativeModel(GEMINI_MODEL_NAME, system_instruction=system_instruction)
        response = await model.generate_content_async(prompt)
        reason = response.text.strip()

        # Robustly strip the template prefix if the AI includes it anyway
        sentence_template = config.get("sentence", "This website stood out to me {{ generated_text }}")
        template_prefix = sentence_template.split('{{ generated_text }}')[0].strip()
        
        if reason.lower().startswith(template_prefix.lower()):
            reason = reason[len(template_prefix):].strip()

        # Ensure the first letter is lowercase to fit into the sentence
        if reason:
            return reason[0].lower() + reason[1:]
        return reason
    except Exception as e:
        print(f"    - Gemini API Error (Reason): {e}")
        return config["default_error_reason"]

async def process_row(session: aiohttp.ClientSession, row_data: tuple) -> dict | None:
    """
    Orchestrates the end-to-end processing for a single company row from the input file.

    This function handles the entire workflow for one company:
    1. Skips processing if essential data (name, URL) is missing or if already processed.
    2. Calls `crawl_website` to fetch website content.
    3. Concurrently calls `get_website_evaluation` and `get_reason_to_be_impressed`.
    4. Compiles all results into a dictionary for updating the main DataFrame.

    Args:
        session: The active aiohttp.ClientSession.
        row_data: A tuple containing the row's index, data, column names, and total row count.

    Returns:
        A dictionary with all the generated data (scores, reason, status) for the row,
        or None if the row was skipped.
    """
    index, row, web_col, total_rows = row_data
    website_url = str(row[web_col])

    print(f"\n[{index + 1}/{total_rows}] Processing: {website_url}")

    # Validate the URL before proceeding
    if not is_valid_url(website_url):
        print(f"    - Skipping row due to invalid or missing URL: {website_url}")
        return {'status': 'Skipped', 'reason': 'Invalid URL', 'url': website_url}
    
    # Check if 'generated_reason' exists and is a non-empty string
    reason_value = row.get('generated_reason')
    if pd.notna(reason_value) and isinstance(reason_value, str) and reason_value.strip() != "":
        print("    - Skipping row, 'generated_reason' already contains a value.")
        return {'status': 'Skipped', 'pages_crawled': 0, 'url': website_url}

    # 1. Crawl the entire website
    full_site_text, pages_crawled = await crawl_website(session, website_url)
    
    # 2. Get website evaluation and reason concurrently
    evaluation_task = get_website_evaluation(full_site_text)
    reason_task = get_reason_to_be_impressed(full_site_text)
    
    evaluation, reason_phrase = await asyncio.gather(evaluation_task, reason_task)
    
    # Determine status and error reason
    status = "OK"
    error_reason = None
    if "error" in evaluation:
        status = "Error"
        error_reason = evaluation.get("error", "Unknown evaluation error")
    elif reason_phrase == config["default_error_reason"]:
        status = "Error"
        error_reason = "AI could not generate a reason."

    print(f"    - AI Evaluation: {evaluation}")
    print(f"    - AI-Generated Phrase: {reason_phrase}")
    print(f"    - Status: {status}")

    # 4. Return results to be updated in the DataFrame
    return {
        'index': index,
        'eval_clarity': evaluation.get('clarity', 0),
        'eval_professionalism': evaluation.get('professionalism', 0),
        'eval_credibility': evaluation.get('credibility', 0),
        'generated_reason': reason_phrase, # Save only the generated phrase
        'status': status,
        'pages_crawled': pages_crawled,
        'url': website_url,
        'error_reason': error_reason
    }

async def process_file(file_path: str, web_col: str, api_key: str) -> dict:
    """
    Main controller function to manage the entire file processing workflow.

    This function reads the specified Excel file, sets up the asynchronous processing
    tasks for each row, executes them, and then updates the DataFrame with the results.
    Finally, it saves the updated DataFrame back to the original Excel file.

    Args:
        file_path: The full path to the Excel file to be processed.
        web_col: The column name for the company's website URL.
        api_key: The Gemini API key for authentication.

    Returns:
        A dictionary containing statistics about the completed run, such as the number
        of processed rows, errors, and pages scanned.
    """
    try:
        engine = 'openpyxl' if file_path.endswith('xlsx') else 'xlrd'
        df = pd.read_excel(file_path, engine=engine)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    new_cols = ['eval_clarity', 'eval_professionalism', 'eval_credibility', 'generated_reason', 'status']
    for col in new_cols:
        if col not in df.columns:
            df[col] = None

    genai.configure(api_key=api_key)
    
    tasks = []
    total_rows = len(df)
    async with aiohttp.ClientSession() as session:
        for index, row in df.iterrows():
            row_data = (index, row, web_col, total_rows)
            tasks.append(process_row(session, row_data))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)

    # Update DataFrame with results and collect stats
    update_count = 0
    error_count = 0
    total_pages_scanned = 0
    error_details = []

    for result in results:
        if isinstance(result, Exception):
            error_count += 1
            # Attempt to get URL from task if possible, though it's complex
            error_details.append({'url': 'Unknown', 'reason': str(result)})
            continue
        
        if result and result['status'] != 'Skipped':
            total_pages_scanned += result.get('pages_crawled', 0)
            if result['status'] == 'Error':
                error_count += 1
                error_details.append({'url': result['url'], 'reason': result.get('error_reason', 'Unknown')})
            
            # Update DataFrame
            idx = result['index']
            df.loc[idx, 'eval_clarity'] = result['eval_clarity']
            df.loc[idx, 'eval_professionalism'] = result['eval_professionalism']
            df.loc[idx, 'eval_credibility'] = result['eval_credibility']
            df.loc[idx, 'generated_reason'] = result['generated_reason']
            df.loc[idx, 'status'] = result['status']
            update_count += 1
    
    print(f"\nProcessed {len(results)} rows. Updated {update_count} rows in the DataFrame.")

    try:
        if update_count > 0:
            df.to_excel(file_path, index=False, engine='openpyxl')
            print(f"✅ Successfully saved updates to '{file_path}'.")
        else:
            print("No rows were updated, so the file was not saved.")
    except Exception as e:
        print(f"\n❌ Error saving updated file: {e}")
        print("Your results may not have been saved.")
    
    return {
        'total_processed': len(results),
        'successful_additions': update_count - error_count,
        'error_count': error_count,
        'total_pages_scanned': total_pages_scanned,
        'error_details': error_details
    }


# --- Main Execution ---

config = {}

def load_config() -> None:
    """
    Loads script settings from the `config.json` file into the global `config` dictionary.

    This function is critical for externalizing configuration, allowing users to change
    parameters like column names, AI settings, and crawl depth without modifying the script's code.
    It includes error handling for a missing or malformed JSON file.
    """
    global config
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print("Error: 'config.json' not found. Please create it.")
        exit(1)
    except json.JSONDecodeError:
        print("Error: 'config.json' is not a valid JSON file.")
        exit(1)

def write_log_file(stats: dict, duration: float) -> None:
    """
    Generates and saves a detailed log file summarizing the script's execution.

    The log provides a snapshot of the run, including performance metrics and a
    detailed breakdown of any errors encountered. This is essential for debugging
    and tracking the script's effectiveness over time.

    Args:
        stats: A dictionary of statistics collected during the `process_file` execution.
        duration: The total time in seconds the script took to run.
    """
    log_content = f"""
--- Script Execution Summary ---
Date: {time.strftime("%Y-%m-%d %H:%M:%S")}
Total Run Time: {duration:.2f} seconds

--- Statistics ---
Total Companies Processed: {stats['total_processed']}
Successful Additions: {stats['successful_additions']}
Errors: {stats['error_count']}
Total Pages Scanned: {stats['total_pages_scanned']}
"""
    if stats['error_count'] > 0:
        log_content += "\n--- Error Details ---\n"
        for error in stats['error_details']:
            log_content += f"- URL: {error['url']}\n  Reason: {error['reason']}\n"

    # Ensure the logs directory exists before writing the file
    log_folder = 'logs'
    os.makedirs(log_folder, exist_ok=True)

    log_file_path = os.path.join(log_folder, f"log_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    with open(log_file_path, 'w') as f:
        f.write(log_content)
    print(f"✅ Summary log has been saved to '{log_file_path}'.")

def main():
    """
    Main execution block of the script.

    This function initializes the script by loading environment variables and configuration,
    validating the environment (e.g., checking for the input file), and orchestrating
    the primary workflow by calling `process_file`. It also handles API key retrieval
    and measures total execution time.
    """
    load_dotenv()  # Load environment variables from .env file
    load_config()
    print("--- Business Outreach Automation Script ---")
    start_time = time.time()

    # Securely retrieve the API key from environment variables or prompt the user
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        try:
            api_key = getpass.getpass("Please enter your Gemini API Key: ")
        except Exception as e:
            print(f"Could not read API key: {e}")
            exit(1)

    # Validate the input directory and ensure a single Excel file is present
    xls_folder = 'xls'
    if not os.path.isdir(xls_folder):
        print(f"Error: The '{xls_folder}' directory was not found. Please create it.")
        exit(1)

    excel_files = [f for f in os.listdir(xls_folder) if f.endswith(('.xls', '.xlsx'))]

    if len(excel_files) == 0:
        print(f"No Excel files found in the '{xls_folder}' directory.")
        exit(1)
    elif len(excel_files) > 1:
        print(f"Multiple Excel files found in '{xls_folder}'. Please ensure only one exists.")
        exit(1)

    file_path = os.path.join(xls_folder, excel_files[0])
    print(f"Processing file: {file_path}")

    # Load column names from config, with sensible defaults
    website_column = config.get("website_column", "Web Address")
    print(f"Using column for website URLs: '{website_column}'")

    # Set the event loop policy for Windows to avoid common asyncio errors with aiohttp
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Run the main asynchronous processing function
    stats = asyncio.run(process_file(file_path, website_column, api_key))

    # Calculate and display execution time and write the final log file
    duration = time.time() - start_time
    print(f"\n--- Script Finished in {duration:.2f} seconds ---")
    if stats:
        write_log_file(stats, duration)

if __name__ == "__main__":
    main()




