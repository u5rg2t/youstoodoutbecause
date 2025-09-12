# Business Outreach Automation Script (Advanced Crawler & Analyser)
#
# This script reads an Excel file, crawls each company's website (up to 15 pages),
# uses the Gemini AI for a multi-criteria website evaluation, generates a
# deeply informed reason for outreach, and creates a personalized email.
#
# Author: Scott Murray (with assistance from Gemini)
#
# How to Use:
# 1. Save this script as 'outreach_script.py'.
# 2. Ensure 'requirements.txt' is in the same directory.
# 3. Install/update libraries: pip install -r requirements.txt
# 4. Get a Gemini API key: https://aistudio.google.com/app/apikey
# 5. Run the script: python outreach_script.py
# 6. Follow prompts. Results are saved to 'outreach_results.csv'.

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

# --- Configuration ---
# The EMAIL_TEMPLATE has been removed as it's no longer needed.

GEMINI_MODEL_NAME = 'gemini-2.5-pro'

# --- Core Functions ---

async def crawl_website(session, base_url):
    """
    Asynchronously crawls a website starting from the base_url, collecting text from internal pages.
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

async def get_website_evaluation(company_name, website_text):
    """
    Uses Gemini API to evaluate the website based on several criteria.
    """
    if not website_text:
        return {"error": "No text to analyse."}

    try:
        system_instruction = """You are a website quality analyst. Your task is to evaluate a company's website based on the provided text content. Provide a score from 1-10 for each criterion. Return your response ONLY as a valid JSON object with the keys "clarity", "professionalism", and "credibility"."""
        
        prompt = f"""
        Analyse the text content from the website of "{company_name}". Based on this text, score the site on the following criteria (1=Poor, 10=Excellent):
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

async def get_reason_to_be_impressed(company_name, website_text):
    """
    Uses the Gemini API to analyze full website text and generate a reason
    for being impressed with the company.
    """
    if not website_text:
        return config["default_error_reason"]

    try:
        system_instruction = """You are an M&A analyst working for Scott Murray, an individual investor looking to buy a great company for the long term. Your task is to find the single most compelling strength of a company based on its website text. The phrase you generate will be used in an email directly to the company's owner. Therefore, you MUST write in the second person (using "your" instead of "their" or "the company's"). Focus on aspects that indicate a stable, well-run business: a strong legacy, a clear market niche, loyal customers, or a unique, high-quality product/service. Ignore generic marketing fluff."""

        # A more direct prompt for the AI
        # Create a dynamic prompt using the sentence template from the config file
        sentence_template = config.get("sentence", "Your company, {{ company_name }}, stood out to me {{ generated_text }}")
        prompt_sentence = sentence_template.replace("{{ company_name }}", company_name).replace("{{ generated_text }}", "[THE PHRASE YOU GENERATE]")

        prompt = f"""
        Analyse the text from the website of a company named "{company_name}".
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
        sentence_template = config.get("sentence", "Your company, {{ company_name }}, stood out to me {{ generated_text }}")
        template_prefix = sentence_template.split('{{ generated_text }}')[0].replace("{{ company_name }}", company_name).strip()
        
        if reason.lower().startswith(template_prefix.lower()):
            reason = reason[len(template_prefix):].strip()

        # Ensure the first letter is lowercase to fit into the sentence
        if reason:
            return reason[0].lower() + reason[1:]
        return reason
    except Exception as e:
        print(f"    - Gemini API Error (Reason): {e}")
        return config["default_error_reason"]

async def process_row(session, row_data):
    """
    Asynchronously processes a single row from the DataFrame.
    """
    index, row, name_col, web_col, total_rows = row_data
    company_name = str(row[name_col])
    website_url = str(row[web_col])

    print(f"\n[{index + 1}/{total_rows}] Processing: {company_name} ({website_url})")

    if pd.isna(company_name) or pd.isna(website_url) or website_url.lower() in ['nan', '']:
        print("    - Skipping row due to missing name or website.")
        return None
    
    # Check if 'generated_reason' exists and is a non-empty string
    reason_value = row.get('generated_reason')
    if pd.notna(reason_value) and isinstance(reason_value, str) and reason_value.strip() != "":
        print("    - Skipping row, 'generated_reason' already contains a value.")
        return {'status': 'Skipped', 'pages_crawled': 0, 'url': website_url}

    # 1. Crawl the entire website
    full_site_text, pages_crawled = await crawl_website(session, website_url)
    
    # 2. Get website evaluation and reason concurrently
    evaluation_task = get_website_evaluation(company_name, full_site_text)
    reason_task = get_reason_to_be_impressed(company_name, full_site_text)
    
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

async def process_file(file_path, name_col, web_col, api_key):
    """
    Main processing function to orchestrate reading the file,
    analyzing data asynchronously, and updating the file.
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
            row_data = (index, row, name_col, web_col, total_rows)
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

def load_config():
    global config
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print("Error: 'config.json' not found. Please create it.")
        exit()
    except json.JSONDecodeError:
        print("Error: 'config.json' is not a valid JSON file.")
        exit()

def write_log_file(stats, duration):
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
        
        # Ensure the logs directory exists
        log_folder = 'logs'
        if not os.path.isdir(log_folder):
            os.makedirs(log_folder)
            
        log_file_path = os.path.join(log_folder, 'log.txt')
        with open(log_file_path, 'w') as f:
            f.write(log_content)
        print(f"✅ Summary log has been saved to '{log_file_path}'.")

if __name__ == "__main__":
    load_dotenv()
    load_config()
    print("--- Business Outreach Automation Script ---")
    start_time = time.time()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        try:
            api_key = getpass.getpass("Please enter your Gemini API Key: ")
        except Exception as e:
            print(f"Could not read API key: {e}")
            exit()

    xls_folder = 'xls'
    if not os.path.isdir(xls_folder):
        print(f"Error: The '{xls_folder}' directory was not found.")
        exit()

    excel_files = [f for f in os.listdir(xls_folder) if f.endswith(('.xls', '.xlsx'))]

    if len(excel_files) == 0:
        print(f"No Excel files found in the '{xls_folder}' directory.")
        exit()
    elif len(excel_files) > 1:
        print(f"Multiple Excel files found in the '{xls_folder}' directory. Please ensure there is only one.")
        exit()
    
    file_path = os.path.join(xls_folder, excel_files[0])
    print(f"Processing file: {file_path}")

    company_name_column = config.get("company_name_column", "Business Name")
    website_column = config.get("website_column", "Web Address")
    print(f"Using '{company_name_column}' for company names and '{website_column}' for website URLs.")

    # On Windows, the default event loop policy can cause issues with aiohttp.
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    stats = asyncio.run(process_file(file_path, company_name_column, website_column, api_key))
    
    end_time = time.time()
    duration = end_time - start_time
    
    if stats:
        write_log_file(stats, duration)




