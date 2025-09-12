# Business Outreach Automation Script

This script is an advanced tool designed to automate the initial stages of business outreach. It takes a list of companies from an Excel file, crawls their websites to gather intelligence, uses the Gemini 2.5 Pro AI to perform a qualitative analysis, and generates a personalized reason for outreach.

The entire process is asynchronous, allowing it to process multiple websites concurrently for high efficiency.

## How It Works

1.  **File Detection**: The script automatically finds the target Excel file (`.xls` or `.xlsx`) inside the `xls` directory.
2.  **Configuration**: It reads settings from `config.json` to determine which columns to use for the company name and website, how many pages to crawl, and what default messages to use.
3.  **Crawling**: For each company, the script crawls up to a specified number of pages on their website, extracting clean text content.
4.  **AI Analysis**: The extracted text is sent to the Gemini AI for two concurrent tasks:
    *   **Website Evaluation**: Scores the website on clarity, professionalism, and credibility.
    *   **Reason Generation**: Generates a concise, professional, and context-aware phrase that completes the outreach sentence defined in the config.
5.  **Update Excel**: The original Excel file is updated in place with new columns containing the AI's evaluation scores, a status (`OK` or `Error`), and the generated outreach phrase.
6.  **Logging**: A detailed summary of the entire run is saved to `logs/log.txt`, including statistics on successes, errors, and total pages scanned.

## Requirements

*   Python 3.7+
*   A Google Gemini API Key.

## Getting Started

Follow these steps to set up and run the script.

### 1. Clone the Repository

```bash
git clone https://github.com/u5rg2t/youstoodoutbecause.git
cd youstoodoutbecause
```

### 2. Set Up the Virtual Environment

It is highly recommended to use a Python virtual environment to manage dependencies.

```bash
# Create the virtual environment
python3 -m venv venv

# Activate it (on macOS/Linux)
source venv/bin/activate

# On Windows, use:
# venv\Scripts\activate
```

### 3. Install Dependencies

Install all the required Python packages from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4. Configure the Script

There are two essential configuration files:

**A. `.env` file (for your API Key)**

Create a file named `.env` in the root of the project and add your Gemini API key to it:

```
GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"
```

**B. `config.json` file (for script settings)**

This file controls the script's behavior. You can modify it to suit your needs.

```json
{
  "company_name_column": "Business Name",
  "website_column": "Web Address",
  "default_error_reason": "because I was unable to analyze the website content.",
  "max_pages_to_crawl": 15,
  "sentence": "Your company stood out to me {{ generated_text }}"
}
```

*   `company_name_column`: The exact name of the column in your Excel file that contains the company names.
*   `website_column`: The exact name of the column for the website URLs.
*   `max_pages_to_crawl`: The maximum number of pages the script will crawl on each website.
*   `sentence`: The template for your outreach message. The script will generate text to replace the `{{ generated_text }}` token.

### 5. Add Your Excel File

Place your Excel file (either `.xls` or `.xlsx`) inside the `xls` directory. The script is designed to process **only one file** at a time.

### 6. Run the Script

Execute the script from your terminal:

```bash
python3 main.py
```

The script will process all companies in the file and update the Excel sheet with the new data. A log of the operation will be saved in the `logs` directory.