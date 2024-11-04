# Email Analyzer

A Python script to analyze Gmail inbox patterns, including sender statistics and email frequency visualization.

## Setup Instructions

1. **Install Required Packages**

    ```bash
    pip install -r requirements.txt
    ```

2. **Set Up Google Cloud Project**

    1. Go to [Google Cloud Console](https://console.cloud.google.com/)
    2. Create a new project or select an existing one
    3. Enable the Gmail API for your project:
        - Go to "APIs & Services" → "Library"
        - Search for "Gmail API"
        - Click "Enable"

3. **Create OAuth Credentials**

    1. Go to "APIs & Services" → "Credentials"
    2. Click "Create Credentials" → "OAuth client ID"
    3. Choose "Desktop application" as the application type
    4. Give it a name (e.g., "Gmail Analyzer")
    5. Click "Create"
    6. Download the credentials file
    7. Rename the downloaded file to `client_secret.json` and place it in the project root directory

4. **Configure Environment**
    - Ensure `client_secret.json` is in the same directory as `gmail_analysis.py`
    - The first time you run the script, it will:
        - Open your browser for authentication
        - Ask you to sign in to your Google account
        - Request permission to access your Gmail
        - Create a `token.pickle` file for future authentication

## Usage

Run the script:

```bash
python gmail_analysis.py
```

The script will:

-   Fetch the last 500 emails from your Gmail account
-   Analyze sender patterns
-   Generate visualizations:
    -   `top_senders.png`: Top senders by email volume
    -   `top_frequent_senders.png`: Top senders by email frequency

## Output

-   Console output showing top 10 senders and their statistics
-   Two PNG files with visualizations:
    -   Email volume by sender
    -   Email frequency by sender

## Files

-   `gmail_analysis.py`: Main script
-   `requirements.txt`: Required Python packages
-   `client_secret.json`: OAuth credentials (you need to create this)
-   `token.pickle`: Authentication token (created automatically)

## Security Note

-   Keep your `client_secret.json` and `token.pickle` files secure
-   Do not commit these files to version control
-   Add them to your `.gitignore` file
