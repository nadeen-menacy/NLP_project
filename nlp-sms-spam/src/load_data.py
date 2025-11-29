import pandas as pd
import os
import ssl
import urllib.request

def download_dataset(save_path="data/sms_spam.csv"):
    """Downloads the SMS Spam Collection dataset and saves it locally."""
    os.makedirs("data", exist_ok=True)
    url = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv"
    
    # Handle SSL certificate verification issues
    try:
        df = pd.read_csv(url, sep='\t', names=['label', 'text'])
    except urllib.error.URLError as e:
        if 'CERTIFICATE_VERIFY_FAILED' in str(e):
            print("⚠️  SSL certificate issue detected. Downloading with SSL verification disabled...")
            # Create unverified SSL context
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            # Download with custom SSL context
            with urllib.request.urlopen(url, context=ssl_context) as response:
                df = pd.read_csv(response, sep='\t', names=['label', 'text'])
        else:
            raise
    
    df.to_csv(save_path, index=False)
    print(f"✅ Dataset saved to {save_path} ({len(df)} rows)")
    return df

if __name__ == "__main__":
    df = download_dataset()
    print(df.head())
