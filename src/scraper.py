import requests
from bs4 import BeautifulSoup
import pandas as pd

# Function to fetch headlines from a website
def fetch_headlines(url, class_name, tag_name='h3'):
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch {url}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    headlines = [tag.get_text(strip=True) for tag in soup.find_all(tag_name, class_=class_name)]
    return headlines

# Function to get financial headlines from a source
def get_financial_headlines(source_url, headline_class):
    print(f"Fetching headlines from: {source_url}")
    headlines = fetch_headlines(source_url, headline_class)
    return headlines

if __name__ == "__main__":
    # Example: Yahoo Finance headlines
    url = 'https://finance.yahoo.com/topic/stock-market-news/'
    class_name = "clamp yf-18q3fnf"
    headlines = get_financial_headlines(url, class_name)

    # Save headlines to CSV
    df = pd.DataFrame({'headline': headlines})
    df.to_csv('data/yahoo_finance_headlines.csv', index=False)
