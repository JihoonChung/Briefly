import pandas as pd
from newspaper import Article
import datetime
import os
import uuid

def scrape_to_csv(url, category="Uncategorized", csv_path="articles.csv"):
    # Generate an unique UUID
    article_id = str(uuid.uuid4())

    article = Article(url)
    try:
        article.download()
        article.parse()
    except Exception as e:
        print(f"Failed to download or parse article: {e}")
        return None
    
    # Extract details
    date_of_article = article.publish_date.strftime('%Y-%m-%d') if article.publish_date else datetime.datetime.now().strftime('%Y-%m-%d')
    title = article.title
    body = article.text
    
    # Create df
    data = {
        "ID": [article_id],
        "Date of Article": [date_of_article],
        "Category": [category],
        "Title": [title],
        "Body": [body],
        "URL": [url]
    }
    df = pd.DataFrame(data)
    
    # Check if CSV file exists and handle dupes
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        
        # Check if URL already exists
        if url in existing_df['URL'].values:
            print("Duplicate article found based on URL. Skipping...")
            return None

        # Append
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else: # Create a new CSV 
        df.to_csv(csv_path, index=False)
    
    print(f"Article '{title}' saved to {csv_path}.")
    return df

if __name__ == "__main__":
    url = "https://globalnews.ca/news/10863638/donald-trump-tom-homan-border-czar/"  # Replace with an actual URL
    category = "Politics"
    filename = "articles.csv"
    scrape_to_csv(url, category, filename)
