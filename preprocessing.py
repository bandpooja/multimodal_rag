import os
import subprocess
import requests
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
from tqdm import tqdm
import pandas as pd


class DataPreprocessor:
    def __init__(self, csv_file, save_folder='content/image_data'):
        self.csv_file = csv_file
        self.df = pd.read_csv(csv_file)
        self.save_folder = save_folder
        self.image_list = []
        self.texts = []
        self.final_images = []
        self.final_text = []
        
        # Ensure the save folder exists
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

    def fetch_image_urls(self, url_list):
        """Fetches image URLs from a list of URLs using subprocess and BeautifulSoup"""
        image_urls = []
        for url in tqdm(url_list):
            result = subprocess.run(["curl", "-s", url], capture_output=True, text=True)
            if result.returncode == 0:
                soup = BeautifulSoup(result.stdout, "html.parser")
                images = soup.find_all("img")
                img_urls = [img['src'] for img in images if 'src' in img.attrs]
                if img_urls:
                    image_urls.append(img_urls[0])  # Add first image URL from the page
                else:
                    image_urls.append("None")
            else:
                image_urls.append("None")
        return image_urls

    def process_data(self):
        """Main method to process data, extract text, and save images."""
        image_urls = self.fetch_image_urls(self.df["url"])

        for idx, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            image_url = image_urls[idx]
            if image_url != "None":
                self.image_list.append(image_url)
                u_id = row["product_id"]
                title = row["title"]
                description = row["description"]
                available = row["availability"]
                price = row["price"]
                brand = row["brand"]

                page_content = f"title: {title}, description: {description}, u_id: {u_id}, available: {available}, img_url: {image_url}, price: {price}, brand: {brand}"
                self.texts.append(page_content)

        self.download_images()

    def download_images(self):
        """Download and save images from the URLs."""
        for idx, img_url in enumerate(self.image_list):
            try:
                response = requests.get(img_url)
                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content))
                    img_filename = os.path.join(self.save_folder, f'image_{idx + 1}.jpg')
                    self.final_images.append(img_filename)
                    self.final_text.append(self.texts[idx])

                    # Save the image
                    img.save(img_filename)
                    print(f"Image {idx + 1} saved to {img_filename}")
                else:
                    print(f"Failed to fetch image from {img_url}. Status code: {response.status_code}")
            except Exception as e:
                print(f"Error while saving image from {img_url}: {e}")
                continue

    def save_results(self, text_output_file='content/text_data.txt'):
        """Save the final text data to a file."""
        with open(text_output_file, 'w') as f:
            for text in self.final_text:
                f.write(text + "\n")

def main():
    csv_file = "data.csv"  # Path to your CSV file
    preprocessor = DataPreprocessor(csv_file)

    # Process data and download images
    preprocessor.process_data()

    # Save the results to a text file
    preprocessor.save_results()

if __name__ == "__main__":
    main()
