import os
import serpapi
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
api_key = os.getenv('SERPAPI_KEY')
client = serpapi.Client(api_key=api_key)

results = client.search(
    engine="google_play_product",
    product_id="com.dabchy.mobile.dabchyapp",
    store="apps",
    all_reviews="true",
    num=199

)

data = results['reviews']
df = pd.DataFrame(data)
df.to_csv('app_reviews.csv', index=False)