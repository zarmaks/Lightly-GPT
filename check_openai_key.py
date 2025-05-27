import os
import requests

API_KEY = "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"

url = "https://api.openai.com/v1/models"

headers = {
    "Authorization": f"Bearer {API_KEY}"
}

response = requests.get(url, headers=headers)

if response.status_code == 200:
    print("✅ Το API key είναι ενεργό!")
else:
    print(f"❌ Σφάλμα: {response.status_code}")
    print(response.json())