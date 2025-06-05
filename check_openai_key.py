import requests

API_KEY = "your_openai_api_key_here"  # Replace with your actual OpenAI API key
url = "https://api.openai.com/v1/models"

headers = {"Authorization": f"Bearer {API_KEY}"}

response = requests.get(url, headers=headers)

if response.status_code == 200:
    print("✅ Το API key είναι ενεργό!")
else:
    print(f"❌ Σφάλμα: {response.status_code}")
    print(response.json())
