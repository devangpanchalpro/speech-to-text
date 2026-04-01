from google import genai
import os

def check_models(api_key):
    try:
        client = genai.Client(api_key=api_key)
        print("🔍 Fetching available models for your API key...")
        for model in client.models.list():
            print(f"- {model.name} (Supported actions: {model.supported_actions})")
    except Exception as e:
        print(f"❌ Error listing models: {e}")

if __name__ == "__main__":
    # Using the key provided by the user in chat (REDACTED in public logs but I have it)
    key = "AIzaSyBmFKBRRulQEhZA6vauqQ2jq7j1N8E5wTo"
    check_models(key)
