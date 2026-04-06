import requests
import sys

URL = "http://localhost:8000/api/v1/chat"

import uuid

# Generate a random session ID for this chat session
SESSION_ID = str(uuid.uuid4())
print(f"🔑 Session ID: {SESSION_ID}")

def chat():
    print("\n" + "="*50)
    print("🤖 Aqar-AI Interactive Terminal (Memory Enabled)")
    print("💬 Type your request in Arabic or English")
    print("🚪 Type 'exit' or 'quit' to stop")
    print("="*50 + "\n")

    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            if not user_input:
                continue
            
            if user_input.lower() in ['exit', 'quit']:
                print("\n👋 Goodbye!")
                break

            print("⏳ Thinking...", end="\r", flush=True)

            try:
                # Send session_id with the request
                payload = {"message": user_input, "session_id": SESSION_ID}
                response = requests.post(URL, json=payload)
                print(" " * 20, end="\r", flush=True) # Clear "Thinking..."
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"🤖 AI: {data['answer']}")
                    
                    if data['properties']:
                        print(f"\n🏘️  Found {len(data['properties'])} Properties:")
                        for i, p in enumerate(data['properties'], 1):
                            print(f"   {i}. {p['title']} | {p['price']:,.0f} EGP | {p['location']}")
                            print(f"      🔗 {p['url']}")
                else:
                    print(f"❌ Error {response.status_code}: {response.text}")

            except requests.exceptions.ConnectionError:
                print("\n❌ Error: Could not connect to server. Is it running?")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break

if __name__ == "__main__":
    chat()
