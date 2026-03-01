import requests
import json
import time

URL = "http://localhost:8000/v1/chat"
SESSION_ID = "test_user_123"

def send(msg):
    print(f"\n👤 Sending: '{msg}'")
    try:
        res = requests.post(URL, json={"message": msg, "session_id": SESSION_ID})
        if res.status_code == 200:
            data = res.json()
            print(f"🤖 Answer: {data['answer'][:100]}...")
            if data['properties']:
                print(f"🏘️  Properties: {len(data['properties'])}")
                for p in data['properties']:
                    print(f"   - {p['title']} ({p['price']:,.0f}) in {p['location']}")
            else:
                print("ℹ️  No properties returned.")
        else:
            print(f"❌ Error {res.status_code}: {res.text}")
    except Exception as e:
        print(f"❌ Connection Error: {e}")

print("Waiting for server...")
time.sleep(5)

# 1. First Turn: Location
send("عايز شقة في التجمع")

# 2. Second Turn: Price (Should remember Tagamo from Turn 1)
send("في حدود 5 مليون")

# 3. Third Turn: Bedrooms (Should remember Tagamo + 5M)
send("تكون 3 غرف")
