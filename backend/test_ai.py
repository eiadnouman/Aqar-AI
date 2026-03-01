import requests
import json
import time

URL = "http://localhost:8000/v1/chat"

def test_query(query):
    print(f"\nSending Query: '{query}'")
    try:
        start = time.time()
        response = requests.post(URL, json={"message": query})
        end = time.time()
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Response ({end-start:.2f}s):")
            print("-" * 40)
            print(f"🤖 Answer: {data['answer']}")
            print("-" * 40)
            if data['properties']:
                print(f"🏘️ Found {len(data['properties'])} Properties:")
                for p in data['properties']:
                    print(f"   - {p['title']} ({p['price']:,.0f} EGP) in {p['location']}")
            else:
                print("ℹ️ No properties returned (Conversation mode)")
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"❌ Connection Failed: {e}")

# Wait for server to be up
print("Waiting for server...")
time.sleep(5)

# Test 1: Vague Request (Should ask for clarification)
test_query("عايز شقة")

# Test 2: Specific Request (Should show results)
test_query("عايز شقة في التجمع بـ 3 مليون")
