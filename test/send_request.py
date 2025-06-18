import requests
import base64

API_URL = "http://localhost:8314/predict"  # hoặc IP nếu deploy từ xa
IMAGE_PATH = "/home/truongnn/chaos/code/repo/medical_inferneces/examples/test_1.png"

# Step 1: Encode image to base64
with open(IMAGE_PATH, "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

# Step 2: Prepare request payload
payload = {
    "images": encoded_image,
    "text": "Ảnh X-quang này có gì bất thường?",
    "model_name": "clara"
}

# Step 3: Send POST request
response = requests.post(API_URL, json=payload)

# Step 4: Print response
if response.status_code == 200:
    print("🧠 Model Output:\n", response.json()["outputs"])
else:
    print("❌ Error:", response.status_code, response.text)
