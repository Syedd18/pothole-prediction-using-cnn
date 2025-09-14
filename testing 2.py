import requests

url = "http://127.0.0.1:8000/predict"

# Use raw string for Windows path
image_path = r"C:\Users\acer\Downloads\pothole-detection-system-using-convolution-neural-networks-master\pothole-detection-system-using-convolution-neural-networks-master\My Dataset\train\Pothole\p 248.jpg"

files = {"image": open(image_path, "rb")}
response = requests.post(url, files=files)
print(response.json())
