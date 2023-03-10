import requests


import base64
from io import BytesIO

# pip3 install pillow
from PIL import Image

# 若img.save()报错 cannot write mode RGBA as JPEG
# 则img = Image.open(image_path).convert('RGB')


def image_to_base64(image_path):
    img = Image.open(image_path)
    output_buffer = BytesIO()
    img.save(output_buffer, format='PNG')
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)
    return base64_str.decode('ascii')


API_URL = "https://kmollee-is-cat.hf.space/run/predict"
SAMPLE_IMAGE_PATH = "./images/cat.jpg"


sample = "data:image/png;base64,{}".format(image_to_base64(SAMPLE_IMAGE_PATH))

# print("sample:",sample)
response = requests.post(API_URL, json={
    "data": [sample,
             ]
}).json()

print(response)

data = response["data"]
print(data)
