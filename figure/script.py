#import zipfile

#with zipfile.ZipFile('YourZipFile.zip', 'r') as zip_ref:
#    zip_ref.extractall('YourDestinationFolder')

import requests

url = 'https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt'
filename = 'yolov9-c.pt'

# Send a GET request to the URL to download the file
response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Open the file in binary write mode and write the content of the response to it
    with open(filename, 'wb') as f:
        f.write(response.content)
    print(f"File '{filename}' downloaded successfully.")
else:
    print("Failed to download the file.")