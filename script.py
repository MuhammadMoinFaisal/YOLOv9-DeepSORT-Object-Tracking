import zipfile

with zipfile.ZipFile('deep_sort_pytorch.zip', 'r') as zip_ref:
    zip_ref.extractall()