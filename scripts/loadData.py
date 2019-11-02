import os
from bs4 import BeautifulSoup as bs
import json








FILE_PATH = "./data/test/"


print("test")
files = [f for f in os.listdir(FILE_PATH) if os.isfile(f)]
print(files)