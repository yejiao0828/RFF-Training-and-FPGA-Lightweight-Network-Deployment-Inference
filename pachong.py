import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re

# 基础 URL 和目标目录
url = 'https://research.engr.oregonstate.edu/hamdaoui/RFFP-dataset/LoRa-Dataset/Diff_Days_Indoor_Setup/Day4/Device17/'

# 本地保存文件的根目录
local_dir = 'C:/Lora_data_Day4/Device17'

# 创建本地目录
if not os.path.exists(local_dir):
    os.makedirs(local_dir)

# 获取网页内容
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# 获取所有文件的链接，过滤掉包含查询参数的链接
file_links = []
for link in soup.find_all('a'):
    href = link.get('href')
    # 过滤掉无效的链接（如 Parent Directory 和包含查询参数的链接）
    if href and not href.startswith('..') and '?' not in href:
        file_links.append(href)

# 下载文件
for link in file_links:
    # 拼接完整的文件URL
    file_url = urljoin(url, link)

    # 计算本地保存路径
    local_path = local_dir

    # 确保目标文件夹存在
    if not os.path.exists(os.path.dirname(local_path)):
         os.makedirs(os.path.dirname(local_path))

    # 下载文件
    print(f'Downloading {file_url} to {local_path}')
    file_response = requests.get(file_url)
    with open(local_path, 'wb') as f:
        f.write(file_response.content)

print("Download complete.")
