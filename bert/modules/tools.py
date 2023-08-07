import os


def bookscorpus_download(store_pth: str = './data'):
    url = 'https://t.co/J3EaSEgwW0'
    download_command = f'wget -O {store_pth}/bookscorpus.tar.gz {url}'
    os.makedirs(store_pth, exist_ok=True)
    os.system(download_command)
    os.makedirs(f'{store_pth}/bookscorpus/', exist_ok=True)
    os.system(f'tar -xzvf {store_pth}/bookscorpus.tar.gz -C {store_pth}/')

