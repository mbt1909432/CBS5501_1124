import os
import time
import requests
from bs4 import BeautifulSoup
import re
from pathlib import Path

raw_data_path = Path("raw_data")
raw_data_path.mkdir(exist_ok=True)
processed_data_path=Path("processed_data")
processed_data_path.mkdir(exist_ok=True)
txt_files_content = []


web_name_list = [
    "xue-er", "wei-zheng", "ba-yi", "li-ren", "gong-ye-chang", "yong-ye",
    "shu-er", "tai-bo", "zi-han", "xiang-dang", "xian-jin", "yan-yuan",
    "zi-lu", "xian-wen", "wei-ling-gong", "ji-shi", "yang-huo", "wei-zi",
    "zi-zhang", "yao-yue"
]
print(len(web_name_list))

class web_crawler:

    def __init__(self):
        self.pattern=r'\[Show all\]\s([\s\S]+)\sURN'

    def get_url(self) -> list:
        url_list = list()
        for index in range(0, len(web_name_list)):
            url_list.append(f"https://ctext.org/analects/{web_name_list[index]}/ens")
        return url_list

    # Web Crawling
    def get_content(self,url: str):

        with requests.get(url) as response:
            if response.status_code == 200:
                print(response.text)
                soup = BeautifulSoup(response.text, 'html.parser')

                english_text =""

                for tag in soup.find_all(class_='etext'):
                    if tag.text:
                        english_text+=tag.text.strip()+"\n"

                return english_text
            else:
                print(f"error：{response.status_code}")
    def get_all_chapter_content(self):

        for index,url in enumerate(self.get_url()):
            txt_path = f"{str(raw_data_path)}/{web_name_list[index]}.txt"
            if os.path.exists(txt_path):
                print(web_name_list[index] + "存在")
            else:
                print(url)
                time.sleep(1)
                result = self.get_content(url)
                with open(f"{txt_path}", 'w', encoding='utf-8') as file:
                    file.write(result)


    def save_cleaned_data(self):
        for txt in raw_data_path.glob("*.txt"):
            with open(txt, 'r', encoding='utf-8') as file:
                content = file.read()
                with open(processed_data_path / txt.name, 'w', encoding='utf-8') as f:
                    f.write(re.findall(self.pattern, content)[0])


if __name__=="__main__":

    crawler=web_crawler()
    crawler.get_all_chapter_content()#start to crawl the data from the webpages
    crawler.save_cleaned_data()






