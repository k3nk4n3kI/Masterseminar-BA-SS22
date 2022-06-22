import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
# from fake_useragent import UserAgent
import pandas as pd
import numpy as np
import time
import concurrent.futures
from multiprocessing import freeze_support
import random

# Lists

modell_list = []
skin_name_list = []
group_list = []
zustand_list = []
price_list = []
price_sug_list = []
rating_stars_list = []
rating_vot_list = []
kollektion_list = []
msv_list = []
lk_list = []
seltenheit_list = []
trade_list = []
usage_list = []
sticker_title_list0 = []
sticker_title_list1 = []
sticker_title_list2 = []
sticker_title_list3 = []
sticker_status_list0 = []
sticker_status_list1 = []
sticker_status_list2 = []
sticker_status_list3 = []
discount_list =[]
Farberverlauf_list =[]
namensschild_list=[]
extras_list=[]


# Siehe https://stackoverflow.com/questions/49565042/way-to-change-google-chrome-user-agent-in-selenium/49565254#49565254


#-----------API Request for all available Items-----------

# siehe:https://docs.skinport.com/?python#items


r = requests.get("https://api.skinport.com/v1/items").json()

print(r)

#-----------Getting links for all available skin groups-----------

all_links=[]

for index in range(len(r)):
    for key in r[index]:
        if key == "market_page":
            if r[index][key] in all_links:
                continue
            all_links.append(r[index][key])


#-----------Getting links for all indidividual Item-----------

links = []

#Siehe https://medium.com/analytics-vidhya/the-art-of-not-getting-blocked-how-i-used-selenium-python-to-scrape-facebook-and-tiktok-fd6b31dbe85f


from selenium. common. exceptions import TimeoutException, WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.proxy import Proxy, ProxyType
from random_user_agent.user_agent import UserAgent
from random_user_agent.params import SoftwareName, OperatingSystem
from time import sleep
import logging
import traceback
import sys

class Request:
    logger = logging.getLogger('django.project.requests')
    selenium_retries = 0
    
    def __init__(self, url):
        self.url = url
        
    def get_selenium_res(self, class_name):
        try:
            software_names = [SoftwareName.CHROME.value]
            operating_systems=[OperatingSystem.WINDOWS.value,
                               OperatingSystem.LINUX.value]
            
            user_agent_rotator=UserAgent(software_names=software_names,
                                         operating_systems=operating_systems,
                                         Limit=100)
            user_agent=user_agent_rotator.get_random_user_agent()
            
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--window-size=1420,1080')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument(f'user-agent={user_agent}')
            # chrome_options.binary_location("/Applications/Google Chrome.app")
            browser = webdriver.Chrome(executable_path="/Users/lars/Downloads/chromedriver", chrome_options=chrome_options)
            browser.get(self.url)
            time_to_wait = 1.5
            try:
                WebDriverWait(browser, time_to_wait).until(
                EC.presence_of_element_located((By.CLASS_NAME,class_name)))
            finally:
                browser.maximize_window()
                SCROLL_PAUSE_TIME = 0.5
                
                # Get scroll height
                last_height = browser.execute_script("return document.body.scrollHeight")
                
                while True:
                    # Scroll down to bottom
                    browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                
                    # Wait to load page
                    time.sleep(SCROLL_PAUSE_TIME)
                
                    # Calculate new scroll height and compare with last scroll height
                    new_height = browser.execute_script("return document.body.scrollHeight")
                    if new_height == last_height:
                        break
                    last_height = new_height
                page_html = browser.page_source
                soup = BeautifulSoup(page_html, "html.parser")
                for link in soup.find_all('a', class_="ItemPreview-link"):
                    links.append("https://skinport.com" + link["href"])
                browser.close()
        except (TimeoutException, WebDriverException):
            self.logger.error(traceback.format_exc())
            sleep(3)
            self.selenium_retries += 1
            self.logger.info('Selenium retry #: ' + str(self.selenium_retries))
            return self.get_selenium_res(class_name)


for i in all_links:
    Request(i).get_selenium_res('MessageContainer')
    


#--------------------


with open("links1.txt", "w") as output:
    output.write(str(links))


#---------------LINKS-----------

a = open("/Users/lars/Documents/Uni/Seminar_BA/links1.txt", "r").read()
b = a.replace("'", '').replace(' ', '').strip("[]")
#b = b.split().remove("'")
test_links = b.split(",")


#------Selenium Request-----------

proxy = ['31.187.66.99:1212', '192.36.168.109:1212', '194.126.237.84:1212', '194.132.107.86:1212', '31.187.66.33:1212', '194.126.237.52:1212', '193.17.219.35:1212','31.187.66.189:1212', 
'193.180.244.30:1212', '31.187.66.77:1212', '31.187.66.108:1212', '31.187.66.54:1212', '193.181.15.90:1212', '193.17.219.30:1212', '194.126.237.113:1212', '31.187.66.232:1212', '193.181.15.104:1212', '194.126.237.74:1212', 
'31.187.66.163:1212', '194.126.237.35:1212', '193.180.244.42:1212', '31.187.66.86:1212', '194.126.237.51:1212', '193.17.219.18:1212', '31.187.66.127:1212','193.17.219.17:1212']


software_names = [SoftwareName.CHROME.value]
operating_systems=[OperatingSystem.WINDOWS.value,
                   OperatingSystem.LINUX.value]

user_agent_rotator=UserAgent(software_names=software_names,
                             operating_systems=operating_systems,
                             Limit=100)


options = Options()
options.add_argument("--headless")
options.add_argument('--no-sandbox')
options.add_argument('--window-size=1420,1080')
options.add_argument('--disable-gpu')


#-----------Scraping each offer-----------

def get_data(url):
    link = url.split("/item")
    link2 = (link[0] + "/de/item" + link[1])
    print(link2)
    
    proxies = random.choice(proxy)
    user_agent=user_agent_rotator.get_random_user_agent()
    options.add_argument(f'user-agent={user_agent}')
    options.add_argument('--proxy-server=http://' + proxies)
    browser = webdriver.Chrome(executable_path="/Users/lars/Downloads/chromedriver", options=options)


    # For Loop Starts
    
    browser.get(link2)
    time_to_wait = 2
    class_name = "ItemList-header"
    try:
        WebDriverWait(browser, time_to_wait).until(
        EC.presence_of_element_located((By.CLASS_NAME,class_name)))
    finally:
        browser.maximize_window()
        page_html = browser.page_source
        # Scraping Website
        soup = BeautifulSoup(page_html, "html.parser")
        ItemPage_Info = soup.find_all("div", class_="ItemPage-info") # left side of Item Page
        ItemPage_Item = soup.find_all("div", class_="ItemPage-item") # right side of Item Page
        for value in ItemPage_Info:
            try:
                Price = value.find("div", class_="Tooltip-link").text
                Price_Sug = value.find("div", class_="ItemPage-suggested").text
                Discount = value.find("div", class_="ItemPage-discount")
                if Discount is None:
                    discount_list.append("0")
                else:
                    Discount = Discount.text
                    Discount = Discount.split()
                    Discount_new = Discount[1].split("%")
                    discount_list.append(Discount_new[0])
            except AttributeError:
                Price = None
                continue
            try:
                Extras = value.find("span", class_="ItemPage-title").text
                if "Souvenir" in Extras:
                    Extras = Extras.split()
                    print(Extras)
                    extras_list.append(Extras[0])
                elif "StatTrak" in Extras:
                    Extras = Extras.split()
                    print(Extras)
                    extras_list.append(Extras[0])
                else:
                    extras_list.append("0")
            except AttributeError:
                extras_list.append("0")
                print("Fehler")
                
            Skin_Name = value.find("span", class_="ItemPage-name").text
            if "Farbverlauf" in Skin_Name:
                farbe = Skin_Name.split("(")
                farbe_sole = farbe[1].split("%")
                Farberverlauf_list.append(farbe_sole[0])
            else:
                Farberverlauf_list.append(0)
            Zustand = value.find("div", class_="ItemPage-text").text
            
                
            Rating_Stars = value.find("div", class_="StarsRating-rating").text
            Rating_vot = value.find("div", class_="StarsRating-votes").text
            Kollektion = value.find("div", class_="ItemDetailTable-value").text
            # Mustervorlage = value.find("div", class_="Tooltip-link").text
            Mustervorlage = value.find_all(class_="ItemDetailTable-value ItemDetailTable-value--tooltip")
            try:
                MSV = Mustervorlage[0].text
            except IndexError:
                MSV = 0
            try:
                LK = Mustervorlage[1].text
            except IndexError:
                LK = 0
            #print(LK)
            #print(MSV)
            #Lackkatalog = Mustervorlage.next_sibling
            Seltenheit = value.find("div", class_="ItemDetailTable-value ItemDetailTable-value--color").text
            Namenschild = value.find_all("div" ,class_="ItemDetailTable-key")
            try:
                if "Namensschild" in Namenschild[3]:
                    namensschild_list.append(1)
                else:
                    namensschild_list.append(0)
            except IndexError:
                namensschild_list.append(0)
                
            # Checking for Stickers included and append to specific list
            
            stickers = value.find_all("div", class_="ItemPage-sticker")
            try:
                if stickers[0] is not None:
                    Sticker_Titel0 = stickers[0].find("div", class_="ItemPage-stickerTitle").text
                    sticker_title_list0.append(1)
                    Sticker_Status = stickers[0].find("div", class_="ItemPage-stickerStatus").text
                    sticker_status_list0.append(Sticker_Status)
            except IndexError:
                sticker_title_list0.append(0)
                sticker_status_list0.append(None)
            try:
                if stickers[1] is not None:
                    Sticker_Titel1 = stickers[1].find("div", class_="ItemPage-stickerTitle").text
                    sticker_title_list1.append(1)
                    Sticker_Status = stickers[1].find("div", class_="ItemPage-stickerStatus").text
                    sticker_status_list1.append(Sticker_Status)
            except IndexError:
                sticker_title_list1.append(0)
                sticker_status_list1.append(None)
            try:
                if stickers[2] is not None:
                    Sticker_Titel2 = stickers[2].find("div", class_="ItemPage-stickerTitle").text
                    sticker_title_list2.append(1)
                    Sticker_Status = stickers[2].find("div", class_="ItemPage-stickerStatus").text
                    sticker_status_list2.append(Sticker_Status)
            except IndexError:
                sticker_title_list2.append(0)
                sticker_status_list2.append(None)
            try:
                if stickers[3] is not None:
                    Sticker_Titel3 = stickers[3].find("div", class_="ItemPage-stickerTitle").text
                    sticker_title_list3.append(1)
                    Sticker_Status = stickers[3].find("div", class_="ItemPage-stickerStatus").text
                    sticker_status_list3.append(Sticker_Status)
            except IndexError:
                sticker_title_list3.append(0)
                sticker_status_list3.append(None)
                
            #Prepare rest of Data for Lists
            
            Zustand = Zustand.split()
            Price = Price.split()
            Price_Sug = Price_Sug.split()
            #Discount = Discount.split()
            Rating_Stars = Rating_Stars.split()
            Rating_vot = Rating_vot.split()
            Kollektion = Kollektion.split("„")
            
            # adding to lists
    
            zustand_list.append(Zustand[0])
            price_list.append(Price[0])    
            price_sug_list.append(Price_Sug[2])
            # discount_list.append(Discount[1])
            if Rating_Stars[0] == "Noch":
                rating_stars_list.append(0)
            else:
                rating_stars_list.append(Rating_Stars[0])
            rating_vot_list.append(Rating_vot[0])
            try:
                kollektion_list.append(Kollektion[1])
            except IndexError:
                kollektion_list.append(None)
            msv_list.append(MSV)
            lk_list.append(LK)
            seltenheit_list.append(Seltenheit)
            
        #------Extracting Infos from ItemPage_Item------
    
        for value2 in ItemPage_Item:
            if Price == None:
                continue
            categorie = value2.find_all("a", class_="Breadcrumbs-item")
            #print(categorie)
            group = categorie[1].text
            group_list.append(group)
            modell = categorie[2].text
            modell_list.append(modell)
            try:
                skin_name = categorie[3].text
                skin_name_list.append(skin_name)
            except IndexError:
                skin_name_list.append(0)
            Trade = value2.find("div", class_="ItemPage-lock").text
            if "Handelbar" in Trade:
                trade_list.append(0)
            elif "Minuten" in Trade:
                Trade = Trade.split()
                Trade = pd.to_numeric(Trade[1])
                Trade = Trade/1440
                trade_list.append(Trade)
            elif "Stunden" in Trade:
                Trade =Trade.split()
                Trade = pd.to_numeric(Trade[1])
                Trade = Trade/24
                trade_list.append(Trade)
            else:
                Trade = Trade.split()
                print(Trade)
                Trade = pd.to_numeric(Trade[1])
                trade_list.append(Trade)
            try:
                Usage = value2.find("div", class_="WearBar-value").text
                usage_list.append(Usage)
            except AttributeError:
                usage_list.append(0)
        
        #------Saving Data Local if system crashes------
        
        with open("modell.txt", "w") as output:
            output.write(str(modell_list))
        with open("skin_name.txt", "w") as output:
            output.write(str(skin_name_list))
        with open("group.txt", "w") as output:
            output.write(str(group_list))
        with open("zustand.txt", "w") as output:
            output.write(str(zustand_list))
        with open("price.txt", "w") as output:
            output.write(str(price_list))
        with open("price_sug.txt", "w") as output:
            output.write(str(price_sug_list))
        with open("stars.txt", "w") as output:
            output.write(str(rating_stars_list))
        with open("vot.txt", "w") as output:
            output.write(str(rating_vot_list))
        with open("kollektion.txt", "w") as output:
            output.write(str(kollektion_list))
        with open("MSV.txt", "w") as output:
            output.write(str(msv_list))
        with open("LK.txt", "w") as output:
            output.write(str(lk_list))
        with open("rarity.txt", "w") as output:
            output.write(str(seltenheit_list))
        with open("Trade.txt", "w") as output:
            output.write(str(trade_list))
        with open("Usage.txt", "w") as output:
            output.write(str(usage_list))
        with open("Sticker_Titel0.txt", "w") as output:
            output.write(str(sticker_title_list0))
        with open("Sticker_Title1.txt", "w") as output:
            output.write(str(sticker_title_list1))
        with open("Sticker_Title2.txt", "w") as output:
            output.write(str(sticker_title_list2))
        with open("Sticker_Title3.txt", "w") as output:
            output.write(str(sticker_title_list3))
        with open("Sticker_Status0.txt", "w") as output:
            output.write(str(sticker_status_list0))
        with open("Sticker_Status1.txt", "w") as output:
            output.write(str(sticker_status_list1))
        with open("Sticker_Status2.txt", "w") as output:
            output.write(str(sticker_status_list2))
        with open("Sticker_Status3.txt", "w") as output:
            output.write(str(sticker_status_list3))
        with open("Discount.txt", "w") as output:
            output.write(str(discount_list))
        with open("Farbverlauf.txt", "w") as output:
            output.write(str(Farberverlauf_list))
        with open("Namensschild.txt", "w") as output:
            output.write(str(namensschild_list))
        with open("Extras.txt", "w") as output:
            output.write(str(extras_list))
        
        
        print(len(extras_list))
        browser.close()


def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=9) as executor:
        test = test_links[94000:110000]
        result = executor.map(get_data, test)

if __name__ == "__main__":
    freeze_support()
    main()

#Data to DF

df_dic = {'Group':group_list, 'Modell':modell_list, 'Name':skin_name_list, 'Condition':zustand_list, 'Usage':usage_list, 'Price':price_list, 'Price Suggested':price_sug_list, 'Discount':discount_list,
'Collection':kollektion_list, 'Extras':extras_list, 'Rarity':seltenheit_list, 'Tradable':trade_list, 'Stars':rating_stars_list, 'Numbe Votes':rating_vot_list,
'Farberverlauf':Farberverlauf_list, 'Lackkatalog':lk_list, 'Mustervorlage':msv_list, 'Nametag':namensschild_list, 'Sticker1 Status':sticker_status_list0, 'Sticker2 Status': sticker_status_list1,'Sticker3 Status':sticker_status_list2,
'Sticker4 Status':sticker_status_list3, 'Sticker1':sticker_title_list0, 'Sticker2':sticker_title_list1,
'Sticker3':sticker_title_list2, 'Sticker4':sticker_title_list3}



df = pd.DataFrame(df_dic)
print(df)










