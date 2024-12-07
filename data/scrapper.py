import os
import time
import json
import logging
import csv

import undetected_chromedriver as uc
from typing import Optional, Any

from urllib.parse import urljoin

from datetime import datetime, timezone, timedelta


from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement  import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from selenium.common.exceptions import TimeoutException,NoSuchElementException


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Scrapper")

class VietLottScrapper:
    def __init__(self, params:list=[]):
        self.driver = None
        self.url = 'https://www.ketquadientoan.com/'
        self.endUrl = '.html'
        self.mega645_url='https://www.ketquadientoan.com/tat-ca-ky-xo-so-mega-6-45.html'
        self.power655_url='https://www.ketquadientoan.com/tat-ca-ky-xo-so-power-655.html'
        now_ts = datetime.now(timezone(timedelta(hours=7)))
        self.date_now=now_ts.strftime("%d-%m-%Y")
        self.params =params


    def __enter__(self):
        chrome_options = uc.ChromeOptions()
        chrome_options.add_argument("--disable-popup-blocking")
        # chrome_options.add_argument("--headless")

        self.driver = uc.Chrome(use_subprocess=True,options=chrome_options)
        return self
    def __exit__(self, exc_type, exc_value, _):
        logger.info("Existing from the browser...")
        if exc_type:
            logger.error(f"An exception occurred:{exc_value}")
        
        if self.driver:
            self.driver.close()

        return False
    
    def run(self):
        self.crawler645()
        logger.info("Crawler data Mega 6-45")

        self.crawler655()
    
    def crawler645(self):
        date_from= '17-11-2019'
        if not self.driver:
            msg = "Please use context for the `Scrapper`"
            logger.error(msg)
            raise ValueError(msg)

        url= self.mega645_url+'?datef='+date_from+'&datet='+self.date_now

        logger.info(f"URL HISTORY MEGA 6/45: {url}")

        driver = self.driver
        driver.get(url)
        driver.switch_to.window(driver.window_handles[0])
        results=[
            ["Day","Type","Money","Date","Month","Year","Num1","Num2","Num3","Num4","Num5","NumS"]
        ]
        try:
            pageData= WebDriverWait(driver,30).until(
                EC.presence_of_all_elements_located(
                    (
                        By.XPATH,
                        ('//div[@id="noidung"]/table[@class="table-mini-result"]/tbody/tr')
                    ),
                )
            )
            for row in pageData:
                day,datestr =row.find_element(By.XPATH,'.//td[1]').text.split(',')
                date,month,year = datestr.split('/')
                num1,num2,num3,num4,num5,nums =row.find_elements(By.XPATH,'.//td[2]/span')
                value=row.find_element(By.XPATH,'.//td[3]').text.replace(',','.')

                results.append([
                   day,
                   "Mega",
                   str(value),
                   int(date),
                   int(month),
                   int(year),
                   int(num1.text),
                   int(num2.text),
                   int(num3.text),
                   int(num4.text),
                   int(num5.text),
                   int(nums.text),
                ])
            if len(results):
                self.setToCsvFile(results,type="Mega")

        except NoSuchElementException as exception:
            logger.info(f"Can't found element power 6/45: {url}")
            logger.error(exception)
            pass   
        except TimeoutException:
            logger.info(f"Can't found history power 6/45: {url}")
            pass


    def crawler655(self):
        date_from= '17-11-2019'
        if not self.driver:
            msg = "Please use context for the `Scrapper`"
            logger.error(msg)
            raise ValueError(msg)

        url= self.power655_url+'?datef='+date_from+'&datet='+self.date_now

        logger.info(f"URL HISTORY POWER 6/55: {url}")

        driver = self.driver
        driver.get(url)
        driver.switch_to.window(driver.window_handles[0])
        results=[
            ["Day","Type","Money","Date","Month","Year","Num1","Num2","Num3","Num4","Num5","Num6","NumS"]
        ]
        try:
            pageData= WebDriverWait(driver,60).until(
                EC.presence_of_all_elements_located(
                    (
                        By.XPATH,
                        ('//div[@id="noidung"]/table[@class="table-mini-result power-mini"]/tbody/tr')
                    ),
                )
            )
            for row in pageData:
                day,datestr =row.find_element(By.XPATH,'.//td[1]').text.split(',')
                date,month,year = datestr.split('/')
                num1,num2,num3,num4,num5,num6,nums =row.find_elements(By.XPATH,'.//td[2]/span')
                value=row.find_element(By.XPATH,'.//td[3]').text.replace(',','.')

                results.append([
                   day,
                   "Power",
                   str(value),
                   int(date),
                   int(month),
                   int(year),
                   int(num1.text),
                   int(num2.text),
                   int(num3.text),
                   int(num4.text),
                   int(num5.text),
                   int(num6.text),
                   int(nums.text),
                ])
            if len(results):
                self.setToCsvFile(results,type="Power")

        except NoSuchElementException as exception:
            logger.info(f"Can't found element mega 6/55: {url}")
            logger.error(exception)
            pass   
        except TimeoutException:
            logger.info(f"Can't found history mega 6/55: {url}")
            pass
        
    def setToCsvFile(self, data:list=[], type:str ="Mega"):
        try:
            if type == "Power":
                pathFile = os.path.dirname(os.path.abspath(__file__))+'/results/power.csv'
            else:
                pathFile = os.path.dirname(os.path.abspath(__file__))+'/results/mega.csv'

            with open(pathFile,mode='w',newline='', encoding='utf-8')as file:
                writer = csv.writer(file)
                writer.writerows(data)
                logger.info(f"Write new file {type}.csv completed!")
        except Exception as e:
            logger.error(e)
            pass   
        
        