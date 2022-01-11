from os import stat
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup


driver = webdriver.Chrome()
url = 'https://fresno.craigslist.org/search/sss?auto_bodytype=1&auto_bodytype=10&auto_bodytype=11&auto_bodytype=12&auto_bodytype=2&auto_bodytype=3&auto_bodytype=4&auto_bodytype=5&auto_bodytype=6&auto_bodytype=7&auto_bodytype=8&auto_bodytype=9&condition=40&condition=50&condition=60&lat=37.16118916691665&lon=-118.76287619266607&purveyor-input=all&search_distance=250&sort=date'
driver.get(url)

first_title = driver.find_element_by_class_name('result-heading')
first_title.click()


stats = {}


with open('listings.csv','a') as f:
    # f.write('Model,Brand,Price,Odometer,Paint color,Condition,Fuel,Transmission,Type\n')
    while True:
        print('crawling...')

        try:
            page = driver.page_source
            soup = BeautifulSoup(page,'html.parser')
            price  = soup.find('span',class_='price').text.replace(',','')
            attrs = soup.find_all('p',class_='attrgroup')
            type = attrs[0].find('span').text
        except:
            pass

        stats = {'model': type.split()[0],
        'brand':type.split()[1]
        ,'price':price
        ,}

        spans = attrs[1].find_all('span')

        

        for sp in spans:
            try:
                stats[sp.text[:sp.text.find(':')]] = sp.find('b').text
            except:
                pass

        
        

        try:
            f.write(f'{stats["model"]},{stats["brand"]},{stats["price"]},{stats["odometer"]},{stats["paint color"]},{stats["condition"]},{stats["fuel"]},{stats["transmission"]},{stats["type"]}\n')
        except:
            pass
         


        try:
            driver.find_element_by_class_name('next').click() 
        except:
            driver.refresh()
    
    driver.close()


# comment