from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd

chromedriver = Service(ChromeDriverManager().install())
options = Options()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
driver = webdriver.Chrome(service=chromedriver, options=options)

news_data = []

page = 1

while page <= 5:
    
    print(f'Page: {page}')
    
    NEWS_URL = "https://portal.fiocruz.br/noticias?created=All&page="+str(page)
    driver.get(NEWS_URL)
    
    wait = WebDriverWait(driver, 10)
    
    container = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="block-system-main"]/div/div/div[1]/div/div/div/div/div[1]/div[3]')))
    
    links = container.find_elements(By.TAG_NAME, "a")
    
    news_links = []
    
    for link in links:
        a = link.get_attribute('href')
        if "/noticia/" in a:
            news_links.append(a)

    for link in news_links:
        driver.get(link)

        try:
            title = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "titulo-pagina"))).text

            content = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "region-content"))).text

            news_data.append({"title": title, "content": content, "url": link})

            #print(f"Coletado: {title}")

        except Exception as e:
            print(f"Erro ao coletar {link}: {e}")

        #driver.get(NEWS_URL)

    #for news in news_data:
    #    print("\nTítulo:", news["title"])
    #    print("Conteúdo:", news["content"][:500], "...")
    #    print("Link:", news["url"])
        
    page+=1
    
driver.quit()
    
df = pd.DataFrame(news_data)

df.to_csv("noticias.csv", index=False, encoding="utf-8")
