import requests
from bs4 import BeautifulSoup

pares = []
links = ["https://worldpopulationreview.com/cities/iran","https://worldpopulationreview.com/cities/venezuela","https://worldpopulationreview.com/cities/equatorial-guinea","https://worldpopulationreview.com/cities/honduras","https://worldpopulationreview.com/cities/nepal","https://worldpopulationreview.com/cities/madagascar","https://worldpopulationreview.com/cities/cuba","https://worldpopulationreview.com/cities/haiti","https://worldpopulationreview.com/cities/nicaragua"]

for link in links:
    URL = link
    page = requests.get(URL)

    soup = BeautifulSoup(page.content, "html.parser")

    ciudades = soup.find_all("tr", class_="table-row hover:bg-gray-50")

    for ciudad in ciudades:
        ciudades1 = ciudad.find_all("td")
        paresdict = {
            "Ciudad": ciudades1[0].text,
            "Poblacion": int(ciudades1[1].text.replace(',',''))
        }
        pares.append(paresdict)

def myFunc(e):
    return e['Poblacion']
    
pares.sort(reverse=True, key=myFunc)

cont = 0

while(cont < 19):
    print(pares[cont])
    cont = cont + 1
