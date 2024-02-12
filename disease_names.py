import requests
from bs4 import BeautifulSoup

def get_disease_names():
    # URL of the Wikipedia page
    url = 'https://en.wikipedia.org/wiki/List_of_infectious_diseases'

    # Fetch the content from the URL
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the table that contains the list of diseases
    # Assuming it is the first table in the article
    table = soup.find('table', {'class': 'wikitable'})

    # Initialize lists to store the infection agents and common names
    infection_agents = []
    common_names = []

    # Iterate through table rows to extract the information
    for row in table.find_all('tr')[1:]:  # Skipping the header row
        cells = row.find_all('td')
        if cells:  # Checking if row contains cells to avoid empty rows
            # Infection agent is in the first column
            infection_agent = cells[0].text.strip()
            infection_agents.append(infection_agent)
            
            # Common name is in the second column, check if it exists
            if len(cells) > 1:
                common_name = cells[1].text.strip()
                common_names.append(common_name)
            else:
                common_names.append("N/A")  # For rows where the common name might not exist
    names = common_names + infection_agents
    return names

