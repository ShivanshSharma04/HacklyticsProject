"""
Data Collection
"""
import requests
import articleFetch
import time
import csv

def fetchAllArticleIDs(numArticles)->list[str]:
    """
    fetches numArticles # of open-access articles on PMC
    """
    idx = 0
    articleIDs = []
    for idx in range(4):
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pmc&retmode=json&retmax=1000&retstart={idx}&term=free+fulltext%5bfilter%5d"
        response = requests.get(url)
        articleIDs.extend(response.json()['esearchresult']['idlist'])
    return articleIDs

def main():
    articleList = fetchAllArticleIDs(100)
    outList = []
    for idx in range(len(articleList)):
        articleJSON, idx = articleFetch.getArticleJSON(articleList)
        allText = articleFetch.getSectionText(articleJSON, ["ABSTRACT"], True)
        abstractText = articleFetch.getSectionText(articleJSON, ["ABSTRACT"], False)
        outList.append([allText, abstractText])
    with open('summarizedata.csv', 'w') as f: #turn list into csv
        write = csv.writer(f)
        write.writerows(outList)

if __name__ == "__main__":
    main()