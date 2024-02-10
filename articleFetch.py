"""
Medical Article Retriever
"""
import requests
import json

def getPaperText(tags: list[str])->str:
    """
    """
    idList = fetchRelevantArticle(tags)
    articleJSON = getArticleJSON(idList)
    return getSectionText(articleJSON, "ABSTRACT")

def fetchRelevantArticle(tags: list[str])->list[str]:
    """
    Given a list of keywords, returns up to 1000 relevant article IDs
    """
    combinedTags = " ".join(tags)
    combinedTags = combinedTags.replace(" ", "%20")
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&retmode=json&retmax=1000&term={combinedTags}&field=abstract"
    response = requests.get(url)
    return response.json()['esearchresult']['idlist']

def getArticleJSON(idlist:list):
    idIdx = 0
    while True:
        id = idlist[idIdx]
        api_url = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{id}/unicode"
        response = requests.get(api_url)
        if response.status_code == 404:
            idIdx += 1 #if article ID is not in open access database, continue to next id
            
            continue
        return response.json()

def getSectionText(articleJSON: dict, sectionID: str = "ABSTRACT"):
    """
    Returns a section of a pubmed article section text.
    articleJSON: a JSON file retrieved from the pubmed restf API
    section: a string parameter for conclusions. 
        example params: ABSTRACT -> paper abstract
                        CONCL -> paper conclusions
    """
    sections = articleJSON["documents"][0]["passages"]
    outText = []
    for section in sections:
        if section["infons"]["section_type"] == sectionID:
            outText.append(section["text"])
    return ' '.join(outText)


#print(getPaperText(["e-health", "diabetes"]))