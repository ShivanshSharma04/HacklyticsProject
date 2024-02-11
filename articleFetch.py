"""
Medical Article Retriever
"""
import requests
import json
import time

def getPaperText(tags: list[str])->str:
    """
    """
    idList = fetchRelevantArticles(tags)
    articleJSON, _ = getArticleJSON(idList)
    return getSectionText(articleJSON, ["ABSTRACT"])

def fetchRelevantArticles(tags: list[str])->list[str]:
    """
    Given a list of keywords, returns up to 1000 relevant article IDs
    """
    combinedTags = " ".join(tags)
    combinedTags = combinedTags.replace(" ", "%20")
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&retmode=json&retmax=1000&term={combinedTags}&field=abstract"
    response = requests.get(url)
    return response.json()['esearchresult']['idlist']

def getArticleJSON(idlist:list)->(dict, int):
    idIdx = 0
    while True:
        id = idlist[idIdx]
        api_url = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{id}/unicode"
        response = requests.get(api_url)
        if response.status_code == 404:
            idIdx += 1 #if article ID is not in open access database, continue to next id
            #time.sleep(0.) #for rate limits
            continue
        return response.json(), idIdx

def getSectionText(articleJSON: dict, sectionIDs: list[str] = ["ABSTRACT"], allText: bool = False):
    """
    Returns a section of a pubmed article section text.
    articleJSON: a JSON file retrieved from the pubmed restf API
    section: a string parameter for conclusions. 
        example params: ABSTRACT -> paper abstract
                        CONCL -> paper conclusions
    """
    sections = articleJSON["documents"][0]["passages"]
    outText = []
    sectionIDs = set(sectionIDs)
    for section in sections:
        if allText or section["infons"]["section_type"] in sectionIDs:
            outText.append(section["text"])
    return ' '.join(outText)


#print(getPaperText(["e-health", "diabetes"]))