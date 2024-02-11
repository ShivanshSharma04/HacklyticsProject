from transformers import T5Tokenizer, T5ForConditionalGeneration
import articleFetch

def breakText(text, n): 
    """
    Break string into list of n-word long strings
    """
    textList = text.split(" ")
    # looping till length l 
    for i in range(0, len(textList), n):  
        yield " ".join(textList[i:i + n]) 
  


def getArticleSummary(tags: list[str]):
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained("t5-base")

    article = articleFetch.getPaperText(tags)
    maxWordLength = 300
    if len(article.split(" ")) > maxWordLength:
        articleList = list(breakText(article, maxWordLength))
    else: 
        articleList = [article]
    summarySentences = []
    for chunk in articleList:
        prompt = "summarize: " + chunk
        #print(prompt)
        input_ids = tokenizer(prompt,return_tensors="pt").input_ids
        outputs = model.generate(input_ids, max_length = 512)
        summarySentences.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
    return " ".join(summarySentences)

print(getArticleSummary(["e-health", "diabetes"]))