from transformers import T5Tokenizer, T5ForConditionalGeneration
import articleFetch


def getArticleSummary(tags: list[str]):
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")

    prompt = "sumarize: " + articleFetch.getPaperText(tags)
    print(prompt)
    input_ids = tokenizer(prompt,return_tensors="pt").input_ids

    outputs = model.generate(input_ids, max_length = 500)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)