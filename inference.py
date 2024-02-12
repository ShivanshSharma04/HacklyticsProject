from transformers import T5Tokenizer, T5ForConditionalGeneration

def method(text):
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")

    passage = "Generate Blog: "+text
    input_ids = tokenizer(passage,return_tensors="pt").input_ids

    outputs = model.generate(input_ids, max_length = 1000, max_new_tokens=1000)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))