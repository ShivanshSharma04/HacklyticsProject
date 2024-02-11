from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

passage = "Summarize: Exploiting information in health-related social media services is of great interest for patients, researchers and medical companies. The challenge is, however, to provide easy, quick and relevant access to the vast amount of information that is available. One step towards facilitating information access to online health data is opinion mining. Even though the classification of patient opinions into positive and negative has been previously tackled, most works make use of machine learning methods and bags of words. Our first contribution is an extensive evaluation of different features, including lexical, syntactic, semantic, network-based, sentiment-based and word embeddings features to represent patient-authored texts for polarity classification. The second contribution of this work is the study of polar facts (i.e. objective information with polar connotations). Traditionally, the presence of polar facts has been neglected and research in polarity classification has been bounded to opinionated texts. We demonstrate the existence and importance of polar facts for the polarity classification of health information."
input_ids = tokenizer(passage,return_tensors="pt").input_ids

outputs = model.generate(input_ids, max_length = 1000, max_new_tokens=1000)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))