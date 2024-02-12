from transformers import AutoModelWithHeads, AutoTokenizer

# Replace 'adapter_hub_name' with the actual name of the adapter on Hugging Face
adapter_hub_name = "AndyYu25/hacklytics24-medsummarizer"

# Load the pretrained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(adapter_hub_name)
model = AutoModelWithHeads.from_pretrained(adapter_hub_name)

# Load the adapter from Hub
adapter_name = model.load_adapter(adapter_hub_name)

# Activate the adapter
model.active_adapters = adapter_name

# Prepare the input text
text_to_summarize = "Your text goes here."
inputs = tokenizer(text_to_summarize, return_tensors="pt", truncation=True)

# Forward pass
outputs = model(**inputs)

# Now you need to process the outputs to get the summary. This will depend on the specific model and task.
