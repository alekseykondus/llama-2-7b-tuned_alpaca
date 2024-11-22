from transformers import pipeline
import torch

generator = pipeline("text-generation", model="oleksiikondus/llama-2-7b-tuned_alpaca", device=0 if torch.cuda.is_available() else -1)

while True:
    input_text = input("Please enter your text (or type 'exit' to stop): ")

    if input_text.lower() == 'exit':
        break

    output = generator(input_text, max_new_tokens=512, return_full_text=True)
    generated_text = output[0]["generated_text"]

    print(generated_text)