import re
import gradio as gr
from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="lvwerra/distilbert-imdb")
tokenizer = classifier.tokenizer

def clean(text):
    return re.sub(r"<br\s*/?>", "", text)

# def head_tail(text, tokenizer, max_len=510, head=128):
#     ids = tokenizer.encode(text, add_special_tokens=False)
#     if len(ids) <= max_len:
#         return text
#     ids = ids[:head] + ids[-(max_len - head):]
#     return tokenizer.decode(ids)

def predict(text):
    text = clean(text)
    # text = head_tail(text,tokenizer)
    result = classifier(text, truncation = True)[0]
    return {result["label"]: result["score"]}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=8, placeholder="Post your movie review..."),
    outputs=gr.Label(),
    title="IMDB Sentiment Analysis of Movie Reviews",
    description="Enter an English movie review and determine whether it is positive or negative.",
)

if __name__ == "__main__":
    demo.launch()
