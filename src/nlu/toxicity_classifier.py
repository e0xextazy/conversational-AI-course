import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'cointegrated/rubert-tiny-toxicity'
tokenizer_toxic = AutoTokenizer.from_pretrained(model_path)
model_toxic = AutoModelForSequenceClassification.from_pretrained(model_path)
model_toxic = model_toxic.to(device)


def text2toxicity(text: str, aggregate: bool = True) -> float:
    """ Calculate toxicity of a text (if aggregate=True) or a vector of toxicity aspects (if aggregate=False)"""
    with torch.no_grad():
        inputs = tokenizer_toxic(
            text, return_tensors='pt', truncation=True).to(model_toxic.device)
        proba = torch.sigmoid(model_toxic(**inputs).logits).cpu().numpy()
    if isinstance(text, str):
        proba = proba[0]
    if aggregate:
        return 1 - proba.T[0] * (1 - proba.T[-1])
    return proba
