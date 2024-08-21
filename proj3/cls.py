# STEP 1
from transformers import pipeline
# from transformers import AutoTokenizer
# from transformers import AutoModelForSequenceClassification

# STEP 2
# model = 슬래쉬 앞에는 누가 만들었는지 확인, 그렇지 않을경우 허킹 페이스에서 로그인해야 됨
classifier = pipeline("sentiment-analysis", model="snunlp/KR-FinBert-SC")
# tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_model")
# model = AutoModelForSequenceClassification.from_pretrained("stevhliu/my_awesome_model")

# STEP 3
# text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."
text = "현대바이오, '폴리탁셀' 코로나19 치료 가능성에 19% 급등"

# STEP 4
result = classifier(text)
# inputs = tokenizer(text, return_tensors="pt")

# with torch.no_grad():
#     logits = model(**inputs).logits

# predicted_class_id = logits.argmax().item()
# model.config.id2label[predicted_class_id]

# STEP 5
print(result)
