# STEP 1, 필요한 패키지 설치
from transformers import pipeline

# STEP 2, 모델 선정
summarizer = pipeline("summarization", model="stevhliu/my_awesome_billsum_model", max_length=50)

# STEP 3, 데이터 가져오기
text = "summarize: The Inflation Reduction Act lowers prescription drug costs, health care costs, and energy costs. It's the most aggressive action on tackling the climate crisis in American history, which will lift up American workers and create good-paying, union jobs across the country. It'll lower the deficit and ask the ultra-wealthy and corporations to pay their fair share. And no one making under $400,000 per year will pay a penny more in taxes."

# STEP 4, 데이터 추론시키기
summarizer(text)

# STEP 5, 출력
print(summarizer)
