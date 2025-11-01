<h1 align="center"> Natural Language Processing  with Hugging Face Transformers </h1>
<p align="center"> Generative AI Guided Project on Cognitive Class by IBM</p>

<div align="center">

<img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54">
<img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white">

</div>

## Name : Mohammad Raiyan

## My todo : 

### 1. Example 1 - Sentiment Analysis

```
# TODO
original_model = pipeline("sentiment-analysis")
data ="This vehicle has changed my life.  Full self driving has decreased my workday, commute stress, and when I want control , I have never driven anything that handles so well and accelerates so smoothly. Love my new model Y!"
original_model(data)
```

Result : 

```
[{'label': 'POSITIVE', 'score': 0.9995622038841248}]
```

Analysis on example 1 : 

jadi setniment analisis pada review sebuah mobil tesla yang merek terbaru di twitter itu menunjukkan jenis review positive dengan tingkat keyakinan sebesar 99 % 


### 2. Example 2 - Topic Classification

```
# TODO
topik_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
topik_classifier(
    "Cybersecurity is the practice of protecting systems, networks, and sensitive data from unauthorized access, attacks, and disruption. Modern threats include phishing, ransomware, and data breaches, so organizations need prevention, monitoring, and incident response.",
    candidate_labels=["cybersecurity", "finance", "healthcare"],
)
```

Result : 

```
{'sequence': 'Cybersecurity is the practice of protecting systems, networks, and sensitive data from unauthorized access, attacks, and disruption. Modern threats include phishing, ransomware, and data breaches, so organizations need prevention, monitoring, and incident response.',
 'labels': ['cybersecurity', 'finance', 'healthcare'],
 'scores': [0.992111086845398, 0.004041106905788183, 0.0038478232454508543]}
```

Analysis on example 2 : 

model zero-shot classifier yang digunakan disini berhasil mengidentifikasi kalau topik pada teks tersebut itu mengenai cybersecurity dengan tingkat keyakinan 99 %, jika dibandingkan dengan label yang lain yang tersedia itu itu seperti finance dengan healthcare hanya mendapatkan score 0.004% dan 0.003% yang artinya tidak sesuai dengan data topik yang diberikan.

### 3. Example 3 and 3.5 - Text Generator

```
# TODO :
generator = pipeline("text-generation", model="gpt2")
generator("this time i will")
```

Result : 

```
[{'generated_text': "this time i will stop going to this place and start going again}.
```

Analysis on example 3 : 

model text generation berhasil menghasilkan beberapa kalimat yang baru yang sesuai dengan konteks kalimat sebelumnya seperti "this time i will", dengan di generate-nya teks baru yang menghasilkan kalimat seperti "this time i will stop going to this place and start going again" .

```
unmasker = pipeline("fill-mask", "distilroberta-base")
unmasker("This person is the one who <mask> my purse", top_k=4)
```

Result : 

```
[{'score': 0.8569591641426086,
  'token': 8268,
  'token_str': ' stole',
  'sequence': 'This person is the one who stole my purse'},
 {'score': 0.030922001227736473,
  'token': 25702,
  'token_str': ' snatched',
  'sequence': 'This person is the one who snatched my purse'},
 {'score': 0.02246157079935074,
  'token': 12297,
  'token_str': ' steals',
  'sequence': 'This person is the one who steals my purse'},
 {'score': 0.01934182271361351,
  'token': 2263,
  'token_str': ' broke',
  'sequence': 'This person is the one who broke my purse'}]
```

Analysis on example 3.5 : 

The fill-mask pipeline accurately infers masked words based on context. The top result "stole" makes sense, supported by a high confidence score. Other predictions are also contextually appropriate, illustrating the model's nuanced understanding of sentence structure and intent.

### 4. Example 4 - Name Entity Recognition (NER)

```
# TODO :
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", grouped_entities=True)
ner("My name is Arifian, I am an AI Technical Mentor at Infinite Learning, Batam Island")
```

Result : 

```
[{'entity_group': 'PER',
  'score': np.float32(0.9978566),
  'word': 'Arifian',
  'start': 11,
  'end': 18},
 {'entity_group': 'ORG',
  'score': np.float32(0.7615841),
  'word': 'AI',
  'start': 28,
  'end': 30},
 {'entity_group': 'ORG',
  'score': np.float32(0.9623977),
  'word': 'Infinite Learning',
  'start': 51,
  'end': 68},
 {'entity_group': 'LOC',
  'score': np.float32(0.9913697),
  'word': 'Batam Island',
  'start': 70,
  'end': 82}]
```

Analysis on example 4 : 

The named entity recognizer successfully identifies personal, organizational, and location entities from the sentence. Grouped outputs are relevant and accurate, with high confidence scores, demonstrating the model’s effectiveness in real-world applications like information extraction or document tagging.

### 5. Example 5 - Question Answering

```
# TODO :
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
question = "What four-legged animal sometimes comes inside the house and likes to sleep?"
context = "Four-legged animal that sometimes comes inside the house and likes to sleep is a cat"
qa_model(question = question, context = context)
```

Result : 

```
{'score': 0.6314472556114197, 'start': 79, 'end': 84, 'answer': 'a cat'}
```

Analysis on example 5 : 

The question-answering model correctly extracts the most relevant phrase "a cat" from the provided context. Its confidence score is decent, and the model showcases strong capabilities in understanding natural questions and matching them with the most likely answer span.

### 6. Example 6 - Text Summarization

```
# TODO :
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
summarizer(
    """
Machine Learning adalah cabang dari Kecerdasan Buatan yang memungkinkan sistem komputer untuk belajar dari data tanpa diprogram secara eksplisit. 1  Melalui algoritma, mesin dapat mengidentifikasi pola, membuat prediksi, dan meningkatkan kinerja seiring waktu. Penerapannya luas, mulai dari rekomendasi produk hingga diagnosis medis, mengubah cara kita berinteraksi dengan teknologi. 
"""
)
```

Result : 

```
[{'summary_text': ' Machine Learning adalah cabang dari Kecerdasan Buatan yang memungkinkan komputer untuk belajar dari data tanpa diprogram secara eksplisit . Melalui algoritma, mesin dapat mengidentifikasi pola, membuat prediksi, dan meningkatkan kinerja seiring waktu .'}]

```

Analysis on example 6 :

The summarization pipeline effectively condenses the core idea of the paragraph into a shorter version. It maintains key concepts like machine learning, pattern recognition, and practical applications, reflecting the model's strength in content compression without major loss of information.

### 7. Example 7 - Translation

```
# TODO :
translator_id = pipeline("translation", model="Helsinki-NLP/opus-mt-id-fr")
translator_id("Hari ini masak apa, chef?")
```

Result : 

```
[{'translation_text': "Qu'est-ce qu'on fait aujourd'hui, chef ?"}]

```

Analysis on example 7 :

The translation model delivers an accurate and context-aware French translation of the Indonesian sentence. It handles informal, conversational input smoothly, making it suitable for multilingual communication tasks and cross-language understanding in casual or daily scenarios.

---

## Analysis on this project

This project offers a practical introduction to various NLP tasks using Hugging Face pipelines. Each example is easy to follow and demonstrates real-world use cases. The variety of models shows the flexibility of transformer-based solutions in solving different types of language problems.
