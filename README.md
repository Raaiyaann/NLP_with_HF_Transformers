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
unmasker(" I <mask> You ", top_k=4)
```

Result : 

```
[{'score': 0.53789883852005,
  'token': 3437,
  'token_str': ' Love',
  'sequence': ' I Love You '},
 {'score': 0.09510327875614166,
  'token': 4250,
  'token_str': ' See',
  'sequence': ' I See You '},
 {'score': 0.05283082276582718,
  'token': 27032,
  'token_str': ' Hate',
  'sequence': ' I Hate You '},
 {'score': 0.04998588189482689,
  'token': 3837,
  'token_str': ' Thank',
  'sequence': ' I Thank You '}]
```

Analysis on example 3.5 : 

model fill-mask ini berhasil mengisi beberapa kata untuk melengkapi kalimat yang sudah di sediakan .

### 4. Example 4 - Name Entity Recognition (NER)

```
# TODO :
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", grouped_entities=True)
ner(
    "My name is Mohammad Raiyan. I am currently studying Computer Science at Tadulako University in Palu, Indonesia, and I previously completed an internship at Microsoft in Singapore."
)
```

Result : 

```
[{'entity_group': 'PER',
  'score': np.float32(0.99907947),
  'word': 'Mohammad Raiyan',
  'start': 11,
  'end': 26},
 {'entity_group': 'ORG',
  'score': np.float32(0.32507682),
  'word': 'Computer',
  'start': 52,
  'end': 60},
 {'entity_group': 'MISC',
  'score': np.float32(0.415785),
  'word': 'Science',
  'start': 61,
  'end': 68},
 {'entity_group': 'ORG',
  'score': np.float32(0.99373794),
  'word': 'Tadulako University',
  'start': 72,
  'end': 91},
 {'entity_group': 'LOC',
  'score': np.float32(0.9642228),
  'word': 'Palu',
  'start': 95,
  'end': 99},
 {'entity_group': 'LOC',
  'score': np.float32(0.99935406),
  'word': 'Indonesia',
  'start': 101,
  'end': 110},
 {'entity_group': 'ORG',
  'score': np.float32(0.99932563),
  'word': 'Microsoft',
  'start': 156,
  'end': 165},
 {'entity_group': 'LOC',
  'score': np.float32(0.9997336),
  'word': 'Singapore',
  'start': 169,
  'end': 178}]
```

Analysis on example 4 : 

untuk  named entity recognizer berhasil mengidentifikasi nama orang, tempat dan nama brand seperti "mohammad raiyan" untuk nama, "computer" nama device, "microsoft" untuk nama brand dan masih banyak lagi yang berhasil di identifikasi .

### 5. Example 5 - Question Answering

```
# TODO :
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
question = "Who was the command module pilot who orbited the Moon?"
context = "The Apollo 11 mission was a historic moment in human exploration. It was the specific spaceflight conducted by NASA, the United States' space agency, that first landed humans on the Moon. The mission launched on July 16, 1969. The commander of the mission was Neil Armstrong, and the lunar module pilot was Buzz Aldrin. Armstrong was the first person to step onto the lunar surface on July 21, 1969, and he famously uttered the words, 'That's one small step for [a] man, one giant leap for mankind.' Michael Collins served as the command module pilot, orbiting the Moon while Armstrong and Aldrin were on the surface."
qa_model(question = question, context = context)
```

Result : 

```
{'score': 0.9751918911933899,
 'start': 500,
 'end': 515,
 'answer': 'Michael Collins'}
```

Analysis on example 5 : 

model question-answering berhasil menjawab pertanyaan yang dibuat dari teks yang sangat panjang dengan tingkat keyakinan 97% yang artinya sudah cukup baik dalam menjawab pertanyaan.

### 6. Example 6 - Text Summarization

```
# TODO :
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", max_length=59)

summarizer(
    """
Modern manufacturing depends on consistent product quality, because even small defects can lead to returns, safety issues, or damage to the brand's reputation. Quality control is not only about rejecting bad products at the end of the line, but about understanding why defects happen in the first place and preventing them from repeating. To do this, factories collect measurements such as weight, thickness, temperature, and pressure at every stage of production. These values are compared against tolerance limits to detect early signs of drift or abnormal behavior. If a batch begins to fall outside the acceptable range, the system can alert operators, pause the line, and trigger an investigation before the issue spreads. This approach reduces waste, improves safety, and keeps production reliable and cost-efficient.
"""
)
```

Result : 

```
[{'summary_text': ' Quality control is not only about rejecting bad products at the end of the line, but about understanding why defects happen . To do this, factories collect measurements such as weight, thickness, temperature, and pressure at every stage of production . These values are compared against tolerance limits to detect early signs of drift or abnormal behavior .'}]

```

Analysis on example 6 :

untuk summarization pipeline berhasil meringkas sebuah kalimat paragraf yang sangat panjang untuk memberikan point utama mengenai deskripsi teks panjang tersebut agar lebih mudah dipahami .

### 7. Example 7 - Translation

```
# TODO :
translator = pipeline("translation_en_to_de", model="t5-small")
print(translator("i love playing with my friends", max_length=40))
```

Result : 

```
[{'translation_text': 'ich liebe es, mit meinen Freunden zu spielen'}]

```

Analysis on example 7 :

mode translation sudah berhasil mengubah bahasa inggris yang saya masukan menjadi bahasa german, saat saya translate kembali bahasa german yang di hasilkan ke bahasa inggris, artinya masih sama  .

---

## Analysis on this project

This project offers a practical introduction to various NLP tasks using Hugging Face pipelines. Each example is easy to follow and demonstrates real-world use cases. The variety of models shows the flexibility of transformer-based solutions in solving different types of language problems.
