# 🧠 NLP Project: Sentiment Analysis using Transformers

A **beginner-friendly** NLP project that uses a pre-trained Transformer model
to detect whether a sentence is **POSITIVE** or **NEGATIVE**.

---

## 📁 Project Structure

```
nlp_project/
│
├── sentiment_analyzer.py   ← Main Python file (all logic here)
├── requirements.txt        ← Libraries to install
├── .gitignore              ← Files to ignore in Git
└── README.md               ← This file
```

---

## 🧩 Modules Inside `sentiment_analyzer.py`

| Module | Function | What it does |
|--------|----------|--------------|
| 1 | `load_model()` | Downloads and loads the AI model |
| 2 | `analyze_one()` | Analyzes one sentence |
| 3 | `analyze_many()` | Analyzes a list of sentences |
| 4 | `interactive_mode()` | Lets user type their own text |

---

## 🚀 How to Run

### Step 1 — Install Python
Make sure Python 3.8+ is installed.
```bash
python --version
```

### Step 2 — Create & Activate Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### Step 3 — Install Libraries
```bash
pip install -r requirements.txt
```

### Step 4 — Run the Project
```bash
python sentiment_analyzer.py
```

---

## 💡 Sample Output

```
[1] Text     : I absolutely love this product!
    Sentiment: POSITIVE 😊
    Confidence: 99.98%

[2] Text     : The service was terrible.
    Sentiment: NEGATIVE 😞
    Confidence: 99.87%

Enter sentence: (you type here!)
```

---

## 🔑 Key Concepts (For Beginners)

| Term | Meaning |
|------|---------|
| **NLP** | Natural Language Processing — teaching computers to understand human text |
| **Transformer** | A powerful AI architecture (invented by Google in 2017) |
| **BERT / DistilBERT** | Pre-trained transformer models that understand English |
| **Sentiment Analysis** | Classifying text as Positive or Negative |
| **Pipeline** | A helper tool in `transformers` library that makes using models easy |
| **Confidence Score** | How sure the model is (0% = not sure, 100% = very sure) |

---

## 📚 What to Learn Next

1. Text Classification (more categories, not just pos/neg)
2. Named Entity Recognition (NER) — find names, places in text
3. Question Answering — ask questions from a paragraph
4. Fine-tuning — train a pre-built model on your own data
