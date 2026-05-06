# ============================================================
#   NLP PROJECT: Sentiment Analysis using Transformers
#   Level     : Beginner (Fresher-Friendly)
#   Author    : Your Name
#   Libraries : transformers, torch
# ============================================================
#
#   WHAT IS SENTIMENT ANALYSIS?
#   It means reading a piece of text and deciding whether
#   the person is saying something POSITIVE, NEGATIVE, or NEUTRAL.
#
#   Example:
#       "I love this movie!"     --> POSITIVE
#       "This food is terrible"  --> NEGATIVE
#
#   WHAT IS A TRANSFORMER?
#   It's a type of AI model that understands human language.
#   We use a pre-trained model called "BERT" — it was already
#   trained by Google on millions of sentences, so we just USE it.
# ============================================================


# Step 1: Import the tools we need
# ---------------------------------
# 'pipeline' is a helper from the 'transformers' library.
# It handles everything (loading model, tokenizing, predicting)
# with just one line of code!
from transformers import pipeline


# ============================================================
# MODULE 1: Load the Model
# ============================================================

def load_model():
    """
    This function loads a pre-trained sentiment analysis model.

    'pipeline' downloads and sets up the model automatically.
    The model we use is 'distilbert-base-uncased-finetuned-sst-2-english'
    - distilbert : a smaller, faster version of BERT
    - sst-2      : it was trained on a movie review dataset (SST-2)
    - english    : works with English text
    """
    print("Loading the AI model... (first time may take a minute to download)")
    print("-" * 55)

    # This one line does everything:
    # downloads the model, sets up tokenizer, ready to predict
    sentiment_model = pipeline(
        task="sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    print("Model loaded successfully!\n")
    return sentiment_model


# ============================================================
# MODULE 2: Analyze a Single Sentence
# ============================================================

def analyze_one(model, sentence):
    """
    Takes one sentence and returns its sentiment.

    Parameters:
        model    : the loaded transformer model
        sentence : a string (the text to analyze)

    Returns:
        label    : 'POSITIVE' or 'NEGATIVE'
        score    : confidence (0.0 to 1.0)
    """

    # The model reads the sentence and returns a prediction
    result = model(sentence)

    # result looks like: [{'label': 'POSITIVE', 'score': 0.9998}]
    # We take the first (and only) item from the list
    label = result[0]['label']
    score = result[0]['score']

    return label, score


# ============================================================
# MODULE 3: Analyze Multiple Sentences
# ============================================================

def analyze_many(model, sentences):
    """
    Takes a list of sentences and prints the result for each one.

    Parameters:
        model     : the loaded transformer model
        sentences : a list of strings
    """
    print("=" * 55)
    print("BATCH SENTIMENT ANALYSIS RESULTS")
    print("=" * 55)

    for i, sentence in enumerate(sentences, start=1):
        label, score = analyze_one(model, sentence)
        confidence = round(score * 100, 2)  # Convert to percentage

        # Pick an emoji to make output friendly
        emoji = "😊" if label == "POSITIVE" else "😞"

        print(f"\n[{i}] Text     : {sentence}")
        print(f"    Sentiment: {label} {emoji}")
        print(f"    Confidence: {confidence}%")

    print("\n" + "=" * 55)


# ============================================================
# MODULE 4: Interactive Mode (User types their own sentence)
# ============================================================

def interactive_mode(model):
    """
    Lets the user type their own sentence and see the result.
    Type 'quit' to exit.
    """
    print("\n--- INTERACTIVE MODE ---")
    print("Type any sentence to analyze its sentiment.")
    print("Type 'quit' to exit.\n")

    while True:
        # Get input from user
        user_input = input("Enter sentence: ").strip()

        # Exit condition
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break

        # Skip empty input
        if not user_input:
            print("Please enter a sentence.\n")
            continue

        # Run the model on user's sentence
        label, score = analyze_one(model, user_input)
        confidence = round(score * 100, 2)
        emoji = "😊" if label == "POSITIVE" else "😞"

        print(f"  --> Sentiment : {label} {emoji}")
        print(f"  --> Confidence: {confidence}%\n")


# ============================================================
# MAIN FUNCTION: The Starting Point of the Program
# ============================================================

def main():
    print("\n" + "=" * 55)
    print("   NLP PROJECT: Sentiment Analysis with Transformers")
    print("=" * 55 + "\n")

    # Step 1: Load the model
    model = load_model()

    # Step 2: Demo — analyze some pre-written sentences
    demo_sentences = [
        "I absolutely love this product, it changed my life!",
        "The service was terrible and I will never come back.",
        "Today was a great day, everything went perfectly.",
        "I'm so disappointed with this laptop, it keeps crashing.",
        "The food was okay, nothing special.",
        "Best movie I have seen in years!"
    ]

    analyze_many(model, demo_sentences)

    # Step 3: Let the user try it themselves
    interactive_mode(model)


# ============================================================
# Python Entry Point
# This runs main() only when you run this file directly.
# It will NOT run if this file is imported by another file.
# ============================================================
if __name__ == "__main__":
    main()
