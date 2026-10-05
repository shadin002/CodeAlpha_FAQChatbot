# Presentation / LinkedIn Video Notes

## 60–90 second explanation

Hello, this is my CodeAlpha Artificial Intelligence Internship Task 2 project,
an FAQ Chatbot.

The goal of this project is to answer frequently asked questions by finding the
stored FAQ that is most similar to the user's question.

I created a JSON FAQ dataset related to the CodeAlpha AI internship. The text is
preprocessed using NLP techniques. I convert it to lowercase, clean unnecessary
characters, tokenize it, remove common stop words, and apply Porter stemming
using NLTK.

After preprocessing, I use scikit-learn's TF-IDF vectorizer to convert the text
into numerical vectors. Then I calculate cosine similarity between the user's
question and all stored FAQs. The chatbot returns the answer belonging to the
highest-scoring FAQ.

I also added a confidence threshold so that the chatbot does not return an
unrelated answer when the similarity is too low.

For the user interface, I used Streamlit and created a simple chat experience
with optional similarity details.

This project demonstrates NLP preprocessing, information retrieval, similarity
matching, Python programming, and UI development.

Thank you.

## Demo flow

1. Show the project folder in VS Code.
2. Briefly show `data/faqs.json`.
3. Briefly show `src/chatbot.py`.
4. Run `python -m unittest discover -s tests -v`.
5. Run `python -m streamlit run app.py`.
6. Ask: "How many projects must I finish?"
7. Ask: "Where do I upload my source code?"
8. Open "How this answer was matched" and show the similarity score.
9. Ask an unrelated question to demonstrate the confidence fallback.
10. Show the GitHub repository.

## Viva questions

### What is TF-IDF?
TF-IDF stands for Term Frequency–Inverse Document Frequency. It converts text
into numerical features and gives more importance to words that help
distinguish one document from others.

### What is cosine similarity?
Cosine similarity compares two vectors using the cosine of the angle between
them. A higher score means the text vectors are more similar.

### Why did you use stemming?
Stemming reduces related word forms to a common root-like form so variations
can match more easily.

### Why did you remove stop words?
Very common words often contribute little meaning to FAQ matching, so removing
them can make important terms more influential.

### Why use a threshold?
Without a threshold, the system would always return some FAQ even for unrelated
questions. The threshold lets the chatbot reject low-confidence matches.

### Is this a generative AI chatbot?
No. It is a retrieval-based FAQ chatbot. It selects the best answer from a
curated FAQ dataset rather than generating a new answer with a large language
model.
