# CodeAlpha FAQ Chatbot

A FAQ chatbot created for **CodeAlpha Artificial Intelligence Internship — Task 2**.

The application preprocesses user questions with NLP, represents FAQ text using
TF-IDF, measures cosine similarity, and returns the closest matching FAQ answer.
It also includes a Streamlit chat interface and a confidence threshold to reduce
incorrect matches.

## Features

- FAQ dataset stored in JSON
- Text cleaning and tokenization
- Stop-word removal
- Porter stemming with NLTK
- TF-IDF vectorization
- Cosine-similarity matching
- Confidence threshold and fallback response
- Suggested related FAQs
- Streamlit chat interface
- Optional match/debug details
- Automated unit tests
- No external API key required
- No NLTK data download required

## Project Structure

```text
CodeAlpha_FAQChatbot/
│
├── app.py
├── requirements.txt
├── README.md
├── demo_questions.txt
├── .gitignore
│
├── data/
│   └── faqs.json
│
├── src/
│   ├── __init__.py
│   └── chatbot.py
│
└── tests/
    └── test_chatbot.py
```

## How the NLP Pipeline Works

1. User enters a question.
2. Text is converted to lowercase and cleaned.
3. NLTK `wordpunct_tokenize` tokenizes the sentence.
4. Common English stop words are removed.
5. NLTK `PorterStemmer` reduces words to stems.
6. `TfidfVectorizer` converts the stored FAQs and user question into vectors.
7. Cosine similarity compares the user vector with all FAQ vectors.
8. The FAQ with the highest similarity score is selected.
9. If the score is below the selected threshold, the chatbot returns a safe
   fallback response instead of an unrelated answer.

## Recommended Python Version

Python 3.11 or 3.12 is recommended.

Check your version:

```bash
python --version
```

On Windows, this may also work:

```bash
py --version
```

## Setup in VS Code (Windows)

### 1. Open the project

Extract the ZIP and open the `CodeAlpha_FAQChatbot` folder in VS Code.

### 2. Open a terminal

Use:

`Terminal` → `New Terminal`

### 3. Create a virtual environment

```powershell
py -m venv .venv
```

If `py` is not available:

```powershell
python -m venv .venv
```

### 4. Activate it

PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

If PowerShell blocks activation, use Command Prompt in VS Code instead of
changing security settings.

### 5. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 6. Run automated tests

```powershell
python -m unittest discover -s tests -v
```

All tests should report `OK`.

### 7. Start the chatbot

```powershell
python -m streamlit run app.py
```

A browser window should open automatically. If it does not, copy the local URL
printed in the terminal, normally `http://localhost:8501`.

## Quick Demonstration

Try:

- `How many projects must I finish?`
- `Where do I upload my code?`
- `What should my repository be called?`
- `Which NLP methods does Task 2 suggest?`
- `Can I use YOLO for task 4?`
- `How does this chatbot work?`

Then test an unrelated question such as:

- `What is the distance between Earth and Neptune?`

The app should give a low-confidence fallback instead of pretending to know the
answer.

## GitHub Upload

Create an empty GitHub repository called:

```text
CodeAlpha_FAQChatbot
```

Do not add a README, `.gitignore`, or license on GitHub if you plan to push this
local folder directly.

From the project folder:

```bash
git init
git add .
git commit -m "Complete CodeAlpha FAQ Chatbot"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/CodeAlpha_FAQChatbot.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your real GitHub username.

## Troubleshooting

### `python` is not recognized

Install Python and make sure "Add Python to PATH" is enabled, or try `py`
instead of `python`.

### `streamlit` is not recognized

Run it through Python:

```bash
python -m streamlit run app.py
```

### PowerShell says script execution is disabled

Use the VS Code terminal dropdown and select **Command Prompt**, then activate:

```cmd
.venv\Scripts\activate.bat
```

### Module not found

Make sure the virtual environment is activated, then run:

```bash
pip install -r requirements.txt
```

### Git push is rejected

If your GitHub repository was created with an online README, the remote may
already contain a commit. The easiest beginner-friendly solution is to create a
new empty repository and push this project to it.

## Internship Explanation

**Problem:** Users need quick answers to frequently asked internship questions.

**Solution:** This project stores a set of FAQs and applies NLP preprocessing,
TF-IDF vectorization, and cosine similarity to find the FAQ that is most similar
to the user's question.

**Why TF-IDF?** It represents text numerically while giving more importance to
terms that are useful for distinguishing documents.

**Why cosine similarity?** It measures how similar the direction of two text
vectors is, making it suitable for comparing a user query with FAQ questions.

**Why a threshold?** A threshold prevents the chatbot from returning an
unrelated FAQ answer when similarity is weak.

## Technologies

- Python
- NLTK
- scikit-learn
- Streamlit

## Internship Task

CodeAlpha Artificial Intelligence Internship — Task 2: Chatbot for FAQs.
