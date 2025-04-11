# Web Content Semantic Search

A powerful application that semantically searches web content using natural language processing. This tool fetches HTML content from any URL, processes it, and allows you to search for relevant information using semantic understanding rather than just keyword matching.

##  Features

- Fetch raw HTML content from any website URL
- Process and chunk HTML content into semantically meaningful pieces (max 500 tokens each)
- Generate vector embeddings using transformer-based models
- Perform semantic search across content chunks
- Return and display top 10 matching results with original HTML formatting
- Score results by relevance to your query

## 🛠️ Technology Stack

- **Backend**: Python, Flask, BeautifulSoup4, Transformers, scikit-learn
- **Frontend**: React, HTML, CSS
- **NLP**: Hugging Face Transformers (`sentence-transformers/all-MiniLM-L6-v2`)

##  Prerequisites

- Python 3.7+ with pip
- Node.js 14+ with npm

### Setup

```bash
# Create and navigate to project directory
git clone https://github.com/yourusername/web-content-search.git
cd web-content-search

# Create virtual environment
cd backend
python -m venv venv

# Activate virtual environment
# For Windows
venv\Scripts\activate
# For macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# If any issue occurs 
pip install scikit-learn 

cd .. /frontend

npm i # install all the node module


### Start the server

cd backend
python app.py

cd frontend
npm start


