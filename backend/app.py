# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Function to generate embeddings
def generate_embedding(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        model_output = model(**inputs)
    
    # Mean pooling
    attention_mask = inputs['attention_mask']
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    return embeddings[0].numpy()

# Function to clean and extract text from HTML
def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove script and style elements
    for script_style in soup(["script", "style"]):
        script_style.extract()
    
    # Get text
    text = soup.get_text()
    
    # Break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # Break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # Remove blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)
    
    return text

# Function to chunk HTML content with original HTML
def chunk_html_content(html_content, max_tokens=500):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Get all elements that might contain meaningful content
    elements = soup.find_all(['p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'span', 'li', 'td', 'th', 'article', 'section'])
    
    chunks = []
    current_chunk = ""
    current_token_count = 0
    
    for element in elements:
        # Skip empty elements
        if not element.get_text().strip():
            continue
        
        # Get the HTML of this element
        element_html = str(element)
        element_text = element.get_text()
        
        # Count tokens in this element text
        tokens = tokenizer.encode(element_text)
        token_count = len(tokens)
        
        # If adding this element would exceed our chunk size, start a new chunk
        if current_token_count + token_count > max_tokens and current_chunk:
            chunks.append(current_chunk)
            current_chunk = element_html
            current_token_count = token_count
        else:
            current_chunk += element_html
            current_token_count += token_count
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    url = data.get('url')
    query = data.get('query')
    
    if not url or not query:
        return jsonify({"error": "URL and query are required"}), 400
    
    try:
        # 1. Fetch HTML content
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()  # Raise exception for 4XX/5XX responses
        html_content = response.text
        
        # 2. Tokenize and chunk HTML content (max 500 tokens per chunk)
        html_chunks = chunk_html_content(html_content, max_tokens=500)
        
        if not html_chunks:
            return jsonify({"error": "No content chunks could be extracted from the URL"}), 400
        
        # 3. Generate embeddings for chunks and query
        chunk_embeddings = []
        for chunk in html_chunks:
            # Extract text for embedding generation
            chunk_text = BeautifulSoup(chunk, 'html.parser').get_text()
            chunk_embeddings.append(generate_embedding(chunk_text))
        
        query_embedding = generate_embedding(query)
        
        # 4. Calculate similarity scores
        chunk_embeddings_np = np.array(chunk_embeddings)
        query_embedding_np = query_embedding.reshape(1, -1)
        
        similarities = cosine_similarity(query_embedding_np, chunk_embeddings_np)[0]
        
        # 5. Get top 10 results
        top_k = min(10, len(html_chunks))
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        # 6. Prepare results
        results = []
        for i in top_indices:
            results.append({
                "content": html_chunks[i],
                "score": float(similarities[i]),
                "url": url
            })
        
        return jsonify({"results": results})
    
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Error fetching URL: {str(e)}"}), 400
    except Exception as e:
        import traceback
        return jsonify({
            "error": f"Server error: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500

if __name__ == '__main__':
    app.run(debug=True)