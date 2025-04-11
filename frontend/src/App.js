import React, { useState } from 'react';
import './App.css';

function App() {
  const [url, setUrl] = useState('');
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('http://localhost:5000/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ url, query }),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `Server returned ${response.status}`);
      }
      
      const data = await response.json();
      setResults(data.results);
    } catch (err) {
      setError(err.message);
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Web Content Semantic Search</h1>
        <p>Enter a website URL and search query to find relevant content</p>
      </header>
      
      <main>
        <form onSubmit={handleSubmit} className="search-form">
          <div className="form-group">
            <label htmlFor="url">Website URL:</label>
            <input
              type="text"
              id="url"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://example.com"
              required
            />
          </div>
          
          <div className="form-group">
            <label htmlFor="query">Search Query:</label>
            <input
              type="text"
              id="query"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Enter your search query"
              required
            />
          </div>
          
          <button type="submit" disabled={loading}>
            {loading ? 'Searching...' : 'Search'}
          </button>
        </form>
        
        {error && <div className="error-message">Error: {error}</div>}
        
        {loading && <div className="loading">Processing your request...</div>}
        
        {results.length > 0 && (
          <div className="results-container">
            <h2>Top {results.length} Matches</h2>
            <div className="results-list">
              {results.map((result, index) => (
                <div key={index} className="result-card">
                  <h3>Match #{index + 1} (Relevance Score: {(result.score * 100).toFixed(2)}%)</h3>
                  <div className="result-content" dangerouslySetInnerHTML={{ __html: result.content }} />
                </div>
              ))}
            </div>
          </div>
        )}
        
        {!loading && results.length === 0 && url && query && (
          <div className="no-results">No results found. Try a different query or URL.</div>
        )}
      </main>
      
      <footer>
        <p>Web Content Semantic Search © 2025</p>
      </footer>
    </div>
  );
}

export default App;
