Market Research Agent - Project Documentation

Overview
--------
This project implements an Agentic RAG (Retrieval-Augmented Generation) workflow that performs
live web search using DuckDuckGo, extracts relevant content, and generates insights using Cohere’s
Command-R model. A simple Gradio UI allows users to query market research insights.

Features
--------
- Real-time web search via DuckDuckGo
- HTML scraping with BeautifulSoup
- Cohere LLM-powered reasoning and summarization
- Agentic RAG workflow
- Gradio user interface
- Environment variables via .env file

Project Structure
-----------------
app.py                  - Main application file
requirements.txt        - Dependencies
README.md               - Documentation
.env                    - API key configuration

Requirements
-----------
gradio
beautifulsoup4
requests
python-dotenv
cohere

Environment Variables
---------------------
Create a .env file:
COHERE_API_KEY=your_api_key_here

How It Works
------------
1. Web Search using DuckDuckGo
2. Web Page Retrieval using BeautifulSoup
3. Cohere LLM reasoning using retrieved documents
4. Display results via Gradio UI

Running the Application
-----------------------
Run:
python app.py

Example Queries
---------------
- Market size of electric scooters in India
- AI-based customer support tools trends
- Competitors of Freshworks and pricing models

Notes & Limitations
-------------------
- DuckDuckGo HTML scraping is unofficial.
- Scraping may break depending on website structure.
- API key required for Cohere.

Future Enhancements
-------------------
- Add caching
- Structured output (SWOT, pricing tables)
- Vector DB integration
- Streaming responses

