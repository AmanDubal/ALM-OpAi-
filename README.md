# Audio Language Model (ALM)

AI-powered audio scene understanding system using OpenRouter for LLM reasoning.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your OpenRouter API key
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY

# 3. Run the app
streamlit run app.py
```

Get your free API key at https://openrouter.ai

## Supported Models (set in .env)
- `openai/gpt-4-turbo` — best quality (default)
- `openai/gpt-4o`
- `anthropic/claude-3-opus`
- `mistralai/mistral-7b-instruct` — fastest / cheapest
- `meta-llama/llama-3-70b-instruct`
