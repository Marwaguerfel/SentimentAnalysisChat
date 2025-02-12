# Sentiment Analysis Chatbot

A real-time chatbot system that combines BERT-based sentiment analysis with OpenAI's GPT-3.5 for generating contextual responses. The system maintains conversation memory and provides conversation summaries.

## Features

- Real-time sentiment analysis using BERT
- Contextual responses using OpenAI GPT-3.5
- Conversation memory management
- Conversation summarization
- Web interface with real-time updates

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sentiment-chatbot.git
cd sentiment-chatbot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install requirements:
```bash
pip install -r requirements.txt
```

4. Set up your configuration:
   - Copy `key.example.py` to `key.py`
   - Add your OpenAI API key in `key.py`

5. Download BERT model files and place them in `bert_sentiment_model/` directory:
   - config.json
   - model.safetensors
   - tokenizer_config.json
   - vocab.txt
   - special_tokens_map.json
   - training_args.bin

6. Run the application:
```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Project Structure

```
sentiment-chatbot/
├── app.py              # Main application file
├── key.example.py      # Example configuration file
├── requirements.txt    # Project dependencies
├── README.md          # Project documentation
├── bert_sentiment_model/
│   ├── config.json
│   ├── model.safetensors
│   └── ...
└── templates/
    └── index.html     # Web interface
```

## Usage

1. Start the server:
```bash
python app.py
```

2. Open your browser and navigate to `http://localhost:5000`

3. Start chatting! The system will:
   - Analyze sentiment of your messages
   - Provide contextual responses
   - Maintain conversation history
   - Generate conversation summaries

## API Endpoints

### POST /chat
Send a message:
```json
{
    "text": "Hello, how are you?",
    "session_id": "unique_session_id"
}
```

### POST /summarize
Get conversation summary:
```json
{
    "session_id": "unique_session_id"
}
```

## Configuration

Configure your OpenAI API key in `key.py`:
```python
OPENAI_API_KEY = 'your-openai-api-key-here'
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
