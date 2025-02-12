from flask import Flask, request, jsonify, render_template
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import nltk
from nltk.tokenize import word_tokenize
import re
import os
from flask_cors import CORS
from openai import OpenAI
from datetime import datetime
from key import OPENAI_API_KEY

# Initialize basic setup
app = Flask(__name__)
CORS(app)
client = OpenAI(api_key=OPENAI_API_KEY)

# Constants
MODEL_PATH = "./bert_sentiment_model"
MAX_MEMORY = 10
conversation_memory = {}

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

def process_text(text):
    """Clean and analyze text with sentiment"""
    # Clean text
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\S+|#\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    cleaned_text = " ".join(word_tokenize(text))
    
    # Analyze sentiment
    inputs = tokenizer(
        cleaned_text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_class].item()
    
    sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
    sentiment = sentiment_map[predicted_class]
    
    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "scores": {sentiment_map[i]: prob.item() for i, prob in enumerate(probs[0])}
    }

def manage_conversation(session_id, role, content, sentiment=None):
    """Manage conversation memory and history"""
    # Initialize session if needed
    if session_id not in conversation_memory:
        conversation_memory[session_id] = {
            'messages': [],
            'start_time': datetime.now(),
            'sentiment_history': []
        }
    
    # Add message to history
    message = {
        'role': role,
        'content': content,
        'timestamp': datetime.now().isoformat()
    }
    
    if sentiment:
        message['sentiment'] = sentiment
        conversation_memory[session_id]['sentiment_history'].append(sentiment)
    
    conversation_memory[session_id]['messages'].append(message)
    
    # Keep only recent messages
    if len(conversation_memory[session_id]['messages']) > MAX_MEMORY:
        conversation_memory[session_id]['messages'].pop(0)
    
    return conversation_memory[session_id]

def generate_prompt(session_id):
    """Generate context for OpenAI"""
    if session_id not in conversation_memory:
        return ""
        
    history = conversation_memory[session_id]
    
    # Get sentiment trend
    recent_sentiments = history['sentiment_history'][-3:]
    sentiment_trend = max(set(recent_sentiments), key=recent_sentiments.count) if recent_sentiments else "neutral"
    
    # Calculate duration
    duration = datetime.now() - history['start_time']
    duration_mins = int(duration.total_seconds() / 60)
    
    context = f"""Conversation Context:
    Duration: {duration_mins} minutes
    Overall sentiment: {sentiment_trend}
    
    Recent messages:"""
    
    # Add recent messages
    for msg in history['messages'][-3:]:
        context += f"\n{msg['role']}: {msg['content']}"
    
    return context

def get_ai_response(session_id, user_message, sentiment_info):
    """Get response from OpenAI"""
    try:
        context = generate_prompt(session_id)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": f"""You are an empathetic chatbot with context:
                    {context}
                    Current sentiment: {sentiment_info['sentiment']} 
                    (confidence: {sentiment_info['confidence']*100:.1f}%)"""
                },
                {"role": "user", "content": user_message}
            ],
            max_tokens=150,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI Error: {str(e)}")
        fallbacks = {
            "positive": "I'm glad you're feeling positive! How can I help?",
            "negative": "I understand this might be difficult. How can I help?",
            "neutral": "Would you like to tell me more?"
        }
        return fallbacks.get(sentiment_info['sentiment'], "Please continue...")

def create_summary(session_id):
    """Create conversation summary"""
    if session_id not in conversation_memory:
        return "No conversation to summarize."
        
    history = conversation_memory[session_id]
    
    # Format messages
    messages = "\n".join([
        f"{msg['role']} ({msg.get('sentiment', 'unknown')}): {msg['content']}"
        for msg in history['messages']
    ])
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Create a concise but comprehensive conversation summary including: main topics, key points, emotional context, and next steps."
                },
                {
                    "role": "user",
                    "content": f"Summarize this conversation:\n{messages}"
                }
            ],
            max_tokens=300,
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Summary Error: {str(e)}")
        return "Error generating summary."
    





    

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        message = data.get('text', '').strip()
        session_id = data.get('session_id', 'default')
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
            
        # Process message
        sentiment_info = process_text(message)
        manage_conversation(session_id, 'user', message, sentiment_info['sentiment'])
        
        # Get response
        response = get_ai_response(session_id, message, sentiment_info)
        manage_conversation(session_id, 'assistant', response)
        
        return jsonify({
            'response': response,
            'sentiment': sentiment_info['sentiment'],
            'confidence': sentiment_info['confidence'],
            'additional_info': sentiment_info['scores']
        })
        
    except Exception as e:
        print(f"Chat Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        data = request.json
        session_id = data.get('session_id', 'default')
        summary = create_summary(session_id)
        
        return jsonify({
            'summary': summary,
            'type': 'conversation'
        })
        
    except Exception as e:
        print(f"Summary Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)