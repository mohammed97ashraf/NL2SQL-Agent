from flask import Flask, jsonify, render_template, request
import os
import logging
from logging.handlers import RotatingFileHandler
from groq import Groq
from langchain_utils.sql_agent import get_langgraph_agent
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Set up logging
log_handler = RotatingFileHandler('app.log', maxBytes=10000000, backupCount=3)
log_handler.setLevel(logging.INFO)
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_handler.setFormatter(log_formatter)
app.logger.addHandler(log_handler)

@app.route('/')
def index():
    app.logger.info('Serving index page')
    return render_template('index.html')

# Ensure the 'uploads' folder exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

# Initialize the Groq client
client = Groq()

@app.route('/agent', methods=['POST'])
def agent_response():
    input_type = request.form.get('input_type')
    app.logger.info(f'Received request with input_type: {input_type}')
    
    try:
        if input_type == 'voice':
            voice_file = request.files['voice_data']
            save_path = os.path.join('uploads', 'recording.mp3')
            voice_file.save(save_path)
            app.logger.info(f'Voice file saved to: {save_path}')
            
            # Specify the path to the audio file
            filename = "./uploads/recording.mp3"
            app.logger.info(f'Processing audio file: {filename}')
            
            with open(filename, "rb") as file:
                transcription = client.audio.transcriptions.create(
                    file=(filename, file.read()),
                    model="distil-whisper-large-v3-en",
                    prompt="Specify context or spelling",
                    response_format="json",
                    language="en",
                    temperature=0.0
                )
            app.logger.info(f'Transcription result: {transcription}')
            inputs = {"messages": [("user", transcription.text)]}
        
        else:  # Handle text input
            data = request.json['data']
            app.logger.info(f'Received text data: {data}')
            inputs = {"messages": [("user", data)]}
        
        data = get_langgraph_agent().invoke(inputs)
        parse = JsonOutputParser()
        response = parse.parse(data["messages"][-1].content)
        app.logger.info(f'Response generated: {response}')
    
    except Exception as e:
        app.logger.error(f'Error processing request: {e}', exc_info=True)
        response = {"error": "An error occurred while processing your request."}

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
