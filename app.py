import os
import json
import tempfile
import logging
import base64
from flask import Flask, request, render_template, jsonify, send_file
from google.cloud import speech
from google.cloud import texttospeech
from fuzzywuzzy import fuzz
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'opus', 'webm', 'ogg'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_dictionary():
    try:
        with open("dictionaries/final_spanish.txt", "r", encoding="utf-8") as f:
            words = [line.strip().lower() for line in f if line.strip()]
            return set(words)
    except Exception as e:
        logger.error(f"Error loading dictionary: {e}")
        return set(["hola", "como", "estas"])


def load_references():
    try:
        with open("data/references.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading references: {e}")
        return {
            "beginner": "Hola, ¿cómo estás?",
            "intermediate": "Me gusta viajar y conocer nuevas culturas.",
            "advanced": "La educación es fundamental para el desarrollo de la sociedad."
        }

SPANISH_DICT = load_dictionary()
REFERENCES = load_references()


def transcribe_audio(audio_content):
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(content=audio_content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED,
        sample_rate_hertz=16000,
        language_code="es-ES",
        enable_automatic_punctuation=True
    )
    response = client.recognize(config=config, audio=audio)
    if response.results:
        return " ".join(result.alternatives[0].transcript for result in response.results)
    return ""


def generate_corrected_text(text):
    # Placeholder function (returns same text)
    return text


def actfl_assessment(text):
    words = text.split()
    if not words:
        return {
            "score": 70.0,
            "level": "Novice Mid",
            "feedback": "No speech detected.",
            "strengths": [],
            "areas_for_improvement": ["Speak clearly"]
        }

    accuracy = sum(1 for w in words if w.lower() in SPANISH_DICT) / len(words) * 100
    level = "Intermediate Mid" if accuracy > 75 else "Novice Mid"
    return {
        "score": round(accuracy, 1),
        "level": level,
        "feedback": "Solid pronunciation." if accuracy > 75 else "Keep practicing.",
        "strengths": ["Good pronunciation"] if accuracy > 75 else [],
        "areas_for_improvement": ["Practice vocabulary"] if accuracy <= 75 else []
    }


def generate_tts_feedback(text, level):
    try:
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="es-ES",
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )
        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        return response.audio_content
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return None


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/process-audio', methods=['POST'])
def process_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file"}), 400

    practice_level = request.form.get('practice_level')
    audio_content = file.read()
    transcript = transcribe_audio(audio_content)

    if not transcript:
        return jsonify({"score": 70, "level": "Novice Mid", "transcribed_text": "", "error": "No transcription"})

    if practice_level and practice_level in REFERENCES:
        reference_text = REFERENCES[practice_level]
        similarity = fuzz.token_sort_ratio(transcript.lower(), reference_text.lower())
        base = actfl_assessment(transcript)
        base['similarity'] = similarity
        base['reference_text'] = reference_text
        assessment = base
    else:
        assessment = actfl_assessment(transcript)

    corrected = generate_corrected_text(transcript)
    tts_audio = generate_tts_feedback(corrected, assessment['level'])
    encoded_audio = base64.b64encode(tts_audio).decode('utf-8') if tts_audio else None

    return jsonify({
        "score": assessment['score'],
        "level": assessment['level'],
        "transcribed_text": transcript,
        "corrected_text": corrected,
        "feedback": assessment['feedback'],
        "strengths": assessment['strengths'],
        "areas_for_improvement": assessment['areas_for_improvement'],
        "reference_text": assessment.get('reference_text'),
        "similarity": assessment.get('similarity'),
        "tts_audio": encoded_audio
    })


if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 8080)))
