import os
import json
import tempfile
import logging
from flask import Flask, request, render_template, jsonify, send_file, url_for
from google.cloud import speech
from google.cloud import storage
from google.cloud import texttospeech
from fuzzywuzzy import fuzz
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure Cloud Storage
BUCKET_NAME = 'upgraded-spoon-bucket'
storage_client = storage.Client()

def get_or_create_bucket(bucket_name):
    try:
        bucket = storage_client.get_bucket(bucket_name)
        logger.info(f"Connected to bucket: {bucket_name}")
    except google.api_core.exceptions.NotFound:
        try:
            bucket = storage_client.create_bucket(bucket_name)
            logger.info(f"Bucket {bucket_name} created.")
        except Exception as e:
            logger.error(f"Error creating bucket {bucket_name}: {e}")
            bucket = None
    except Exception as e:
        logger.error(f"Error accessing bucket {bucket_name}: {e}")
        bucket = None
    return bucket

bucket = get_or_create_bucket(BUCKET_NAME)

# Create uploads folder for local testing
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Maximum file size (20MB)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024

# Allowed audio file extensions
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'opus', 'webm', 'ogg'}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load Spanish Dictionary for pronunciation assessment
def load_dictionary():
    """Load Spanish dictionary from either cloud storage or local file"""
    try:
        # First try to load from the local merged dictionary file
        try:
            with open("dictionaries/final_spanish.txt", "r", encoding="utf-8") as f:
                # Assuming one word per line
                words = [line.strip().lower() for line in f if line.strip()]
                return set(words)
        except FileNotFoundError:
            # If local file not found, try to load from Cloud Storage
            if bucket:
                blob = bucket.blob('final_spanish.txt')
                if blob.exists():
                    content = blob.download_as_string().decode('utf-8')
                    words = [line.strip().lower() for line in content.splitlines() if line.strip()]
                    return set(words)
            # Fallback to a small built-in dictionary
            logger.warning("Could not load dictionary file. Using minimal built-in dictionary.")
            return set([
                "hola", "como", "estás", "bien", "gracias", "adios", "buenos", "días", 
                "hasta", "luego", "mañana", "tarde", "noche", "por", "favor", "de", "nada",
                "sí", "no", "tal", "vez", "quizás", "casa", "coche", "trabajo", "escuela",
                "universidad", "restaurante", "tienda", "mercado", "parque", "playa", "montaña",
                "emergencia", "calma", "siga", "instrucciones", "seguridad", "caso"
            ])
    except Exception as e:
        logger.error(f"Error loading dictionary: {e}")
        return set()
# Load reference phrases for assessment and practice
def load_references():
    """Load reference phrases from local JSON file"""
    try:
        with open("data/references.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading local references: {e}")
        return {
            "beginner": "Hola, ¿cómo estás?",
            "intermediate": "Me gusta viajar y conocer nuevas culturas.",
            "advanced": "La educación es fundamental para el desarrollo de la sociedad."
        }
# Initialize Spanish dictionary and references
SPANISH_DICT = load_dictionary()
REFERENCES = load_references()
logger.info(f"Dictionary loaded with {len(SPANISH_DICT)} words")

# Transcribe audio using Google Cloud Speech-to-Text
def transcribe_audio(audio_content):
    """Transcribe Spanish audio using Google Cloud Speech-to-Text"""
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(content=audio_content)
    
    # Try multiple configurations to increase success rate
    configs = [
        speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="es-ES",
            enable_automatic_punctuation=True
        ),
        speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED,
            sample_rate_hertz=16000,
            language_code="es-ES",
            enable_automatic_punctuation=True
        ),
        speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
            language_code="es-ES",
            enable_automatic_punctuation=True
        )
    ]
    
    for config in configs:
        try:
            logger.info(f"Trying transcription with config: {config}")
            response = client.recognize(config=config, audio=audio)
            
            if response.results:
                transcript = " ".join(result.alternatives[0].transcript for result in response.results)
                logger.info(f"Transcription successful: '{transcript}'")
                return transcript
            else:
                logger.warning("No transcription results")
        except Exception as e:
            logger.error(f"Error in transcription with config {config}: {str(e)}")
    
    logger.error("All transcription attempts failed")
    return ""

# Calculate pronunciation score when doing free speech
def assess_free_speech(transcribed_text):
    """
    Evaluate pronunciation using ACTFL FACT criteria for free speech mode
    """
    return actfl_assessment(transcribed_text)

# Calculate pronunciation score when practicing with reference phrases
def assess_practice_phrase(transcribed_text, reference_level):
    """
    Evaluate pronunciation with reference to a specific practice phrase
    """
    if reference_level not in REFERENCES:
        return actfl_assessment(transcribed_text)
    
    reference_text = REFERENCES[reference_level]
    
    # Compare transcribed text with reference text
    similarity_score = fuzz.token_sort_ratio(transcribed_text.lower(), reference_text.lower())
    
    # Get a base assessment
    base_assessment = actfl_assessment(transcribed_text)
    
    # Adjust score based on similarity to reference
    similarity_bonus = (similarity_score - 60) * 0.2 if similarity_score > 60 else 0
    adjusted_score = min(100, base_assessment["score"] + similarity_bonus)
    
    # Create a new assessment with adjusted scores
    assessment = {
        "score": round(adjusted_score, 1),
        "level": base_assessment["level"],
        "reference_text": reference_text,
        "similarity": similarity_score,
        "feedback": base_assessment["feedback"],
        "strengths": base_assessment["strengths"],
        "areas_for_improvement": base_assessment["areas_for_improvement"]
    }
    
    # Add reference-specific feedback
    if similarity_score < 50:
        assessment["areas_for_improvement"].insert(0, "Your response differed significantly from the reference phrase")
    elif similarity_score < 75:
        assessment["areas_for_improvement"].insert(0, "Try to follow the reference phrase more closely")
    else:
        assessment["strengths"].insert(0, "Good reproduction of the reference phrase")
    
    return assessment

# Calculate pronunciation score based on ACTFL FACT criteria
def actfl_assessment(transcribed_text):
    """
    Evaluate pronunciation using ACTFL FACT criteria:
    - Functions and tasks: Can the speaker communicate their message?
    - Accuracy: How precise is their pronunciation?
    - Context and content: Can they handle the topic appropriately?
    - Text type: Can they produce appropriate sentence structures?
    """
    words = transcribed_text.split()
    if not words:
        logger.warning("No words to score")
        return {
            "score": 70.0,
            "level": "Novice Mid",
            "feedback": "No speech detected. Please try again and speak clearly.",
            "strengths": [],
            "areas_for_improvement": ["Speak clearly into the microphone"]
        }
    
    # Score each word's pronunciation accuracy
    word_scores = []
    mispronounced_words = []
    
    for word in words:
        word = word.lower()
        if word in SPANISH_DICT:
            score = 100  # Perfect match
            logger.info(f"Word '{word}' found in dictionary, score: 100")
        else:
            # Find best match using fuzzy matching
            best_match = None
            best_ratio = 0
            
            # Check against a sample of the dictionary for performance
            dict_sample = set(list(SPANISH_DICT)[:1000]) if len(SPANISH_DICT) > 1000 else SPANISH_DICT
            
            for dict_word in dict_sample:
                ratio = fuzz.ratio(word, dict_word)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match = dict_word
            
            score = best_ratio
            logger.info(f"Word '{word}' not found. Best match: '{best_match}' with score: {score}")
            
            if score < 80:
                mispronounced_words.append(word)
        
        word_scores.append(score)
    
    # Calculate average accuracy score
    accuracy_score = sum(word_scores) / len(word_scores) if word_scores else 70
    
    # Evaluate overall proficiency criteria
    
    # 1. Functions and tasks: Based on successfully recognized words
    recognized_word_percentage = sum(1 for s in word_scores if s >= 80) / len(words)
    
    # 2. Text type: Based on number of words (complexity of response)
    text_complexity = min(100, 60 + len(words) * 4)  # Bonus for longer phrases
    
    # 3. Content: Based on variety of vocabulary (unique words ratio)
    unique_words_ratio = len(set(words)) / len(words)
    vocabulary_score = min(100, unique_words_ratio * 100)
    
    # Calculate composite ACTFL score with different weights
    # Pronunciation accuracy is most important
    composite_score = (
        accuracy_score * 0.6 +
        recognized_word_percentage * 100 * 0.2 +
        text_complexity * 0.1 +
        vocabulary_score * 0.1
    )
    
    # Native speaker adjustment - boost scores for clearly native speakers
    if accuracy_score > 90 and len(words) > 3:
        composite_score = min(100, composite_score + 5)
    
    # Determine ACTFL level
    level = determine_actfl_level(composite_score, len(words), recognized_word_percentage)
    
    # Generate feedback
    strengths = generate_strengths(accuracy_score, recognized_word_percentage, len(words))
    areas_for_improvement = generate_improvements(mispronounced_words, accuracy_score)
    
    logger.info(f"ACTFL Scoring - Accuracy: {accuracy_score}, Recognition: {recognized_word_percentage*100}, " +
              f"Text: {text_complexity}, Vocab: {vocabulary_score}, Final: {composite_score}, Level: {level}")
    
    return {
        "score": round(composite_score, 1),
        "level": level,
        "feedback": generate_feedback(level),
        "strengths": strengths,
        "areas_for_improvement": areas_for_improvement,
        "word_scores": dict(zip(words, word_scores))
    }

def determine_actfl_level(score, word_count, recognized_ratio):
    """Determine ACTFL proficiency level based on score and other factors"""
    
    # Distinguished/Superior level
    if score >= 95:
        return "Distinguished"
    elif score >= 90:
        return "Superior"
    
    # Advanced levels
    elif score >= 85:
        return "Advanced High"
    elif score >= 80:
        return "Advanced Mid"
    elif score >= 75:
        return "Advanced Low"
    
    # Intermediate levels
    elif score >= 70:
        if word_count >= 5 and recognized_ratio >= 0.7:
            return "Intermediate High"
        else:
            return "Intermediate Mid"
    elif score >= 65:
        return "Intermediate Low"
    
    # Novice levels
    elif score >= 60:
        return "Novice High"
    elif score >= 55:
        return "Novice Mid"
    else:
        return "Novice Low"

def generate_feedback(level):
    """Generate feedback text based on ACTFL level"""
    feedback_templates = {
        "Distinguished": "Your pronunciation is exceptional and indistinguishable from a native speaker.",
        "Superior": "Your pronunciation is excellent with native-like fluency and clarity.",
        "Advanced High": "Your pronunciation is very strong with occasional minor inaccuracies.",
        "Advanced Mid": "Your pronunciation is clear and competent. Most sounds are accurate.",
        "Advanced Low": "Your pronunciation is good but uneven, with some inconsistencies.",
        "Intermediate High": "Your pronunciation is understandable with some common errors.",
        "Intermediate Mid": "Your pronunciation is generally understandable despite noticeable errors.",
        "Intermediate Low": "Your pronunciation has frequent errors but is somewhat understandable.",
        "Novice High": "Your pronunciation shows effort but needs considerable improvement.",
        "Novice Mid": "Your pronunciation has significant issues affecting comprehensibility.",
        "Novice Low": "Your pronunciation requires substantial development."
    }
    return feedback_templates.get(level, "Your pronunciation shows varying levels of accuracy.")

def generate_strengths(accuracy, recognition_ratio, word_count):
    """Generate list of strengths based on performance"""
    strengths = []
    
    if accuracy >= 90:
        strengths.append("Excellent pronunciation accuracy")
    elif accuracy >= 80:
        strengths.append("Good pronunciation accuracy")
    elif accuracy >= 70:
        strengths.append("Fair pronunciation accuracy")
    
    if recognition_ratio >= 0.9:
        strengths.append("Nearly all words clearly pronounced")
    elif recognition_ratio >= 0.7:
        strengths.append("Most words clearly pronounced")
    
    if word_count >= 10:
        strengths.append("Good use of extended speech")
    elif word_count >= 5:
        strengths.append("Effective use of phrases")
    
    if not strengths:
        strengths.append("Some speech recognized")
    
    return strengths

def generate_improvements(mispronounced, accuracy):
    """Generate suggested areas for improvement"""
    areas = []
    
    if len(mispronounced) > 0:
        if len(mispronounced) <= 3:
            areas.append(f"Focus on pronouncing: {', '.join(mispronounced)}")
        else:
            areas.append("Several words were mispronounced or unclear")
    
    if accuracy < 70:
        areas.append("Work on overall pronunciation clarity")
    elif accuracy < 80:
        areas.append("Practice common Spanish sounds")
    
    if not areas:
        areas.append("Continue practicing to maintain your skills")
    
    return areas

def generate_corrected_text(transcribed_text):
    """Generate grammatically corrected version of the transcribed text"""
    # This is a simplified version that just returns the transcribed text
    # In a full implementation, you would use a grammar correction model or service
    # For now, we're just implementing some basic corrections
    
    # Simple corrections for common errors
    corrections = {
        "tu eres": "tú eres",
        "el es": "él es",
        "ella esta": "ella está",
        "tu tienes": "tú tienes",
        "yo quero": "yo quiero",
        "buenes dias": "buenos días",
        "como esta": "cómo está",
        "como estas": "cómo estás",
        "gracias por tu ayudar": "gracias por tu ayuda",
        "no problemo": "no hay problema",
        "yo no se": "yo no sé"
    }
    
    corrected = transcribed_text
    for error, correction in corrections.items():
        corrected = corrected.replace(error, correction)
    
    return corrected

def generate_tts_feedback(text, level):
    """Generate Text-to-Speech audio feedback in Spanish"""
    try:
        # Initialize Text-to-Speech client
        client = texttospeech.TextToSpeechClient()
        
        # Select voice based on proficiency level (slower for beginners)
        speaking_rate = 0.8 if level.startswith("Novice") else 1.0
        
        # Build the voice request
        synthesis_input = texttospeech.SynthesisInput(text=text)
        
        # Use a female Spanish voice
        voice = texttospeech.VoiceSelectionParams(
            language_code="es-ES",
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )
        
        # Select the type of audio file
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=speaking_rate
        )
        
        # Perform the text-to-speech request
        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        
        # Generate a unique filename
        filename = f"tts_{uuid.uuid4()}.mp3"
        
        # If we have a bucket, upload to Cloud Storage
        if bucket:
            blob = bucket.blob(f"tts/{filename}")
            blob.upload_from_bytes(response.audio_content, content_type='audio/mpeg')
            
            # Create a signed URL that will be valid for 1 hour
            url = blob.generate_signed_url(
                version="v4",
                expiration=3600,  # 1 hour
                method="GET"
            )
            return url
        else:
            # Save to a temporary file and return its path
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            temp_file.write(response.audio_content)
            temp_file.close()
            
    except Exception as e:
        logger.error(f"Error generating TTS: {e}")
        return None

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check endpoint"""
    bucket_status = "connected" if bucket else "not connected"
    return jsonify({
        "status": "ok",
        "bucket": bucket_status,
        "bucket_name": BUCKET_NAME,
        "dictionary_size": len(SPANISH_DICT)
    })

@app.route('/process-audio', methods=['POST'])
def process_audio():
    """Process uploaded or recorded audio and provide assessment"""
    try:
        # Check if the post request has the file part
        if 'file' not in request.files:
            logger.error("No file in request")
            return jsonify({"error": "No audio file in the request"}), 400
        
        file = request.files['file']
        if file.filename == '':
            logger.error("Empty filename in request")
            return jsonify({"error": "No selected file"}), 400
        
        if not allowed_file(file.filename):
            logger.error(f"Invalid file type: {file.filename}")
            return jsonify({"error": "Invalid file type. Please upload .wav, .mp3, .m4a, .opus, .webm, or .ogg"}), 400
        
        # Check if this is a practice mode assessment
        practice_level = request.form.get('practice_level', None)
        
        # Process in memory
        audio_content = file.read()
        logger.info(f"Received audio file of size: {len(audio_content)} bytes")
        
        # Transcribe audio
        spoken_text = transcribe_audio(audio_content)
        
        if not spoken_text:
            logger.warning("No transcription returned")
            return jsonify({
                "score": 70,
                "level": "Novice Mid",
                "transcribed_text": "No transcription available. Please try speaking clearly in Spanish.",
                "error": "Could not transcribe audio. Please try again.",
                "strengths": [],
                "areas_for_improvement": ["Speak clearly in Spanish", "Check your microphone"]
            })
        
        # Calculate assessment based on mode
        if practice_level and practice_level in REFERENCES:
            # Practice mode with reference phrase
            assessment = assess_practice_phrase(spoken_text, practice_level)
            corrected_text = REFERENCES[practice_level]  # Use reference as corrected text
            logger.info(f"Practice mode assessment: level={practice_level}, score={assessment['score']}")
        else:
            # Free speech mode
            assessment = assess_free_speech(spoken_text)
            corrected_text = generate_corrected_text(spoken_text)
            logger.info(f"Free speech assessment: level={assessment['level']}, score={assessment['score']}")
        
        # Generate TTS feedback
        tts_url = generate_tts_feedback(corrected_text, assessment['level'])
        
        # Prepare response
        response = {
            "score": assessment['score'],
            "level": assessment['level'],
            "transcribed_text": spoken_text,
            "corrected_text": corrected_text,
            "feedback": assessment['feedback'],
            "strengths": assessment['strengths'],
            "areas_for_improvement": assessment['areas_for_improvement'],
            "tts_audio_url": tts_url
        }
        
        # Add practice-specific fields if applicable
        if practice_level and 'reference_text' in assessment:
            response["reference_text"] = assessment['reference_text']
            response["similarity"] = assessment['similarity']
        
        return jsonify(response)
            
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/get-tts-audio/<filename>')
def get_tts_audio(filename):
    """Serve TTS audio files from local storage"""
    file_path = app.config.get(f'TTS_FILE_{filename}')
    if not file_path:
        return "Audio file not found", 404
    
    return send_file(file_path, mimetype='audio/mpeg')

@app.route('/references')
def get_references():
    """Serves the reference phrases for practice"""
    try:
        return jsonify(REFERENCES)
    except Exception as e:
        logger.error(f"Error loading references: {e}")
        return jsonify({
            "error": "Could not load references",
            "beginner": "Hola, ¿cómo estás?"
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
