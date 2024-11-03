from flask import Flask, render_template, request
import speech_recognition as sr
import os
from transformers import pipeline
from moviepy.editor import VideoFileClip
import nltk
import threading
import signal
import psutil
import time

# Ensure that the necessary NLTK resources are available
nltk.download('punkt')

# Initialize the Flask application
app = Flask(__name__)

# Initialize the summarization pipeline once
summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base", framework="pt")

# Set maximum allowed file size (e.g., 50MB) and text length limits
MAX_FILE_SIZE_MB = 50
MAX_TEXT_LENGTH = 5000

# Timeout decorator
class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

signal.signal(signal.SIGALRM, timeout_handler)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    text = request.form.get('text')
    video_file = request.files.get('video')
    recognized_text = ""  # Initialize a variable for recognized text

    # Check for file size limits (in bytes)
    if video_file and len(video_file.read()) > MAX_FILE_SIZE_MB * 1024 * 1024:
        return render_template('index.html', error=f"File size exceeds {MAX_FILE_SIZE_MB} MB limit")
    
    # Reset read pointer after checking size
    if video_file:
        video_file.seek(0)

    if video_file:
        video_path = "temp_video.mp4"
        audio_path = "temp_audio.wav"

        # Save the uploaded video file temporarily
        video_file.save(video_path)
        print("Video file saved successfully.")  # Debugging print statement

        try:
            # Set timeout for processing the video
            signal.alarm(120)  # Set a timeout of 120 seconds

            # Extract audio from the video file
            with VideoFileClip(video_path) as video:
                video.audio.write_audiofile(audio_path, codec='pcm_s16le')
            print("Audio extracted successfully.")  # Debugging print statement

            # Recognize speech from the audio file
            recognizer = sr.Recognizer()
            with sr.AudioFile(audio_path) as source:
                audio_data = recognizer.record(source)
                try:
                    recognized_text = recognizer.recognize_google(audio_data)
                    print("Recognized text:", recognized_text)  # Print recognized text for debugging
                except sr.UnknownValueError:
                    print("Could not understand audio.")
                    return render_template('index.html', error="Could not understand audio")
                except sr.RequestError as e:
                    print(f"Speech recognition service error: {e}")
                    return render_template('index.html', error=f"Error with speech recognition service: {e}")
        except TimeoutError:
            return render_template('index.html', error="Video processing timed out. Please try with a smaller file.")
        except Exception as e:
            print(f"Error during audio extraction or recognition: {e}")
            return render_template('index.html', error=f"Error processing video/audio: {e}")
        finally:
            # Cancel the timeout
            signal.alarm(0)
            
            # Clean up temporary files
            if os.path.exists(video_path):
                os.remove(video_path)
            if os.path.exists(audio_path):
                os.remove(audio_path)

    # Use recognized text if no text is provided in the form
    if not text:
        text = recognized_text

    if not text:
        return render_template('index.html', error="No text provided for summarization")

    if len(text) > MAX_TEXT_LENGTH:
        return render_template('index.html', error=f"Text length exceeds {MAX_TEXT_LENGTH} characters")

    try:
        # Set a timeout for the summarization process
        signal.alarm(60)  # Set a timeout of 60 seconds for summarization

        # Perform text summarization
        summary = summarizer(text, max_length=100, min_length=10, do_sample=False)
        print("Summarization successful.")  # Debugging print statement

        # Cancel the timeout after completion
        signal.alarm(0)
    except TimeoutError:
        return render_template('index.html', error="Summarization process timed out. Try a shorter text.")
    except MemoryError:
        return render_template('index.html', error="Memory limit exceeded during summarization.")
    except Exception as e:
        print(f"Error during summarization: {e}")
        return render_template('index.html', error=f"Error during summarization: {e}")

    # Prepare summary and bullet points
    full_summary = summary[0]['summary_text']
    summary_bullet_points = nltk.sent_tokenize(full_summary)
    recognized_bullet_points = nltk.sent_tokenize(recognized_text)

    return render_template('index.html', summary=full_summary, 
                           summary_bullet_points=summary_bullet_points,
                           original_text=text, recognized_text=recognized_bullet_points)

if __name__ == '__main__':
    # Check for memory availability before starting the app
    if psutil.virtual_memory().available < 500 * 1024 * 1024:  # Example: 500 MB
        print("Warning: Low memory available. This might affect performance.")
    app.run(debug=True)
