from flask import Flask, render_template, request
import speech_recognition as sr
import os
from transformers import pipeline
from moviepy.editor import VideoFileClip
import nltk

# Ensure that the necessary NLTK resources are available
nltk.download('punkt_tab')

# Initialize the Flask application
app = Flask(__name__)

# Initialize the summarization pipeline
summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small", framework="pt")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    text = request.form.get('text')
    video_file = request.files.get('video')
    recognized_text = ""  # Initialize a variable for recognized text

    if video_file:
        video_path = "temp_video.mp4"
        audio_path = "temp_audio.wav"
        
        # Save uploaded video file temporarily
        video_file.save(video_path)
        print("Video file saved successfully.")  # Debugging print statement

        try:
            # Extract audio from video file
            with VideoFileClip(video_path) as video:
                video.audio.write_audiofile(audio_path, codec='pcm_s16le')
            print("Audio extracted successfully.")  # Debugging print statement

            # Recognize speech from audio file
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
        except Exception as e:
            print(f"Error during audio extraction or recognition: {e}")
            return render_template('index.html', error=f"Error processing video/audio: {e}")
        finally:
            # Clean up temporary files
            if os.path.exists(video_path):
                os.remove(video_path)
            if os.path.exists(audio_path):
                os.remove(audio_path)

    if not text:
        text = recognized_text  # Use recognized text if no text is provided

    if not text:
        return render_template('index.html', error="No text provided for summarization")

    try:
        summary = summarizer(text, max_length=100, min_length=10, do_sample=False)
        print("Summarization successful.")  # Debugging print statement
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
    # Add host and port for local development
    app.run(debug=True, host='0.0.0.0', port=5000)

