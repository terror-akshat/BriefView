from flask import Flask, render_template, request
import speech_recognition as sr
import os
from transformers import pipeline
from moviepy.editor import VideoFileClip
import nltk
import gc

nltk.download('punkt')

app = Flask(__name__)

# Use a smaller model for reduced memory usage
summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small", framework="pt")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    text = request.form.get('text')
    video_file = request.files.get('video')
    recognized_text = ""

    if video_file:
        video_path = "temp_video.mp4"
        audio_path = "temp_audio.wav"
        video_file.save(video_path)

        try:
            with VideoFileClip(video_path) as video:
                video.audio.write_audiofile(audio_path, codec='pcm_s16le')

            recognizer = sr.Recognizer()
            with sr.AudioFile(audio_path) as source:
                audio_data = recognizer.record(source)
                recognized_text = recognizer.recognize_google(audio_data)
        except Exception as e:
            return render_template('index.html', error=f"Error processing video/audio: {e}")
        finally:
            if os.path.exists(video_path):
                os.remove(video_path)
            if os.path.exists(audio_path):
                os.remove(audio_path)
            gc.collect()  # Clear memory

    if not text:
        text = recognized_text

    if not text:
        return render_template('index.html', error="No text provided for summarization")

    try:
        summary = summarizer(text, max_length=100, min_length=10, do_sample=False)
        gc.collect()  # Clear memory after summarization
    except Exception as e:
        return render_template('index.html', error=f"Error during summarization: {e}")

    full_summary = summary[0]['summary_text']
    summary_bullet_points = nltk.sent_tokenize(full_summary)
    recognized_bullet_points = nltk.sent_tokenize(recognized_text)

    return render_template('index.html', summary=full_summary, 
                           summary_bullet_points=summary_bullet_points,
                           original_text=text, recognized_text=recognized_bullet_points)

if __name__ == '__main__':
    app.run(debug=True)
