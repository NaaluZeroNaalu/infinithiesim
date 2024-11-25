import os
import sqlite3 as sql
from flask import Flask, render_template, request, redirect, url_for, send_from_directory,session
from werkzeug.utils import secure_filename
import moviepy.editor as mp
import speech_recognition as sr
from pydub import AudioSegment
import requests
import subprocess
from ibm_watson import SpeechToTextV1
from ibm_watson.websocket import RecognizeCallback, AudioSource
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import base64
import ast
import secrets


api_key = "zpseg_CvW4iY1piNQkKhxemS2NoRPTkP2VOAOOXkENns"
url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"
project_id = "6de5c4d1-65ac-43c8-8476-ed6082eee2ed"
model_id = "google/flan-ul2"
auth_url = "https://iam.cloud.ibm.com/identity/token"



def check_connection():

    try:

        conn = sql.connect("videos.db")
        print("connected")
    except Exception as e:
        print(e)
    finally:
        if conn:
            conn.close()

check_connection()


def get_access_token():
    print("Generating token")
    auth_url = "https://iam.cloud.ibm.com/identity/token"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json"
    }
    data = {
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        "apikey": api_key
    }
    response = requests.post(auth_url, headers=headers, data=data)
    if response.status_code != 200:
        print(f"Failed to get access token: {response.text}")
        raise Exception("Failed to get access token.")
    else:
        token_info = response.json()
        print("Token generated")
        return token_info['access_token']


def get_datas():

    try:

        conn = sql.connect("videos.db")
        datas = []
        print("fetching datas from database")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM videos ORDER BY idno DESC")
        for i in cursor.fetchall():
            datas.append({"vidno":i[0],"video":i[1],"transcript":i[3],"summary":i[2],"categories":ast.literal_eval(i[4]),"title":i[5]})
        return datas
    except Exception as e:
        print(e)
    finally:
        if conn:
            conn.close()

def get_summary(transcript):
    try:
        access_token = get_access_token()
        prompt = f"Generate a detailed summary of the following transcription, highlighting the key points and main ideas. The summary should be more than 10 lines long and cover the most important information in points from this transcript: {transcript}"
        print("Generating summary prompt...")
        body = {
            "input": prompt,
            "parameters": {
                "decoding_method": "greedy",
                "max_new_tokens": 300,
                "min_new_tokens": 30,
                "stop_sequences": [";"],
                "repetition_penalty": 1.05,
                "temperature": 0.5
            },
            "model_id": model_id,
            "project_id": project_id
        }

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {access_token}"
        }

        response = requests.post(url, headers=headers, json=body)

        if response.status_code == 403:
            # Extracting only the error code
            error_message = response.json()
            error_code = error_message.get("errors", [{}])[0].get("code", "")
            if error_code == "token_quota_reached":
                return error_code  # Return the error code directly

        if response.status_code != 200:
            raise Exception(f"Error in API request: {response.status_code} - {response.text}")

        summary = response.json()
        print("Summary generated successfully.")
        return summary['results'][0]['generated_text'].strip()

    except Exception as e:
        print(f"Error occurred while generating summary: {str(e)}")
        return f"Error occurred while generating summary: {str(e)}"

def get_title(title):
    try:
        access_token = get_access_token()
        prompt = f"Generate a concise one-line title for the following paragraph: {title}"
        print("Generating title prompt...")
        body = {
            "input": prompt,
            "parameters": {
                "decoding_method": "greedy",
                "max_new_tokens": 300,
                "min_new_tokens": 30,
                "stop_sequences": [";"],
                "repetition_penalty": 1.05,
                "temperature": 0.5
            },
            "model_id": model_id,
            "project_id": project_id
        }

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {access_token}"
        }

        response = requests.post(url, headers=headers, json=body)

        if response.status_code == 403:
            
            error_message = response.json()
            print(error_message)
            error_code = error_message.get("errors", [{}])[0].get("code", "")
            if error_code == "token_quota_reached":

                return error_code 

        if response.status_code != 200:
            raise Exception(f"Error in API request: {response.status_code} - {response.text}")

        summary = response.json()
        print("Title generated successfully.")
        print(summary['results'])
        return summary['results'][0]['generated_text'].strip()

    except Exception as e:
        print(f"Error occurred while generating title: {str(e)}")
        return f"Error occurred while generating title: {str(e)}"


app = Flask(__name__)

app.secret_key = secrets.token_hex(16)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv'}
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    session['current_state'] = "indexpage"
    return render_template('index.html')


@app.route('/videos')
def videos():
    videos = get_datas()
    return render_template('videos.html', videos = videos)

@app.route('/remove', methods=['POST'])
def remove_videos():
    idno = request.form['idno']
    print(idno)
    try:
        conn = sql.connect("videos.db")
        cursor = conn.cursor()
        cursor.execute("DELETE FROM videos WHERE idno = ?", (idno,))
        conn.commit()
        print("deleted successfully")
    except Exception as e:
        print(e)
    finally:
        conn.close()
    return redirect(url_for('videos'))
    



@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        existing_files = os.listdir(app.config['UPLOAD_FOLDER'])
        existing_video_files = [f for f in existing_files if allowed_file(f)]
        for video in existing_video_files:
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], video)
            try:
                os.remove(video_path)
                print(f"Video {video} removed.")
            except PermissionError:
                print(f"Failed to remove {video}. It might be open in another process.")
        
        file.save(file_path)
        title = []
        audio_text = process_video(file_path)
        print("Getting transcript...")

        a = ""
        t = ""
        for i in audio_text[0]:
            a = a + i['name']
            t = get_title(i['name'])
            title.append({"time": i['start'], "title": t})
        
        summary = get_summary(a)

        try:

            conn = sql.connect("videos.db")
            cursor = conn.cursor()

            with open(f'uploads//{filename}','rb') as video_file:
                video_data = video_file.read()
            video_base64 = base64.b64encode(video_data).decode('utf-8')
            cursor.execute("INSERT INTO videos(video, summary, transcript, time, title) values(?, ?, ?, ?, ?)",(video_base64,summary, a, str(title), filename))
            conn.commit()
            print("inserted")
        except Exception as e:
            print(e)
        finally:
            conn.close()
        return redirect(url_for('videos'))

    return 'File type not allowed or no file selected'


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


def convert_seconds_to_minutes(seconds):
    print("Converting seconds to minutes")
    minutes = int(seconds // 60)
    seconds = seconds % 60
    return f"{minutes:02}:{seconds:05.2f}"


def group_words_by_intervals(words_with_times):
    print("Grouping words with timestamps")
    interval_transcript = []
    interval_start_time = words_with_times[0][1]
    current_group = []
    
    for word, start_time, end_time in words_with_times:
        if start_time >= interval_start_time + 30:
            interval_transcript.append((current_group, interval_start_time, interval_start_time + 60))
            interval_start_time = start_time
            current_group = [word]
        else:
            current_group.append(word)
    
    if current_group:
        interval_transcript.append((current_group, interval_start_time, interval_start_time + 60))
    
    return interval_transcript


def process_video(video_path):
    try:
        video = mp.VideoFileClip(video_path)
        audio_file = video.audio
        audio_file.write_audiofile("myaudio.wav")
        video.close()

        audio = AudioSegment.from_wav("myaudio.wav")
        audio = audio.set_frame_rate(8000)
        audio = audio.normalize()
        audio.export("myaudio_clean.wav", format="wav")

        keyfortext = "xvEoKfUVPjFIrrZhKoraEzDpcAcFt0lj1wpCTVOR1boO"
        urlfortext = "https://api.au-syd.speech-to-text.watson.cloud.ibm.com/instances/4d5d4ae6-c7d5-4597-a463-bcc5ed508e36"
        
        authenticator = IAMAuthenticator(keyfortext)
        stt = SpeechToTextV1(authenticator=authenticator)
        stt.set_service_url(urlfortext)
        print("Reading wav file")

        with open("myaudio_clean.wav", "rb") as f:
            res = stt.recognize(audio=f, content_type="audio/wav", model="en-AU_NarrowbandModel", timestamps=True).get_result()

        words_with_times = []
        for result in res['results']:
            for alternative in result['alternatives']:
                for word_info in alternative.get('timestamps', []):
                    word, start_time, end_time = word_info
                    words_with_times.append((word, start_time, end_time))
        
        interval_transcript = group_words_by_intervals(words_with_times)
        timestamps = []

        print("Merging timestamps with transcription")
        for group, start_time, end_time in interval_transcript:
            interval_start = convert_seconds_to_minutes(start_time)
            interval_end = convert_seconds_to_minutes(end_time)
            words_segment = " ".join(group)
            timestamps.append({"name": words_segment, "start": interval_start, "end": interval_end})
        
        return timestamps,
        
    except Exception as e:
        print(f"Error occurred during video processing: {str(e)}")
        return {"error": f"An error occurred: {str(e)}"}


if __name__ == "__main__":
    app.run(debug=True)
