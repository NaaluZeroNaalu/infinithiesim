import os
import sqlite3 as sql
from flask import Flask, render_template, request, redirect, url_for, send_from_directory,session,jsonify, request
from werkzeug.utils import secure_filename
import moviepy.editor as mp
import speech_recognition as sr
from pydub import AudioSegment
import requests
import subprocess
from ibm_watson import SpeechToTextV1, DiscoveryV2
from ibm_watson.websocket import RecognizeCallback, AudioSource
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import base64
import ast
import secrets
from sentence_transformers import SentenceTransformer
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from elasticsearch import Elasticsearch
import requests


api_key = "3pzCWXm_P0T-6pX3iPpOqgB7XqnxuHIkd0wSmpVoY5eP"
url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"
project_id = "b1ba37d6-b43f-4b9f-b1d2-b97e52fbfbcf"
model_id = "meta-llama/llama-3-1-70b-instruct" 
auth_url = "https://iam.cloud.ibm.com/identity/token"


#for discovery
# IBM Watson Discovery credentials
discovery_apikey = 'gS0DerQXpJEED8hFUkjLItD7JSe0nkBcGsZW3O70I51a'
discovery_url = 'https://api.au-syd.discovery.watson.cloud.ibm.com/instances/1e8f1686-fe91-4773-a4e8-4aafd40c62c4'
collection_id = '6dee3f8d-de5e-0bb2-0000-01938bad0002'
environment_id = '76b7bf22-b443-47db-b3db-066ba2988f47'
discovery_project_id = '397c9f13-65a0-47bb-acf3-d9c9c482041f'


# Initialize Watson Discovery
authenticator = IAMAuthenticator(discovery_apikey)
discovery = DiscoveryV2(version='2021-08-01', authenticator=authenticator)
discovery.set_service_url(discovery_url)


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
        client = Elasticsearch(
            "https://9c4ec9125e554d83ba0af1f17c092b6a.us-central1.gcp.cloud.es.io:443",
            api_key="QVlZZWxaTUJmUVpiWmJUQXVtS0c6THN5R3R4RVlRNG1aMGZBZVFRLUxtQQ=="
            )
        client.delete_by_query(index="infinitheism", body={
            "query": {
                "match_all": {}  # Match all documents in the index
                }
            })
        print("Cleared all data")

        conn = sql.connect("videos.db")
        datas = []
        print("fetching datas from database")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM videos ORDER BY idno DESC")
        for i in cursor.fetchall():
            upload_to_elastic(i[0],i[3])
            datas.append({"vidno":i[0],"video":i[1],"transcript":i[3],"summary":i[2],"categories":ast.literal_eval(i[4]),"title":i[5]})
        if len(datas) < 0:
            datas.clear()
            datas.append({"vidno":"","video":"","transcript":"","summary":"","categories":"","title":""})
            return datas
        else:
            return datas
    except Exception as e:
        print(e)
    finally:
        if conn:
            conn.close()


#ithu vanthu summary generate pannum
def get_summary(transcript):
    try:
        print("-------------------PROCESS FIVE GENERATING SUMMARY-----------------")
        access_token = get_access_token()
        print(transcript)
        prompt = f"Generate a concise and meaningful summary of the following transcript, focusing on the most important points. Keep the summary brief and to the point, capturing the essence of the entire content in just a few sentences. transcript: {transcript}"
        print(prompt)
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


# ithu vanthu titles generate pannum
def get_title(title):
    try:
        print("-------------------PROCESS THREE GETTING TITLE-----------------")
        access_token = get_access_token()
        prompt = f"Title Extraction: Based on the provided transcript:{title}, extract a single title that encapsulates the main theme or key focus of the content. The title should be concise and reflect the core idea of the discussion, summarizing the most important aspect in a clear and direct manner."
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


def apply_semantic_search(searchinput, transcript):
    sentences = transcript.split(" ")
    newtranscript = ""
    paragraphs = [" ".join(sentences[i:i+5]) + "" for i in range(0, len(sentences), 5)]
    model = SentenceTransformer('all-MiniLM-L6-v2') 
    doc_embeddings = model.encode(paragraphs)
    query_embedding = model.encode([searchinput])
    cos_similarities = cosine_similarity(query_embedding, doc_embeddings)
    top_n = 5
    top_indices = np.argsort(cos_similarities[0])[::-1][:top_n]
    print("Query:", searchinput)
    print("finding....")
    for idx in top_indices:
        print(f"- {paragraphs[idx]} (Similarity: {cos_similarities[0][idx]:.4f})")
        newtranscript += paragraphs[idx] + ","
    return newtranscript

    

    

@app.route('/semantic', methods=['POST'])
def semantic_search():
    searchinput = request.form['searchinput']
    print(searchinput)
    try:
        conn = sql.connect("videos.db")
        datas = []
        print("fetching datas from database")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM videos ORDER BY idno DESC")
        for i in cursor.fetchall():
            similar = apply_semantic_search(searchinput, i[3])
            datas.append({"vidno":i[0],"transcript":similar})
            # datas.append({"vidno":i[0],"video":i[1],"transcript":i[3],"summary":i[2],"categories":ast.literal_eval(i[4]),"title":i[5]})
        if len(datas) < 0:
            datas.clear()
            datas.append({"vidno":"","video":"","transcript":"","summary":"","categories":"","title":""})
            print(datas)
            return jsonify(datas)
        else:
            print(datas)
            return jsonify(datas)
    except Exception as e:
        print(e)
    finally:
        if conn:
            conn.close()



def upload_to_elastic(idno, transcript):

    try:
        client = Elasticsearch(
            "https://9c4ec9125e554d83ba0af1f17c092b6a.us-central1.gcp.cloud.es.io:443",
            api_key="QVlZZWxaTUJmUVpiWmJUQXVtS0c6THN5R3R4RVlRNG1aMGZBZVFRLUxtQQ=="
            )
        
        print(f"uploading video of no:{idno}")
        document = {
            "idno":idno,
            "transcript":transcript
            }
        response = client.index(index="infinitheism",body=document)
        print(response)
        print(f"upload completed video of no:{idno}")
    except Exception as e:
        print(e)

def upload_to_whisper():
    try:
        API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3-turbo"
        headers = {"Authorization": "Bearer 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"}
        
        with open("myaudio_clean.wav", "rb") as f:
            data = f.read()
        response = requests.post(API_URL, headers=headers, data=data)
        print(response.json())
        return response.json()['text']
    except Exception as e:
        print(e)



@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file and allowed_file(file.filename):
        print("-----------------PROCESS ONE UPLOADING----------------------")
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
        transcript = upload_to_whisper()
    
        print("-------------------PROCESS TWO GETTING TRANSCRIPT-----------------")
    
        print(audio_text)
        a = ""
        t = ""
        for i in audio_text[0]:
            a = a + i['name'] 
            t = get_title(i['name']) 
            title.append({"time": i['start'], "title": t})
        print("-------------------PROCESS FOUR UPLOADING TRANSCRIPT TO DISCOVERY-----------------")
        
        #----------------------transcript end----------------------
        
        print("whisper analysing audio")

        summary = get_summary(transcript)

        try:
            print("-------------------PROCESS FIVE UPDATING DATABASE-----------------")
            conn = sql.connect("videos.db")
            cursor = conn.cursor()

            with open(f'uploads//{filename}','rb') as video_file:
                video_data = video_file.read()
            video_base64 = base64.b64encode(video_data).decode('utf-8')
            cursor.execute("INSERT INTO videos(video, summary, transcript, time, title) values(?, ?, ?, ?, ?)",(video_base64,summary, transcript, str(title), filename))
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

# def create_mp3(video_path):
#     print("Exportinga s mp3")
#     print(video_path)
#     video = mp.VideoFileClip(video_path)
#     audio_file = video.audio
#     audio_file.write_audiofile("myaudio.wav")
#     video.close()

#     audio = AudioSegment.from_wav("myaudio.wav")
#     audio = audio.set_frame_rate(8000)
#     audio = audio.normalize()
#     audio.export("myaudio_clean.mp3", format="mp3")

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
    

        keyfortext = "yjUngCaFd8p7uCy9s3q4imffMn1yuQ-kVeZaNmkwvVPc"
        urlfortext = "https://api.eu-gb.speech-to-text.watson.cloud.ibm.com/instances/95628b8f-b5cb-41b1-930b-e004516e5d6a"
        
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
