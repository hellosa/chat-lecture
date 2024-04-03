import os
import time
import random
import hashlib
from datetime import timedelta

from dotenv import load_dotenv
import gradio as gr
from moviepy.editor import VideoFileClip
import speech_recognition as sr
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_models import ChatOllama


load_dotenv()

SUBTITLE_PATH = './tmp/subtitles/'
AUDIO_PATH = './tmp/audios/'
DB_CHROMA_PATH = './tmp/db_chroma/'

# global variables
global_chat_model = None
global_srt_path = None

with gr.Blocks() as demo:
    # LAYOUT
    with gr.Row():
        chatbot = gr.Chatbot()
        lecture = gr.PlayableVideo()

    # disable interactive before uploading the video
    msg = gr.Textbox(
        interactive=False,
        placeholder="please upload the video first",
        )
    with gr.Row():
        btn_start = gr.Button(value="start")
        btn_clean = gr.ClearButton([msg, chatbot])


    # the process functions
    def process_video(video_path):
        """
            mp4 -> wav -> srt
        """
        # create tmp dir
        os.makedirs(AUDIO_PATH, exist_ok=True)
        os.makedirs(SUBTITLE_PATH, exist_ok=True)

        # define the file path
        video_filename = os.path.basename(video_path)
        video_filename_md5 = md5_checksum(video_path)
        audio_file_path = "{}{}.wav".format(AUDIO_PATH, video_filename_md5)
        srt_file_path = "{}{}.srt".format(SUBTITLE_PATH, video_filename_md5)

        # if the srt file exists, return
        if os.path.exists(srt_file_path) == False:
            # mp4 -> wav
            video2audio(video_path, audio_file_path)

            # wav -> srt
            audio2text(audio_file_path, srt_file_path)

        global global_srt_path
        global_srt_path = srt_file_path

        return [video_path, srt_file_path], gr.Textbox(interactive=True, placeholder="")

    def respond(message, chat_history):
        global global_srt_path
        srt_path = global_srt_path

        system_prompt_template = SystemMessagePromptTemplate.from_template(
                """
                你是一个视频课程的助教，下面是一堂课程视频的字幕文件的内容，你要根据字幕的内容，对用户的问题进行回答。
                ---
                {context}
                ---
                """
            )
        with open(srt_path, 'r') as f:
            srt_content = f.read()
        system_prompt = system_prompt_template.format_messages(context=srt_content)

        prompt = ChatPromptTemplate.from_messages(
            [
                system_prompt[0],
                HumanMessagePromptTemplate.from_template("{question}"),
            ]
        )

        #chat_model = ChatOpenAI(model="moonshot-v1-32k", temperature=0)
        #chat_model = ChatOpenAI(model="gpt-4-0125-preview", temperature=0)
        chat_model = ChatOllama(model="", temperature=0)

        chain = (
            {"question": RunnablePassthrough()}
            | prompt 
            | chat_model
            | StrOutputParser()
        )

        bot_message = chain.invoke(message)

        chat_history.append((message, bot_message))
        return "", chat_history

    btn_start.click(process_video, [lecture], [lecture, msg])
    msg.submit(respond, [msg, chatbot], [msg, chatbot])


def md5_checksum(file_path):
    with open(file_path, 'rb') as fh:
        md5 = hashlib.md5()
        while chunk := fh.read(8192):
            md5.update(chunk)
    return md5.hexdigest()

def video2audio(video_file, audio_file):
    video = VideoFileClip(video_file)
    audio = video.audio
    audio.write_audiofile(audio_file)

def audio2text(audio_file, srt_file):
    r = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = r.record(source)
    # use whisper to stt
    # model options: tiny, base, small, medium, large
    # tiny for speed, large for accuracy
    result = r.recognize_whisper(audio, model='base', show_dict=True)

    with open(srt_file, 'w') as f:
        for i in result['segments']:
            srt_id = int(i['id']) + 1
            start_time = format_timedelta( float(i['start']) )
            end_time = format_timedelta( float(i['end']) )

            f.write('{}\n'.format(srt_id) )
            f.write('{} --> {}\n'.format(start_time, end_time) )
            f.write('{}\n\n'.format(i['text']) )

def txt2vdb(txt_file, vdb_dir):
    loader = TextLoader(txt_file)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
    splits = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    Chroma.from_documents(splits, embeddings, persist_directory=vdb_dir)

def format_timedelta(seconds):
    duration = timedelta(seconds=seconds)
    formatted_duration = '{:02}:{:02}:{:02},{:03}'.format(
        int(duration.total_seconds() // 3600),
        int(duration.total_seconds() % 3600 // 60),
        int(duration.total_seconds() % 60),
        int(duration.microseconds / 1000)
    )
    return formatted_duration


if __name__ == "__main__":
    demo.launch()
