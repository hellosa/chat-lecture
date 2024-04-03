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
from langchain.prompts import PromptTemplate


load_dotenv()

SUBTITLE_PATH = './tmp/subtitles/'
AUDIO_PATH = './tmp/audios/'
DB_CHROMA_PATH = './tmp/db_chroma/'


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
        if os.path.exists(srt_file_path):
            return [video_path, srt_file_path], gr.Textbox(interactive=True, placeholder="")

        # mp4 -> wav
        video2audio(video_path, audio_file_path)

        # wav -> srt
        audio2text(audio_file_path, srt_file_path)

        return [video_path, srt_file_path], gr.Textbox(interactive=True, placeholder="")

    def respond(message, chat_history):
        # VDB
        vdb = Chroma(persist_directory=DB_CHROMA_PATH, embedding_function=OpenAIEmbeddings())

        # LLM
        llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0, streaming=True)

        prompt_template = """
        我给你的是一个视频课程的字幕文件，你要根据整个字幕文件的内容回答我的 question
        不知道的事情要说不知道

        {context}

        Question: {question}
        """
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        chain_type_kwargs = {"prompt": PROMPT}
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vdb.as_retriever(),
            chain_type_kwargs=chain_type_kwargs)

        bot_message = qa_chain.run(message)

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
