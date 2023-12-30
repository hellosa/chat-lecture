import os
import time
import random
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


with gr.Blocks() as demo:
    DB_CHROMA_PATH = './tmp/db_chroma'

    # the layout
    with gr.Row():
        chatbot = gr.Chatbot(scale=2)
        with gr.Column(scale=1):
            upload_file = gr.File()
            btn = gr.Button(value="process")
            ready = gr.Label(value="not ready")

    # disable interactive before uploading the video
    msg = gr.Textbox(
        interactive=False,
        placeholder="please upload the video first",
        )
    clear = gr.ClearButton([msg, chatbot])

    # the process functions
    def process_video(video_file):
        os.makedirs("tmp", exist_ok=True)
        video_file_name = os.path.basename(video_file)

        audio_file = video_file_name.replace(".mp4", "_audio.wav")
        audio_file_path = "./tmp/{}".format(audio_file)

        srt_file = video_file_name.replace(".mp4", ".srt")
        srt_file_path = "./tmp/{}".format(srt_file)

        video2audio(video_file, audio_file_path)
        audio2text(audio_file_path, srt_file_path)
        txt2vdb(srt_file_path, DB_CHROMA_PATH)

        return "ready", gr.Textbox(interactive=True, placeholder="")

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

    btn.click(process_video, [upload_file], [ready, msg])
    msg.submit(respond, [msg, chatbot], [msg, chatbot])


def video2audio(video_file, audio_file):
    video = VideoFileClip(video_file)
    audio = video.audio
    audio.write_audiofile(audio_file)

def audio2text(audio_file, srt_file):
    r = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = r.record(source)
    result = r.recognize_whisper(audio, show_dict=True) # use whisper to stt

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
    demo.queue(max_size=10).launch()

