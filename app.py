import os
import logging
from typing import Any, List, Mapping, Optional
from langchain.llms import HuggingFaceHub
from gradio_client import Client
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import streamlit as st
from pytube import YouTube
# import replicate

DESCRIPTION = """
<div class="max-w-full overflow-auto">
    <table>
        <thead>
            <tr>
                <th>Model</th>
                <th>Llama2</th>
                <th>Llama2-hf</th>
                <th>Llama2-chat</th>
                <th>Llama2-chat-hf</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>7B</td>
                <td><a rel="noopener nofollow" href="https://huggingface.co/meta-llama/Llama-2-7b">Link</a></td>
                <td><a rel="noopener nofollow" href="https://huggingface.co/meta-llama/Llama-2-7b-hf">Link</a></td>
                <td><a rel="noopener nofollow" href="https://huggingface.co/meta-llama/Llama-2-7b-chat">Link</a></td>
                <td><a rel="noopener nofollow" href="https://huggingface.co/meta-llama/Llama-2-7b-chat-hf">Link</a></td>
            </tr>
            <tr>
                <td>13B</td>
                <td><a rel="noopener nofollow" href="https://huggingface.co/meta-llama/Llama-2-13b">Link</a></td>
                <td><a rel="noopener nofollow" href="https://huggingface.co/meta-llama/Llama-2-13b-hf">Link</a></td>
                <td><a rel="noopener nofollow" href="https://huggingface.co/meta-llama/Llama-2-13b-chat">Link</a></td>
                <td><a rel="noopener nofollow" href="https://huggingface.co/meta-llama/Llama-2-13b-chat-hf">Link</a></td>
            </tr>
            <tr>
                <td>70B</td>
                <td><a rel="noopener nofollow" href="https://huggingface.co/meta-llama/Llama-2-70b">Link</a></td>
                <td><a rel="noopener nofollow" href="https://huggingface.co/meta-llama/Llama-2-70b-hf">Link</a></td>
                <td><a rel="noopener nofollow" href="https://huggingface.co/meta-llama/Llama-2-70b-chat">Link</a></td>
                <td><a rel="noopener nofollow" href="https://huggingface.co/meta-llama/Llama-2-70b-chat-hf">Link</a></td>
            </tr>
        </tbody>
    </table>
</div>

"""

models = {
    "Llama2-70b": {
        "model_link": "https://huggingface.co/meta-llama/Llama-2-70b",
        "chat_link": "https://ysharma-explore-llamav2-with-tgi.hf.space/",
    },
    "Llama2-13b": {
        "model_link": "https://huggingface.co/meta-llama/Llama-2-13b",
        "chat_link": "https://huggingface-projects-llama-2-13b-chat.hf.space/",
    }
}

DESCRIPTION = """
Welcome to the **YouTube Video Chatbot** powered by Llama-2 models. Here's what you can do:
- **Transcribe & Understand**: Provide any YouTube video URL, and our system will transcribe it. Our advanced NLP model will then understand the content, ready to answer your questions.
- **Ask Anything**: Based on the video's content, ask any question, and get instant, context-aware answers.
To get started, simply paste a YouTube video URL and select a model in the sidebar, then start chatting with the model about the video's content. Enjoy the experience!
"""
st.title("YouTube Video Chatbot")
st.markdown(DESCRIPTION)

def get_video_title(youtube_url: str) -> str:
    yt = YouTube(youtube_url)
    embed_url = f"https://www.youtube.com/embed/{yt.video_id}"
    embed_html = f'<iframe src="{embed_url}" frameborder="0" allowfullscreen></iframe>'
    return yt.title, embed_html

def transcribe_video(youtube_url: str, path: str) -> List[Document]:
    """
    Transcribe a video and return its content as a Document.
    """
    logging.info(f"Transcribing video: {youtube_url}")
    client = Client("https://sanchit-gandhi-whisper-jax.hf.space/")
    result = client.predict(youtube_url, "translate", True, fn_index=7)
    return [Document(page_content=result[1], metadata=dict(page=1))]

def predict(
    message: str, system_prompt: str = "", model_url: str = models["Llama2-70b"]["chat_link"]
) -> Any:
    """
    Predict a response using a client.
    """
    client = Client(model_url)
    response = client.predict(message, system_prompt, 0.7, 4096, 0.5, 1.2, api_name="/chat_1")
    return response

PATH = os.path.join(os.path.expanduser("~"), "Data")

def initialize_session_state():
    if "youtube_url" not in st.session_state:
        st.session_state.youtube_url = ""
    if "model_choice" not in st.session_state:
        st.session_state.model_choice = "Llama2-70b"
    if "setup_done" not in st.session_state:
        st.session_state.setup_done = False
    if "doneYoutubeurl" not in st.session_state:
        st.session_state.doneYoutubeurl = ""

def sidebar():
    with st.sidebar:
        st.markdown("Enter the YouTube Video URL below🔗")
        st.session_state.youtube_url = st.text_input("YouTube Video URL:")

        model_choice = st.radio("Choose a Model:", list(models.keys()))
        st.session_state.model_choice = model_choice

        if st.session_state.youtube_url:
            # Get the video title
            video_title, embed_html = get_video_title(st.session_state.youtube_url)
            st.markdown(f"### {video_title}")

            # Embed the video
            st.markdown(embed_html, unsafe_allow_html=True)
            
      

sidebar()
initialize_session_state()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-l6-v2")

prompt = PromptTemplate(
    template="""Given the context about a video. Answer the user in a friendly and precise manner.
    Context: {context}
    Human: {question}
    AI:""",
    input_variables=["context", "question"]
)

class LlamaLLM(LLM):
    """
    Custom LLM class.
    """

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None) -> str:
        model_link = models[st.session_state.model_choice]["chat_link"]
        response = predict(prompt, model_url=model_link)
        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}

# Check if a new YouTube URL is provided
if st.session_state.youtube_url != st.session_state.doneYoutubeurl:
    st.session_state.setup_done = False

if st.session_state.youtube_url and not st.session_state.setup_done:
    with st.status("Transcribing video..."):
        data = transcribe_video(st.session_state.youtube_url, PATH)

    with st.status("Running Embeddings..."):
        docs = text_splitter.split_documents(data)

        docsearch = FAISS.from_documents(docs, embeddings)
        retriever = docsearch.as_retriever()
        retriever.search_kwargs["distance_metric"] = "cos"
        retriever.search_kwargs["k"] = 4
    with st.status("Running RetrievalQA..."):
        llama_instance = LlamaLLM()
        st.session_state.qa = RetrievalQA.from_chain_type(llm=llama_instance, chain_type="stuff", retriever=retriever, chain_type_kwargs={"prompt": prompt})

    st.session_state.doneYoutubeurl = st.session_state.youtube_url
    st.session_state.setup_done = True  # Mark the setup as done for this URL

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=("🧑‍💻" if message["role"] == "human" else "🦙")):
        st.markdown(message["content"])

textinput = st.chat_input("Ask anything about the video...")

if prompt := textinput:
    st.chat_message("human", avatar="🧑‍💻").markdown(prompt)
    st.session_state.messages.append({"role": "human", "content": prompt})
    with st.status("Requesting Client..."):
        video_title, _ = get_video_title(st.session_state.youtube_url)
        additional_context = f"Given the context about a video titled '{video_title}' available at '{st.session_state.youtube_url}'."
        response = st.session_state.qa.run(prompt + " " + additional_context)
    with st.chat_message("assistant", avatar="🦙"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
