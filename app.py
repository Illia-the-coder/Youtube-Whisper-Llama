import gradio as gr
from gradio_client import Client




title = "Llama2 70B Chatbot"
description = """
This Space demonstrates model [Llama-2-70b-chat-hf](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) by Meta, a Llama 2 model with 70B parameters fine-tuned for chat instructions. 
| Model | Llama2 | Llama2-hf | Llama2-chat | Llama2-chat-hf |
|---|---|---|---|---|
| 70B | [Link](https://huggingface.co/meta-llama/Llama-2-70b) | [Link](https://huggingface.co/meta-llama/Llama-2-70b-hf) | [Link](https://huggingface.co/meta-llama/Llama-2-70b-chat) | [Link](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) |

"""
css = """.toast-wrap { display: none !important } """
examples=[
    ['Hello there! How are you doing?'],
    ['Can you explain to me briefly what is Python programming language?'],
    ['Explain the plot of Cinderella in a sentence.'],
    ['How many hours does it take a man to eat a Helicopter?'],
    ["Write a 100-word article on 'Benefits of Open-Source in AI research'"],
    ]

whisper_client = Client("https://sanchit-gandhi-whisper-large-v2.hf.space/")
text_client = Client("https://ysharma-explore-llamav2-with-tgi.hf.space/")


def transcribe(wav_path):
    
    return whisper_client.predict(
				wav_path,	# str (filepath or URL to file) in 'inputs' Audio component
				"transcribe",	# str in 'Task' Radio component
				api_name="/predict"
)
    

# Chatbot demo with multimodal input (text, markdown, LaTeX, code blocks, image, audio, & video). Plus shows support for streaming text.


def add_text(history, text):
    history = [] if history is None else history
    history = history + [(text, None)]
    return history, gr.update(value="", interactive=False)


def add_file(history, file):
    history = [] if history is None else history
    text = transcribe(
        file
    )
    
    history = history + [(text, None)]
    return history



def bot(history, system_prompt=""):    
    history = [] if history is None else history

    if system_prompt == "":
        system_prompt = system_message
        
    history[-1][1] = ""
    for character in text_client.submit(
                    history,
                    system_prompt,
                    temperature,
                    4096,
                    temperature,
                    repetition_penalty,
                    api_name="/chat"
    ):
        history[-1][1] = character
        yield history  

    


with gr.Blocks(title=title) as demo:
    gr.Markdown(DESCRIPTION)
    
    
    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False,
    )

    with gr.Row():
        txt = gr.Textbox(
            scale=3,
            show_label=False,
            placeholder="Enter text and press enter, or speak to your microphone",
            container=False,
        )
        txt_btn = gr.Button(value="Submit text",scale=1)
        btn = gr.Audio(source="microphone", type="filepath", scale=4)
        gradio.Examples(examples, txt_btn)
        
    with gr.Row():
        audio = gr.Audio(type="numpy", streaming=True, autoplay=True, label="Generated audio response", show_label=True)

    clear_btn = gr.ClearButton([chatbot, audio])
    
    txt_msg = txt_btn.click(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
        bot, chatbot, chatbot
    ).

    txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
        bot, chatbot, chatbot
    ).
    
    txt_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False)
    
    file_msg = btn.stop_recording(add_file, [chatbot, btn], [chatbot], queue=False).then(
        bot, chatbot, chatbot
    ).
    

    gr.Markdown("""
This Space demonstrates how to speak to a chatbot, based solely on open-source models.
It relies on 3 models:
1. [Whisper-large-v2](https://huggingface.co/spaces/sanchit-gandhi/whisper-large-v2) as an ASR model, to transcribe recorded audio to text. It is called through a [gradio client](https://www.gradio.app/docs/client).
2. [Llama-2-70b-chat-hf](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) as the chat model, the actual chat model. It is also called through a [gradio client](https://www.gradio.app/docs/client).
""")
demo.queue()
demo.launch(debug=True)