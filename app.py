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


# Stream text
def predict(message, chatbot, system_prompt="", temperature=0.9, max_new_tokens=4096):
    
    client = Client("https://ysharma-explore-llamav2-with-tgi.hf.space/")
    return client.predict(
			message,	# str in 'Message' Textbox component
            system_prompt,	# str in 'Optional system prompt' Textbox component
			temperature,	# int | float (numeric value between 0.0 and 1.0)
			max_new_tokens,	# int | float (numeric value between 0 and 4096)
			0.3,	# int | float (numeric value between 0.0 and 1)
			1,	# int | float (numeric value between 1.0 and 2.0)
			api_name="/chat"
    )
        


def transcribe(audio):
    whisper_client = Client("https://sanchit-gandhi-whisper-large-v2.hf.space/")

    return whisper_client.predict(
				audio[1],	# str (filepath or URL to file) in 'inputs' Audio component
				"transcribe",	# str in 'Task' Radio component
				api_name="/predict"
    )
    
def app_interface(input_type, message="", audio=None):
    if input_type == "Text":
        return predict(message)
    elif input_type == "Audio":
        transcribed_message = transcribe(audio.name)
        return predict(transcribed_message)

interface = gr.Interface(
    fn=app_interface,
    inputs=[
        gr.Radio(["Text", "Audio"], label="Input Type"),
        gr.Textbox(label="Message (if text selected)"),
        gr.Audio(label="Record (if audio selected)", type="file")
    ],
    outputs="text",
    live=True,
    title=TITLE,
    description=DESCRIPTION
)

interface.launch()
