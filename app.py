
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

# examples=[
#     ['Hello there! How are you doing?'],
#     ['Can you explain to me briefly what is Python programming language?'],
#     ['Explain the plot of Cinderella in a sentence.'],
#     ['How many hours does it take a man to eat a Helicopter?'],
#     ["Write a 100-word article on 'Benefits of Open-Source in AI research'"],
#     ]




def transcribe(wav_path):
    whisper_client = Client("https://sanchit-gandhi-whisper-large-v2.hf.space/")
    
    return whisper_client.predict(
				wav_path,	# str (filepath or URL to file) in 'inputs' Audio component
				"transcribe",	# str in 'Task' Radio component
				api_name="/predict"
    )


# Stream text
def predict(message, chatbot, system_prompt="", temperature=0.9, max_new_tokens=4096):
    
    client = Client("https://ysharma-explore-llamav2-with-tgi.hf.space/")
    return client.predict(transcribe(message),	# str in 'Message' Textbox component
            system_prompt,	# str in 'Optional system prompt' Textbox component
			temperature,	# int | float (numeric value between 0.0 and 1.0)
			max_new_tokens,	# int | float (numeric value between 0 and 4096)
			0.3,	# int | float (numeric value between 0.0 and 1)
			1,	# int | float (numeric value between 1.0 and 2.0)
			api_name="/chat_1")
        
demo = gr.Interface(predict, title=title, inputs="microphone", outputs="text", description=description, css=css,theme=gr.themes.Base())#, examples=examples)
demo.launch()
