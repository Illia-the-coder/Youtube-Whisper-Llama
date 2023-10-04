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
def predict(message, chatbot, system_prompt="", temperature=0.9, max_new_tokens=4096, top_p=0.6, repetition_penalty=1.0,):
    
    client = Client("https://ysharma-explore-llamav2-with-tgi.hf.space/")
    return client.predict(
			message,	# str in 'Message' Textbox component
            system_prompt,	# str in 'Optional system prompt' Textbox component
			temperature,	# int | float (numeric value between 0.0 and 1.0)
			max_new_tokens,	# int | float (numeric value between 0 and 4096)
			0.3,	# int | float (numeric value between 0.0 and 1)
			1,	# int | float (numeric value between 1.0 and 2.0)
			api_name="/chat_1"
    )
        
additional_inputs=[
    gr.Textbox("", label="Optional system prompt"),
    gr.Slider(
        label="Temperature",
        value=0.9,
        minimum=0.0,
        maximum=1.0,
        step=0.05,
        interactive=True,
        info="Higher values produce more diverse outputs",
    ),
    gr.Slider(
        label="Max new tokens",
        value=4096,
        minimum=0,
        maximum=4096,
        step=64,
        interactive=True,
        info="The maximum numbers of new tokens",
    ),
    gr.Slider(
        label="Top-p (nucleus sampling)",
        value=0.6,
        minimum=0.0,
        maximum=1,
        step=0.05,
        interactive=True,
        info="Higher values sample more low-probability tokens",
    ),
    gr.Slider(
        label="Repetition penalty",
        value=1.2,
        minimum=1.0,
        maximum=2.0,
        step=0.05,
        interactive=True,
        info="Penalize repeated tokens",
    )
]



# Gradio Demo 
with gr.Blocks(theme=gr.themes.Base()) as demo:
    gr.DuplicateButton()
    gr.ChatInterface(predict, title=title,additional_inputs=additional_inputs, description=description, css=css, examples=examples) 
        
demo.queue().launch(debug=True)