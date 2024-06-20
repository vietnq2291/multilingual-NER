import requests, json, os
import gradio as gr
from ner.data_formatter import DataFormatter
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
hf_api_key = os.environ['HF_API_KEY']


def get_completion(model_path, input_text, entity_type, data_formatter): 
    headers = {
      "Authorization": f"Bearer {hf_api_key}",
      "Content-Type": "application/json"
    }

    data = {
        "inputs": data_formatter.format_instruction_input(
            input_text, entity_type
        )
    }
    response = requests.request(
        "POST",
        f"https://api-inference.huggingface.co/models/{model_path}",
        headers=headers,
        data=json.dumps(data)
    )
    return json.loads(response.content.decode("utf-8"))[0]['generated_text']

def get_model_list():
    with open("ner/pipe_config.json", "r") as f:
        configs = json.load(f)
        models = {k: configs[k]['model_id'] for k, v in configs.items()}
    return models


def main():
    data_formatter = DataFormatter()
    models = get_model_list()

    # Create the Gradio interface
    gr.close_all()
    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # Open-domain Named Entity Recognition for Vietnamese text
            """
        )
        with gr.Row():
            model_name = gr.Dropdown(choices=models.keys(), label="Select Model")
        
        with gr.Row():
            input_text = gr.Textbox(label="Input Text", placeholder="Enter text here...")
        
        with gr.Row():
            entity_type = gr.Textbox(label="Entity Type", placeholder="Enter entity type here...")

        with gr.Row():
            submit_button = gr.Button("Submit")

        with gr.Row():
            output_text = gr.Textbox(label="Output", interactive=False)

        submit_button.click(
            fn=lambda model_name, input_text, entity_type: get_completion(
                models[model_name], input_text, entity_type, data_formatter
            ),
            inputs=[model_name, input_text, entity_type],
            outputs=output_text
        )

    # Launch the demo
    demo.launch(share=False)

if __name__=="__main__":
    main()