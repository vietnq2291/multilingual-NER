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
        ),
        "parameters": {
            "max_length": 1024,
            "temperature": 0,
        }
    }
    response = requests.request(
        "POST",
        f"https://api-inference.huggingface.co/models/{model_path}",
        headers=headers,
        data=json.dumps(data)
    )
    return json.loads(response.content.decode("utf-8"))[0]['generated_text']

def get_model_configs():
    with open("ner/pipe_config.json", "r") as f:
        model_configs = json.load(f)
    return model_configs

# Function to extract model info
def get_model_info(model_name, model_list):
    model_base = model_list[model_name]["base_model_id"]
    finetuning_data = model_list[model_name]["finetuning_data"]


    formatted_data = []
    for stage_data in finetuning_data:
        if len(stage_data) > 1:
            data = '\n- '.join([""] + stage_data)
            data = data.strip()
            formatted_data.append(f"Mixed data:\n{data}")
        else:
            formatted_data.append(stage_data[0])

    if len(formatted_data) > 1:
        formatted_data = [f"({i+1}) {data}\n" for i, data in enumerate(formatted_data)]
    elif len(finetuning_data[0]) == 1:
        formatted_data = "\n- " + formatted_data[0]

    formatted_data = "".join(formatted_data).strip()

    return f"Model: {model_base}\n\nFinetuning Data:\n{formatted_data}"

def main():
    data_formatter = DataFormatter()
    model_configs = get_model_configs()

    # Create the Gradio interface
    gr.close_all()
    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # Open-domain Named Entity Recognition for Vietnamese text
            """
        )
        with gr.Row():
            model_name = gr.Dropdown(choices=model_configs.keys(), label="Select Model")

        with gr.Row():
            model_info = gr.Textbox(label="Model Information", interactive=False)

        with gr.Row():
            input_text = gr.Textbox(label="Input Text", placeholder="Enter text here...")

        with gr.Row():
            entity_type = gr.Textbox(label="Entity Type", placeholder="Enter entity type here...")

        with gr.Row():
            submit_button = gr.Button("Submit")

        with gr.Row():
            output_text = gr.Textbox(label="Output", interactive=False)

        def update_model_info(model_name):
            return get_model_info(model_name, model_configs)

        model_name.change(fn=update_model_info, inputs=model_name, outputs=model_info)

        submit_button.click(
            fn=lambda model_name, input_text, entity_type: get_completion(
                model_configs[model_name]['model_id'], input_text, entity_type, data_formatter
            ),
            inputs=[model_name, input_text, entity_type],
            outputs=output_text
        )

    # Launch the demo
    demo.launch(share=False)

if __name__ == "__main__":
    main()
