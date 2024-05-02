class DataFormatter:
    def __init__(self, tokenizer, data_style):
        self.tokenizer = tokenizer
        self.data_style = data_style

        self.system_prompt = "A virtual assistant answers questions from a user based on the provided text."
        self.query_template = (
            lambda entity_type: f"What describes {entity_type} in the text?"
        )
        self.instruction_template = {
            "input": lambda text, query: (
                "[S2S] "
                + self.system_prompt
                + "\n\n### Instruction:\n"
                + query
                + "\n\n### Input:\n"
                + text
                + "\n\n<extra_id_0>"
            ).replace("\n", "[NEWLINE]"),
            "label": lambda target: ("### Response:\n" + target).replace(
                "\n", "[NEWLINE]"
            ),
        }
        self.conversation_template = {
            "input": lambda text, query: [
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {
                    "role": "user",
                    "content": f"Text: {text}",
                },
                {
                    "role": "assistant",
                    "content": "I've read this text.",
                },
                {
                    "role": "user",
                    "content": query,
                },
            ]
        }

    def format_output(self, output, model_config):
        if self.data_style == "instructions":
            output = output.replace("### Response:", "")
            output = output.replace("[NEWLINE]", "\n")
            output = output.strip()
        elif self.data_style == "conversations":
            if "LlamaForCausalLM" in model_config.architectures:
                output = output.split("[/INST]")[-1]
            pass
        else:
            raise ValueError("Invalid data style!")

        try:
            output = eval(output)
        except:
            output = []
        return output

    ### For inference ###
    def gen_data_with_format(self, **kwargs):
        """
        Generate data with the specified format.
        Parameters:
            - conversations: list of dict
            or
            - text: str
            - entity_type: str
        """
        convert_fn = f"gen_{self.data_style}_data"
        return getattr(self, convert_fn)(**kwargs)

    def gen_instructions_data(self, **kwargs):
        if "conversations" in kwargs.keys():
            pass
        else:
            text = kwargs["text"]
            entity_type = kwargs["entity_type"]
            sample = self.instruction_template["input"](
                text, self.query_template(entity_type)
            )

        return sample
    
    def gen_conversations_data(self, **kwargs):
        if "conversations" in kwargs.keys():
            pass
        else:
            text = kwargs["text"]
            entity_type = kwargs["entity_type"]
            conv = self.conversation_template["input"](
                text, self.query_template(entity_type)
            )
            sample = self.tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=True
            )
        return sample

    ### For training ###
