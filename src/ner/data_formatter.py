class DataFormatter:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

        self.system_prompt = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
        )
        self.query_template = (
            lambda entity_type: f"What describes {entity_type} in the text?"
        )
        self.instructions_template = {
            "input": lambda text, query: (
                self.system_prompt
                + "### Instruction: "
                + query
                + "\n\n### Input: "
                + text
                + "\n\n### Response:\n"
            ),
            "label": lambda target: target,
        }
    
    def format_output(self, output):
        try:
            output = eval(output)
        except:
            output = []
        return output
    
    def format_instruction_input(self, text, entity_type):
        instruction_input = self.instructions_template["input"](
            text, self.query_template(entity_type)
        )
        return instruction_input
        