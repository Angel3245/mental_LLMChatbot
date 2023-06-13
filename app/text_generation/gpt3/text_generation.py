import openai
from shared.prompter import Prompter

class GPT3TextGenerator:
    """ Class for creating responses from a GPT3 model

        :param base_model: name of the model
        :param template: template file to create prompts
        
    """
    def __init__(self, model, template="mentalbot"):
        self.model = model

        self.prompter = Prompter(template)

    def generate_response(self, input_text):
        # Set prompt
        prompt = self.prompter.generate_prompt(input_text)

        response = openai.Completion.create(
            model=self.model,
            prompt=prompt,
            stop=["User:", "Expert:", " END"],
            max_tokens=128)
        
        return response['choices'][0]['text'].strip()