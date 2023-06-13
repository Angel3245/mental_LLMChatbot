import openai
from shared.prompter import Prompter

class GPT3TextGenerator:
    """ Class for creating responses from a GPT3 model

        :param base_model: name of the model
        :param template: template file to create prompts
        :param cutoff_len: max length of sentences
        
    """
    def __init__(self, model, template="mentalbot", cutoff_len = 128):
        self.model = model

        self.cutoff_len = cutoff_len

        self.prompter = Prompter(template)

    def generate_response(self, input_text):
        # Set prompt
        prompt = self.prompter.generate_prompt(input_text)

        response = openai.Completion.create(
            model=self.model,
            prompt=prompt,
            stop=["User:", "Expert:", " END"],
            max_tokens=self.cutoff_len)
        
        return response['choices'][0]['text'].strip()