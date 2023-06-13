import torch
import sys
from transformers import GenerationConfig, BloomForCausalLM, BloomTokenizerFast
from peft import PeftModel, PeftConfig
from shared.prompter import Prompter

class BloomPeftTextGenerator:
    """ Class for creating responses from a trained BLOOM model

        :param model_path: path of the trained model in disk
        :param template: template file to create prompts
        
    """
    def __init__(self, model_path, template="mentalbot"):
        self.model_path = model_path

        config = PeftConfig.from_pretrained(self.model_path)

        self.model = BloomForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            load_in_8bit=True,
            return_dict=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        self.model = PeftModel.from_pretrained(self.model, model_path)

        self.tokenizer = BloomTokenizerFast.from_pretrained(config.base_model_name_or_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left" # Allow batched inference

        self.prompter = Prompter(template)

        self.model = self.model.eval()
        
        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)

    def generate_response(self, input_text, max_new_tokens=256, temperature=0.9, top_p=0.9, repetition_penalty=1.1):

        # Set prompt
        prompt = self.prompter.generate_prompt(input_text)

        input_encodings = self.tokenizer(prompt, return_tensors='pt')
        input_ids = input_encodings['input_ids'].to(self.model.device)

        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )
        
        with torch.inference_mode():
            # Use model.generate() to generate the response
            response = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )

        # Decode the response from the model back into text
        decoded_output = self.tokenizer.decode(response.sequences[0][ : -1])

        response = self.prompter.get_response(decoded_output)

        return response