import torch
import sys
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel, PeftConfig
from shared.prompter import Prompter

class PeftChatbot:
    def __init__(self, model_path, template="alpaca"):
        self.model_path = model_path

        config = PeftConfig.from_pretrained(self.model_path)

        self.model = LlamaForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        self.model = PeftModel.from_pretrained(self.model, model_path, torch_dtype=torch.float16)

        self.tokenizer = LlamaTokenizer.from_pretrained(config.base_model_name_or_path)

        self.model.config.pad_token_id = self.tokenizer.pad_token_id = 0  # unk
        self.model.config.bos_token_id = 1
        self.model.config.eos_token_id = 2

        self.prompter = Prompter(template)

        self.model = self.model.eval()
        
        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)

    def generate_response(self, input_text, max_new_tokens=256, temperature=0.1, top_p=0.9, top_k=40, num_beams=4, repetition_penalty=1.1):

        # Set prompt
        prompt = self.prompter.generate_prompt("The following is a conversation with a mental health expert. Expert helps the User by providing emotional support, it also helps solving doubts related to mental health by providing the best option. If the expert does not know the answer to a question, it truthfully says it does not know. The expert is conversational, optimistic, flexible, empathetic, creative and humanly in generating responses.",input_text)

        input_encodings = self.tokenizer(prompt, return_tensors='pt')
        input_ids = input_encodings['input_ids'].to(self.model.device)

        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            #top_k=top_k,
            #num_beams=num_beams,
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
        print(decoded_output)
        response = self.prompter.get_response(decoded_output)

        return response
