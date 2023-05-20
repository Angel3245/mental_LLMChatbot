import torch
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel

class PeftChatbot:
    def __init__(self, model_path, model_name="decapoda-research/llama-7b-hf"):
        self.model_path = model_path

        self.model = LlamaForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        self.model = PeftModel.from_pretrained(self.model, model_path, torch_dtype=torch.float16)

        self.tokenizer = LlamaTokenizer.from_pretrained(model_path)

        self.model.config.pad_token_id = self.tokenizer.pad_token_id = 0  # unk
        self.model.config.bos_token_id = 1
        self.model.config.eos_token_id = 2

        self.model.eval()
        #self.model = torch.compile(self.model)

    def generate_response(self, input_text, max_new_tokens=256, temperature=0.1, top_p=0.9, top_k=40, num_beams=4, repetition_penalty=1.1):

        # Set prompt
        prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n" + input_text + "\n\n### Response:\n"

        input_encodings = self.tokenizer(prompt, return_tensors='pt')
        input_ids = input_encodings['input_ids'].to(self.model.device)

        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty
        )
        
        with torch.inference_mode():
            # Use model.generate() to generate the response
            response = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                #do_sample=True,
                max_new_tokens=max_new_tokens,
            )

        # Decode the response from the model back into text
        decoded_output = self.tokenizer.decode(response.sequences[0])
        response = decoded_output.split("### Response:")[1].strip()

        return response
