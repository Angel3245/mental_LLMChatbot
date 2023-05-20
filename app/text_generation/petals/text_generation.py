import torch
from shared.model_classes import MODEL_CLASSES
from transformers import GenerationConfig, BloomTokenizerFast

from petals import DistributedBloomForCausalLM

class PetalsChatbot:
    def __init__(self, model_path, model_name="bigscience/bloom-petals"):
        self.model_path = model_path

        self.tokenizer = BloomTokenizerFast.from_pretrained(model_path)
        self.tokenizer.padding_side = "right" # Allow batched inference
        self.tokenizer.model_max_length = 256

        self.model = DistributedBloomForCausalLM.from_pretrained(model_name, tuning_mode="ptune", pre_seq_len=16)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #self.model.to(self.device)

        self.model.eval()

    def generate_response(self, input_text, max_length=1000, temperature=0.9, top_k=100):
        # Set prompt: <bos> input <sep>
        input_ids = self.tokenizer(f"User:{input_text}\nBot:", return_tensors='pt')['input_ids'].to(self.model.device)

        response_text = ""

        # Use model.generate() to generate the response
        with self.model.inference_session(max_length=512) as sess:
            while True:
                outputs = self.model.generate(
                    input_ids,
                    temperature=temperature,
                    top_k=top_k,
                    max_new_tokens=1,
                    do_sample=True,
                    session=sess,
                )

                bloom_answer_token = self.tokenizer.decode(outputs[0, -1:])
                print(bloom_answer_token,end="",flush=True)
                response_text += bloom_answer_token
                if bloom_answer_token == "\n":
                    break

                input_ids = None

        return response_text
