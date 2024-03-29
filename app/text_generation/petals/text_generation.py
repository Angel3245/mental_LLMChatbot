# Copyright (C) 2023  Jose Ángel Pérez Garrido
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import torch
from transformers import GenerationConfig, BloomTokenizerFast

from petals import DistributedBloomForCausalLM

class PetalsTextGenerator:
    def __init__(self, model_path, model_name="bigscience/bloom-petals", template="alpaca"):
        self.model_path = model_path

        self.tokenizer = BloomTokenizerFast.from_pretrained(model_path)
        self.tokenizer.padding_side = "left" # Allow batched inference
        self.tokenizer.model_max_length = 256

        self.model = DistributedBloomForCausalLM.from_pretrained(model_name)
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
