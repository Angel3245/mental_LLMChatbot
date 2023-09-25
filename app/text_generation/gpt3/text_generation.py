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