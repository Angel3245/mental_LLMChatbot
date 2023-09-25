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

import json
import os.path as osp
from typing import Union
from pathlib import Path

class Prompter(object):
    """
    A dedicated helper to manage templates and prompt building.
    """

    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "mentalbot"

        # Load template
        path = Path.cwd()
        file_name =  F"{str(path)}/file/templates/"+template_name+".json"

        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        
        with open(file_name) as fp:
            self.template = json.load(fp)

        if self._verbose:
            print(f"Using prompt template {template_name}: {self.template['description']}")

    def generate_prompt(
        self,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt"].format(
                input=input
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        # Returns text beetween response_split and stop_sequence (post-processing)
        return output.split(self.template["response_split"])[1].split(self.template["stop_sequence"])[0].strip()