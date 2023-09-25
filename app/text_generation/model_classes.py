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

model_types = {
    "gpt2": ["gpt2"],
    "bloom": ["bigscience/bloom-560m", "bigscience/bloom-1b7", "bigscience/bloom-7b1"],
    "petals": ["bigscience/bloom-7b1-petals"],
    "llama": ["decapoda-research/llama-7b-hf", "decapoda-research/llama-13b-hf", "chavinlo/alpaca-native"]
}

class ModelDispatcher:
    """ Class for getting model type from a model name"""
    def get_model_type(model_name: str):
        """ Get type from model name

        :param model_name: name of the model
        
        """
        model_type = [i for i in model_types if model_name in model_types[i]][0]

        if(model_type == None):
            raise ValueError('model ' + model_name + ' not supported')
        
        return model_type
    
    def get_supported_types():
        """ Get supported model types from model_types"""
        return [item for list in model_types.values() for item in list] 
