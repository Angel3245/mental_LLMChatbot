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

from clean_data import clean

def ask_chatbot(text_generator):
    """
    Ask the user for an input, clean and check the input sentence and passes it to the text_generator to create an answer.
    It uses the command-line to communicate with the user.
    
    :param text_generator: Model used to generate the answer
    """
    while True:
        user_input = input("User: ")
        if user_input.lower() == 'exit':
            break

        # Clean input sentence
        user_input = clean(user_input)

        # Analyze input and create predefined responses if needed
        # Check if user_input words are higher than 3
        if(len(user_input.split()) < 3):
            response = "Sorry, I did not understand you. Could you explain it better?"
        else:
            response = text_generator.generate_response(user_input)
        
        print(f"Chatbot: {response}")