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

from textblob import Word
import re
import contractions
import redditcleaner

def clean(sentence):
    """ Apply input sentence some transformations to get a clean sentence:
        #1. Remove NAN
        #2. Remove weblinks
        #3. Expand contractions
        #4. Clean Reddit text data
        #5. Remove double spaces
    """
    sentence=str(sentence)
    
    # Remove double spaces
    while "\n\n" in sentence:
        sentence = sentence.replace("\n\n","\n")
    
    # Clean Reddit text data
    sentence=redditcleaner.clean(sentence)

    # Remove unnecesary spaces
    sentence=sentence.strip()

    # Expand contractions
    sentence=contractions.fix(sentence)

    sentence=str(sentence)

    # Remove web links
    sentence=sentence.replace('{html}',"")
    rem_url=re.sub(r'http\S+', '',sentence)
    rx = re.compile(r'([^\W\d_])\1{2,}')
    rem_num = re.sub(r'[^\W\d_]+', lambda x: Word(rx.sub(r'\1\1', x.group())).correct() if rx.search(x.group()) else x.group(), rem_url)

    return rem_num