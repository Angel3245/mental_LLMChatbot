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

    #Remove unnecesary spaces
    sentence=sentence.strip()

    #Expand contractions
    sentence=contractions.fix(sentence)

    sentence=str(sentence)

    #Remove web links
    sentence=sentence.replace('{html}',"")
    rem_url=re.sub(r'http\S+', '',sentence)
    rx = re.compile(r'([^\W\d_])\1{2,}')
    rem_num = re.sub(r'[^\W\d_]+', lambda x: Word(rx.sub(r'\1\1', x.group())).correct() if rx.search(x.group()) else x.group(), rem_url)

    return rem_num