import nltk
#¡nltk.download('wordnet')
#nltk.download('omw-1.4')
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
from textblob import TextBlob, Word
import re
import contractions


def preprocess(sentence):
    """ Apply input sentence several transformations to get a clean sentence:
        #1. Lowercase text
        #2. Remove whitespace
        #3. Remove numbers
        #4. Remove special characters
        #5. Remove emails
        #6. Remove text inside parentheses
        #7. Remove NAN
        #8. Remove weblinks
        #9. Expand contractions
        #10. Tokenize
    """
    sentence=contractions.fix(sentence)
    sentence=str(sentence)
    sentence = sentence.lower()
    sentence=sentence.replace('{html}',"")
    sentence=re.sub("[\(\[<].*?[\)\]>]", "", sentence)
    sentence=re.sub(r'http\S+', '',sentence)
    sentence = re.sub('[0-9]+', '', sentence)
    sentence = re.sub('\u200b', ' ', sentence)
    sentence = re.sub('\xa0', ' ', sentence)
    sentence = re.sub('\n', ' ', sentence)
    sentence = re.sub('\r', ' ', sentence)
    sentence = re.sub('-', ' ', sentence)
    rx = re.compile(r'([^\W\d_])\1{2,}')
    sentence = re.sub(r'[^\W\d_]+', lambda x: Word(rx.sub(r'\1\1', x.group())).correct() if rx.search(x.group()) else x.group(), sentence)

    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence)

    return " ".join(tokens)

def clean(sentence):
    """ Apply input sentence some transformations to get a clean sentence:
        #1. Remove NAN
        #2. Remove weblinks
        #3. Expand contractions
    """
    sentence=contractions.fix(sentence)
    sentence=str(sentence)
    sentence=sentence.replace('{html}',"")
    rem_url=re.sub(r'http\S+', '',sentence)
    rx = re.compile(r'([^\W\d_])\1{2,}')
    rem_num = re.sub(r'[^\W\d_]+', lambda x: Word(rx.sub(r'\1\1', x.group())).correct() if rx.search(x.group()) else x.group(), rem_url)

    return rem_num