import string
import spacy
import re
import ast
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
nltk.download('stopwords')
nlp = spacy.load("en_core_web_sm")
tweet_tokenizer = TweetTokenizer()

punctuations = list(string.punctuation)
punctuations.remove('#')
punctuations.remove('<')
punctuations.remove('.')
punctuations.remove('*')
punctuations.remove(',')
punctuations.remove('>')
punctuations.remove("'")
punctuations.append('..')
punctuations.append('...')
punctuations.append('…')
punctuations.append('–')
punctuations.append('・')

def split_sentence(sentence):
    # Define a set of punctuation marks
    sentence = sentence.replace('\n', '.')
    punctuation = set(string.punctuation)- set(['<', '>', '@', '#',"'", '%', '*'])
    #punctuation.update(set(['..']))

    # Initialize the output list
    sentences = []

    # Split the sentence on punctuation marks
    current_sentence = ''
    for char in sentence:
        if char in punctuation:
            # Add the current sentence to the list
            if current_sentence:
                sentences.append(current_sentence.strip())
            # Start a new sentence
            current_sentence = ''
        else:
            # Add the character to the current sentence
            current_sentence += char

    # Add the final sentence to the list
    if current_sentence:
        sentences.append(current_sentence.strip())

    # Return the list of sentences
    return sentences







def apply_lemmatization(texts):
    """ Apply lemmatization with post tagging operations through Stanza.
    Lower case """

    processed_text = []

    for testo in texts:
        rev = []
        testo = testo.translate(str.maketrans('', '', "".join(punctuations)))

        doc = nlp(testo)

        hashtag = False
        for token in doc:
          # lemmatization only if it's not an hashtag
          if not(hashtag):
            if str(token) not in punctuations:
              rev.append((token.lemma_))
          else:
            rev.append(str(token))
          if str(token) == '#':
            hashtag = True
          else:
            hashtag = False

        # recompone hashtags, mentions, <user> and <url>
        rev = " ".join(rev).replace("# ", '#').replace("< ", '<').replace(" >", '>').replace(" @", '@')

        processed_text.append(rev)

    return processed_text



def preprocessing(text, language = 'en', additional_stopwords=["'", "."]):
  # stopword removal and tokenization
  #text = text.lower()
  if language == 'ar':
    stop_words = set(stopwords.words('arabic'))
  else:
    stop_words = set(stopwords.words('english'))

  stop_words.add("'")
  stop_words.add(".")
  #for s in additional_stopwords:
  #  stop_words.add(s)

  text = text.lower()
  word_tokens = tweet_tokenizer.tokenize(text)
  # converts the words in word_tokens to lower case and then checks whether
  #they are present in stop_words or not
  filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
  #with no lower case conversion

  return filtered_sentence

# tokenizzo i termini in sentences:
def sentences_tokenizer(sentence_list, language = 'en'):
  tokens_list =[]
  for text in sentence_list:
    tokens_list.append(preprocessing(text, language))
    #tokens_list.append(tweet_tokenizer.tokenize(text.lower()))
  return tokens_list

def flatten_list(sentences):
    return [item for sublist in sentences for item in sublist]

def clear_tokens(tokens_lists):
  clear_list=[]
  for tokens_list in tokens_lists:
    unire = False
    clear_sublist = []
    for token in tokens_list:
      if unire and token != '*':
        a = clear_sublist.pop()
        clear_sublist.append(a + token)
        unire = False
      else:
        if not (token in ['€',  '£', '<url>', '<user>', '"', 'r', 'n', '*', '🇧', '’', 'RT', 'rt', '#', '🇪', '🇺','´', '️'] or token.isdigit() or (len(token) == 1 and token.isalpha())):
          clear_sublist.append(token)

        if token == '*' and clear_sublist:
          a = clear_sublist.pop()
          clear_sublist.append(a + token)
          unire = True
        if token == '🇧' and clear_sublist:
          a = clear_sublist.pop()
          clear_sublist.append(a + token)
        if token == '🇺' and clear_sublist:
          a = clear_sublist.pop()
          clear_sublist.append(a + token)
        if token == '🇪' and clear_sublist:
          a = clear_sublist.pop()
          if a == '🇩':
            clear_sublist.append(a + token)
          else:
            clear_sublist.append(a)
            clear_sublist.append(token)

    if clear_sublist:
      clear_list.append(clear_sublist)
  return clear_list

def adjust_split(list_sentences):
  previous_last = ''
  adjusted=[]
  unire= False
  for x in list_sentences:
    if x:
      if unire:
        unire= False
        last_split = adjusted.pop()
        last_split = last_split + ' '+x
        adjusted.append(last_split)
      else:
        # unire 'U.S.'
        last_split = x
        if x.lower()=='s' and previous_last.lower()=='u':
          last_split = adjusted.pop()
          last_split = last_split + '.'+x+'.'
          unire = True

        # unire 'U.K.'
        elif x.lower()=='k' and previous_last.lower()=='u':
          last_split = adjusted.pop()
          last_split = last_split + '.'+x+'.'
          unire = True

        # unire 'P.C.'
        elif x.lower()=='c' and previous_last.lower()=='p':
          last_split = adjusted.pop()
          last_split = last_split + '.'+x+'.'
          unire = True

        #rimuovere i numeri
        if not x.isdigit():
          adjusted.append(last_split)

      previous_last = x[len(x) - 1]
  return adjusted

def get_dataset_labels(df, columns = ['original_text','hard_label','soft_label_0','soft_label_1', 'disagreement']):
  """
  df: dataframe to elaborate
  colums: list of output columns
  ______________________________
  Extract two columns from the soft-label column to represent disagreement on the positive and negative label.
  Add a "disagreemen" column with boolean values (1 for agreement, 0 for disagreement)
  Rename the column "text" in "original text" to distiguish with the token-column "text"
  """
  df['soft_label_1']= df['soft_label'].apply(lambda x: x['1'])
  df['soft_label_0']= df['soft_label'].apply(lambda x: x['0'])
  df['disagreement'] = df['soft_label_0'].apply(lambda x : int(x==0 or x==1))
  df.rename({'text': 'original_text'}, axis=1, inplace=True)
  return df[columns]