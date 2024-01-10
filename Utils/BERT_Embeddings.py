import torch
import pandas as pd
def bert_text_preparation(text, tokenizer):
  """
  Preprocesses text input in a way that BERT can interpret.
  """
  encoding = tokenizer(text)
  tokenized_text = encoding.tokens()
  indexed_tokens = encoding.input_ids
  segments_ids = [1]*len(indexed_tokens)

  # convert inputs to tensors
  tokens_tensor = torch.tensor([indexed_tokens])
  segments_tensor = torch.tensor([segments_ids])

  return tokenized_text, tokens_tensor, segments_tensor, encoding

def get_bert_embeddings(tokens_tensor, segments_tensor, model):
    """
    Obtains BERT embeddings for tokens, in context of the given sentence.
    """
    # gradient calculation id disabled
    with torch.no_grad():
      # obtain hidden states
      outputs = model(tokens_tensor, segments_tensor)
      hidden_states = outputs[2]

    # concatenate the tensors for all layers
    # use "stack" to create new dimension in tensor
    token_embeddings = torch.stack(hidden_states, dim=0)

    # remove dimension 1, the "batches"
    token_embeddings = torch.squeeze(token_embeddings, dim=1)

    # swap dimensions 0 and 1 so we can loop over tokens
    token_embeddings = token_embeddings.permute(1,0,2)

    # intialized list to store embeddings
    token_vecs_sum = []

    # "token_embeddings" is a [Y x 12 x 768] tensor
    # where Y is the number of tokens in the sentence

    # loop over tokens in sentence
    for token in token_embeddings:

        # "token" is a [12 x 768] tensor

        # sum the vectors from the last four layers
        sum_vec = torch.sum(token[-4:], dim=0)
        token_vecs_sum.append(sum_vec)

    return token_vecs_sum

def aggregate_subwords(encoding, list_token_embeddings, text):
    recomposed_tokens = []  # List to store the recomposed tokens
    recomposed_emb = []  # List to store the recomposed embeddings
    hashtag = False  # Flag to indicate if a hashtag is encountered
    hashtag_emb = False  # Flag to indicate if a hashtag is part of the mean calculation

    for i in sorted(list(set(encoding.word_ids())), key=lambda x: (x is None, x)):
      #index_of_token = encoding.word_ids()[i]
      if i != None:
        #if the embedding is related to a single token
        if encoding.word_ids().count(i) ==1:
          recomposed_emb.append(list_token_embeddings[encoding.word_ids().index(i)])
        #if the embed is given by the mean of multiple tokens
        elif encoding.word_ids().count(i) >1:
          #retrive the first one
          emb = list_token_embeddings[encoding.word_ids().index(i)]
          # count the number of tokens to mean
          num = encoding.word_ids().count(i)
          # if I have to iclude an hashtag inside a mean
          if hashtag_emb:
            #remove last element (the hashag) and include it in the mean
            emb = emb + recomposed_emb.pop()
            num = encoding.word_ids().count(i)+1
            hashtag_emb = False
          for a in range(1, encoding.word_ids().count(i)):
            emb = emb + list_token_embeddings[encoding.word_ids().index(i)+a]
          emb = emb/num
          recomposed_emb.append(emb)

        start, end = encoding.word_to_chars(i)
        #print(text[start:end])
        if hashtag:
          recomposed_tokens.append('#'+text[start:end])
          hashtag=False
        elif text[start:end] == '#':
          hashtag=True
          hashtag_emb = True
          hash_emb = list_token_embeddings[encoding.word_ids().index(i)]

        else:
          #print(text[start:end])
          recomposed_tokens.append(text[start:end])

    if "'" in recomposed_tokens:
      pos = recomposed_tokens.index("'")
      new = ''
      new_emb = torch.tensor([0]*768)
      for el in range(pos-1,pos+2):
        new_emb = new_emb + recomposed_emb[el]
        new=new+recomposed_tokens[el]

      recomposed_tokens  = recomposed_tokens[:pos-1] + [new] + recomposed_tokens[pos+2:]
      recomposed_emb  = recomposed_emb[:pos-1] + [new_emb/3] + recomposed_emb[pos+2:]

    if "*" in recomposed_tokens:
      pos = recomposed_tokens.index("*")
      new = ''
      new_emb = torch.tensor([0]*768)
      consec = 1
      next = recomposed_tokens[pos+consec]
      while (next =='*'):
        consec = consec+1
        next = recomposed_tokens[pos+consec]
      for el in range(pos-1,pos+consec+1):
        new_emb = new_emb + recomposed_emb[el]
        new=new+recomposed_tokens[el]

      recomposed_tokens  = recomposed_tokens[:pos-1] + [new] + recomposed_tokens[pos+consec+1:]
      recomposed_emb  = recomposed_emb[:pos-1] + [new_emb/3] + recomposed_emb[pos+consec+1:]


    return recomposed_tokens, recomposed_emb

def text_to_emb(text, tokenizer, model):
  tokenized_text, tokens_tensor, segments_tensors, encoding = bert_text_preparation(text, tokenizer)
  list_token_embeddings = get_bert_embeddings(tokens_tensor, segments_tensors, model)
  recomposed_tokens, recomposed_emb = aggregate_subwords(encoding, list_token_embeddings, text)
  return recomposed_tokens, recomposed_emb

from scipy.spatial.distance import cosine

def find_similar_words(new_sent, new_token_indexes, tokenizer, context_tokens, context_embeddings, model, tokens_df):
    """
    Find similar words to the given new words in a context.

    Args:
        new_sent (str): The input sentence containing the new words.
        new_token_indexes (list): List of indexes of the new words in the sentence.

    Returns:
        tuple: A tuple containing:
            - similar_words (list): List of similar words for each new word.
            - distances_df (DataFrame): DataFrame containing the token, new_token, and distance.
            - new_words (list): List of the new words extracted from the sentence.
    """

    # embeddings for the NEW word 'record'
    list_of_distances = []
    list_of_new_embs = []
    new_words = []

    #tokenized_text, tokens_tensor, segments_tensors = bert_text_preparation(new_sent, tokenizer)
    tokenized_text, new_emb = text_to_emb(new_sent, tokenizer, model)
    #print(tokenized_text)
    if tokenized_text != new_sent.split():
      print(tokenized_text)
      print(new_sent.split())
      return [], 0, new_words

    for new_token_index in new_token_indexes:
        #new_emb = get_bert_embeddings(tokens_tensor, segments_tensors, model)[new_token_index]
        list_of_new_embs.append(new_emb[new_token_index])
        new_words.append(tokenized_text[new_token_index])

    for sentence_1, embed1 in zip(context_tokens, context_embeddings):
        for i in range(0, len(new_token_indexes)):
            cos_dist = 1 - cosine(embed1, list_of_new_embs[i])
            list_of_distances.append([sentence_1, tokenized_text[new_token_indexes[i]], cos_dist])

    distances_df = pd.DataFrame(list_of_distances, columns=['token', 'new_token', 'distance'])
    # tengo solo quelle con i token validi = che compaiono almeno 10 volte
    distances_df = distances_df.merge(tokens_df, on='token')

    similar_words = []

    for i in range(0, len(new_token_indexes)):
      if distances_df.loc[distances_df.new_token == new_sent.split(' ')[new_token_indexes[i]], 'distance'].idxmax():
        similar_words.append([distances_df.loc[distances_df.loc[distances_df.new_token == new_sent.split(' ')[new_token_indexes[i]], 'distance'].idxmax(), 'token']])
      else:
        similar_words.append([])


    return similar_words, distances_df, new_words
