import re

import pandas as pd
from nltk.tokenize import RegexpTokenizer

PATH_STOPWORDS = './data/stopwords.csv'
PATH_TRANSCRICOES = './data/2009-2018/00-transcricoes.csv'
PATH_TOKENS_TRANSCRICOES = './data/2009-2018/02-tokens.csv'

REGEX_PONTUACAO = r'''([.,:;[\](){}\\/ºª|—!?@$%=+\-*'"])'''
REGEX_TOKENS = r'([a-zà-úA-ZÀ-Ú]+)'

transcricoes = pd.read_csv(PATH_TRANSCRICOES)
tokens = transcricoes.filter(items=['id_evento'])

tokenizador = RegexpTokenizer(REGEX_TOKENS)
stopwords = pd.read_csv(PATH_STOPWORDS)['palavra'].values.tolist()

tokens['tokens'] = transcricoes['transcricao'].apply(
    lambda texto: re.sub(REGEX_PONTUACAO, '', texto)
).apply(
    lambda texto: texto.lower()
).apply(
    lambda texto: tokenizador.tokenize(texto)
).apply(
    lambda tokens: [token for token in tokens if token not in stopwords]
)

tokens.to_csv(PATH_TOKENS_TRANSCRICOES, index=False)
