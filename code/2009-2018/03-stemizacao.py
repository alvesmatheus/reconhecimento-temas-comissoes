import pandas as pd
from nltk.stem.rslp import RSLPStemmer

PATH_TOKENS_TRANSCRICOES = './data/2009-2018/02-tokens.csv'
PATH_STEMS_TRANSCRICOES = './data/2009-2018/03-stems.csv'

tokens = pd.read_csv(PATH_TOKENS_TRANSCRICOES)
stems = tokens.filter(items=['id_evento'])

tokens['tokens'] = tokens['tokens'].apply(
    lambda tokens: tokens[1:-1].replace("'", '').split(', ')
)

stemizador = RSLPStemmer()
stems['stems'] = tokens['tokens'].apply(
    lambda tokens: [stemizador.stem(token) for token in tokens]
)

stems.to_csv(PATH_STEMS_TRANSCRICOES, index=False)
