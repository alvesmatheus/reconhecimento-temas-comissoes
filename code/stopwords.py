import pandas as pd
from nltk.corpus import stopwords
from spacy.lang.pt.stop_words import STOP_WORDS

PATH_STOPWORDS = './data/stopwords.csv'

stopwords_dominio = [
    'acho', 'agradeço', 'algum', 'alguma', 'art', 'assunto', 'atenção', 'caso',
    'claro', 'coisas', 'comissão', 'câmara', 'deputada', 'deputado',
    'deputados', 'desses', 'discutir', 'disse', 'dr', 'existe', 'fala',
    'falar', 'fazendo', 'feita', 'feito', 'gostaria', 'haver', 'havia',
    'mesma', 'ministro', 'ministério', 'minutos', 'muitas', 'muitos', 'n',
    'nenhum', 'ordem', 'outro', 'palavra', 'parlamentares', 'peço', 'podemos',
    'possa', 'precisa', 'precisamos', 'preciso', 'presente', 'presença',
    'presidente', 'principalmente', 'queremos', 'queria', 'realmente',
    'relator', 'seguinte', 'senhor', 'senhores', 'sido', 'sr', 'sra', 'srs',
    'vamos', 'vexa', 'vista', 'vou'
]

stopwords_nltk = stopwords.words('portuguese')
stopwords_spacy = STOP_WORDS

conjunto_stopwords = set()
conjunto_stopwords.update(stopwords_nltk)
conjunto_stopwords.update(stopwords_spacy)
conjunto_stopwords.update(stopwords_dominio)
conjunto_stopwords = sorted(conjunto_stopwords)

stopwords = pd.DataFrame({'palavra': conjunto_stopwords})
stopwords.to_csv(PATH_STOPWORDS, index=False)
