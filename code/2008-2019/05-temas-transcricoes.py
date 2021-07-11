import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

N_COMPONENTS = 22
LEARNING_DECAY = 0.6

PATH_METADADOS_TRANSCRICOES = './data/2008-2019/01-metadados.csv'
PATH_STEMS_TRANSCRICOES = './data/2008-2019/03-stems.csv'
PATH_TEMAS_TRANSCRICOES = './data/2008-2019/05-temas-transcricoes.csv'
PATH_TERMOS_PRINCIPAIS = './data/2008-2019/05-termos-principais.csv'

transcricoes = pd.read_csv(PATH_METADADOS_TRANSCRICOES)
transcricoes['stems'] = pd.read_csv(PATH_STEMS_TRANSCRICOES)['stems']
transcricoes = transcricoes.filter(items=['id_evento', 'comissao', 'stems'])

transcricoes['stems'] = transcricoes['stems'].apply(
    lambda stems: stems[1:-1].replace("'", '').split(', ')
)

percentis = {'5': int(transcricoes.shape[0] * 0.05),
             '80': int(transcricoes.shape[0] * 0.80)}

vetorizador = CountVectorizer(min_df=percentis['5'], max_df=percentis['80'],
                              max_features=None, ngram_range=(1, 2),
                              lowercase=False, tokenizer=(lambda x: x))

termo_documento = vetorizador.fit_transform(transcricoes['stems'])
termos = np.array(vetorizador.get_feature_names())

lda = LatentDirichletAllocation(n_components=N_COMPONENTS,
                                learning_decay=LEARNING_DECAY,
                                learning_method='online',
                                random_state=0)

temas = [f'tema_{i}' for i in range(N_COMPONENTS)]
tema_documento = np.round(lda.fit_transform(termo_documento), 4)
temas_transcricoes = pd.DataFrame(data=tema_documento, columns=temas)
temas_principais = np.argmax(temas_transcricoes.values, axis=1)

temas_transcricoes.insert(0, 'tema_principal', temas_principais)
temas_transcricoes.insert(0, 'id_evento', transcricoes['id_evento'].values)
temas_transcricoes.to_csv(PATH_TEMAS_TRANSCRICOES, index=False)

termos_por_tema = []
for probabilidade_tema in lda.components_:
    indices_termos = (-probabilidade_tema).argsort()[:10]
    termos_por_tema.append(termos.take(indices_termos))

termos_principais = pd.DataFrame(termos_por_tema)
termos_principais.columns = [f'termo_{i}' for i in range(10)]
termos_principais.insert(0, 'tema', [i for i in range(N_COMPONENTS)])
termos_principais.to_csv(PATH_TERMOS_PRINCIPAIS, index=False)
