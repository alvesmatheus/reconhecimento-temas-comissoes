import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV

PATH_METADADOS_TRANSCRICOES = './data/2009-2018/01-metadados.csv'
PATH_STEMS_TRANSCRICOES = './data/2009-2018/03-stems.csv'
PATH_AVALIACAO_MODELOS = './data/2009-2018/04-avaliacao-modelos.csv'

COMISSOES_ALVO = [
    'Comissão de Constituição e Justiça e de Cidadania',
    'Comissão de Agricultura, Pecuária, Abastecimento e Desenvolvimento Rural',
    'Comissão de Direitos Humanos e Minorias'
]

transcricoes = pd.read_csv(PATH_METADADOS_TRANSCRICOES)
transcricoes['stems'] = pd.read_csv(PATH_STEMS_TRANSCRICOES)['stems']
transcricoes = transcricoes.filter(items=['id_evento', 'comissao', 'stems'])
transcricoes = transcricoes[transcricoes['comissao'].isin(COMISSOES_ALVO)]

transcricoes['stems'] = transcricoes['stems'].apply(
    lambda stems: stems[1:-1].replace("'", '').split(', ')
)

percentis = {'5': int(transcricoes.shape[0] * 0.05),
             '80': int(transcricoes.shape[0] * 0.80)}

vetorizador = CountVectorizer(min_df=percentis['5'], max_df=percentis['80'],
                              max_features=None, ngram_range=(1, 2),
                              lowercase=False, tokenizer=(lambda x: x))

termo_documento = vetorizador.fit_transform(transcricoes['stems'])

lda = LatentDirichletAllocation(learning_method='online', random_state=0)
parametros = {
    'n_components': [x for x in range(2, 16)],
    'learning_decay': [0.60, 0.75, 0.90]
}

otimizador_parametros = GridSearchCV(estimator=lda, param_grid=parametros)

otimizador_parametros.fit(termo_documento)
resultados = otimizador_parametros.cv_results_

avaliacao_modelos = pd.DataFrame({
    'learning_decay': resultados['param_learning_decay'].data,
    'n_components': resultados['param_n_components'].data,
    'log_likelihood': resultados['mean_test_score']
})

avaliacao_modelos.to_csv(PATH_AVALIACAO_MODELOS, index=False)
