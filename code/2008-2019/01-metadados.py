import pandas as pd

PATH_TRANSCRICOES = './data/2008-2019/00-transcricoes.csv'
PATH_METADADOS = './data/2008-2019/01-metadados.csv'

transcricoes = pd.read_csv(PATH_TRANSCRICOES)

metadados = transcricoes.filter(items=['id_evento', 'comissao', 'ano', 'data',
                                       'categoria_evento'])

metadados.to_csv(PATH_METADADOS, index=False)
