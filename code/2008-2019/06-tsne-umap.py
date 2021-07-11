import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from umap.umap_ import UMAP

PATH_STEMS_TRANSCRICOES = './data/2008-2019/03-stems.csv'
PATH_TSNE_TRANSCRICOES = './data/2008-2019/06-tsne.csv'
PATH_UMAP_TRANSCRICOES = './data/2008-2019/06-umap.csv'

stems = pd.read_csv(PATH_STEMS_TRANSCRICOES)
percentis = {'5': int(stems.shape[0] * 0.05),
             '80': int(stems.shape[0] * 0.80)}

tsne_transcricoes = stems.filter(items=['id_evento'])
umap_transcricoes = stems.filter(items=['id_evento'])

stems['stems'] = stems['stems'].apply(
    lambda stems: stems[1:-1].replace("'", '').split(', ')
)

vetorizador = TfidfVectorizer(min_df=percentis['5'],
                              max_df=percentis['80'],
                              max_features=None,
                              ngram_range=(1, 2),
                              lowercase=False,
                              tokenizer=(lambda x: x))

termo_documento = vetorizador.fit_transform(stems['stems'])

tsne = TSNE(n_components=2, random_state=0, verbose=0)
tsne_embeddings = tsne.fit_transform(termo_documento)
tsne_transcricoes = tsne_transcricoes.assign(
    tsne_x=tsne_embeddings[:, 0],
    tsne_y=tsne_embeddings[:, 1]
)

umap = UMAP(n_components=2, random_state=0)
umap_embeddings = umap.fit_transform(termo_documento)
umap_transcricoes = umap_transcricoes.assign(
    umap_x=umap_embeddings[:, 0],
    umap_y=umap_embeddings[:, 1]
)

tsne_transcricoes.to_csv(PATH_TSNE_TRANSCRICOES, index=False)
umap_transcricoes.to_csv(PATH_UMAP_TRANSCRICOES, index=False)
