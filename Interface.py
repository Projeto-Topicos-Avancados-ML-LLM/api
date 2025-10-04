import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# --- Carregamento e Preparação dos Dados (cache) ---
@st.cache_data(show_spinner=False)
def load_data():
    filmes = pd.read_csv('movies_metadata.csv', low_memory=False)
    avaliacoes = pd.read_csv('ratings.csv')

    filmes = filmes[['id', 'original_title', 'original_language', 'vote_count']]
    filmes.rename(columns={'id': 'ID_FILME', 'original_title': 'TITULO', 'original_language': 'LINGUAGEM', 'vote_count': 'QT_AVALIACOES'}, inplace=True)
    filmes.dropna(inplace=True)
    filmes = filmes.loc[(filmes['QT_AVALIACOES'] > 999) & (filmes['LINGUAGEM'] == 'en')]
    filmes['ID_FILME'] = filmes['ID_FILME'].astype(int)

    avaliacoes.rename(columns={'userId': 'ID_USUARIO', 'movieId': 'ID_FILME', 'rating': 'AVALIACAO'}, inplace=True)
    qt_avaliacoes = avaliacoes['ID_USUARIO'].value_counts() > 999
    y = qt_avaliacoes[qt_avaliacoes].index
    avaliacoes = avaliacoes[avaliacoes['ID_USUARIO'].isin(y)]

    avaliacoes_e_filmes = avaliacoes.merge(filmes, on='ID_FILME')
    avaliacoes_e_filmes.drop_duplicates(['ID_USUARIO', 'ID_FILME'], inplace=True)

    filmes_pivot = avaliacoes_e_filmes.pivot_table(columns='ID_USUARIO', index='TITULO', values='AVALIACAO').fillna(0)
    return filmes, avaliacoes_e_filmes, filmes_pivot


@st.cache_resource(show_spinner=False)
def build_model(filmes_pivot):
    filmes_sparse = csr_matrix(filmes_pivot.values)
    modelo = NearestNeighbors(algorithm='brute')
    modelo.fit(filmes_sparse)
    return modelo


# Load data and model
filmes, avaliacoes_e_filmes, filmes_pivot = load_data()
modelo = build_model(filmes_pivot)


# --- Criação da Interface com Streamlit ---
st.title("Sistema de Recomendação de Filmes")
st.write("Digite o nome de um filme (ou escolha na lista) para encontrar recomendações.")

# Helpful UI: a lista de títulos para facilitar a busca
titles = list(filmes_pivot.index)
chosen_from_list = st.selectbox("Escolha um filme (opcional)", options=[""] + titles)
nome_filme = st.text_input("Ou digite o nome do filme")

# decide which to use
nome_filme_final = chosen_from_list if chosen_from_list else nome_filme.strip()


if st.button("Buscar"):
    if not nome_filme_final:
        st.warning("Por favor digite ou escolha um filme antes de buscar.")
    else:
        if nome_filme_final not in filmes_pivot.index:
            st.error(f"O filme '{nome_filme_final}' não foi encontrado na base de dados.")
        else:
            try:
                # compute neighbors
                query_vec = filmes_pivot.filter(items=[nome_filme_final], axis=0).values.reshape(1, -1)
                distances, sugestions = modelo.kneighbors(query_vec, n_neighbors=6)

                recommended = filmes_pivot.index[sugestions[0][1:]]

                st.subheader("Filmes recomendados:")
                for i, movie in enumerate(recommended, start=1):
                    info = filmes[filmes['TITULO'] == movie]
                    st.markdown(f"**{i}. {movie}**")
                    if not info.empty:
                        # show selected useful columns
                        row = info.iloc[0]
                        st.write({
                            'Título': row['TITULO'],
                            'Idioma': row.get('LINGUAGEM', ''),
                            'ID_FILME': int(row['ID_FILME']),
                            'QT_AVALIACOES': int(row['QT_AVALIACOES'])
                        })
                    else:
                        st.write(movie)
                # optional: show distances
                st.write('\nDistâncias (mais próximo -> mais distante):')
                st.write(distances[0][1:])

            except Exception as e:
                st.error(f"Erro ao buscar recomendações: {e}")

# --- Fim do Código ---


# bambi: [
#     {
#         id: 68721,
#         imdb_id: "tt3312830",
#         name: "harry potter",
#         genres: [{id: 12,name: "Adventure" }, "Fantasy", "Family"]
#     },
#     {
#         id: 19995,
#         "fast and furious",
#                 genres: ["Action", "Crime", "Thriller"]
#     },
#     ...8
# ]
