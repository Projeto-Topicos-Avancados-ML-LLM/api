import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import requests

# ======================================
#  FUNÇÃO: CARREGAMENTO DOS DADOS (CACHE)
# ======================================
@st.cache_data(show_spinner=False)
def load_data():
    filmes = pd.read_csv('movies_metadata.csv', low_memory=False)
    avaliacoes = pd.read_csv('ratings.csv')

    filmes = filmes[['id', 'original_title', 'original_language', 'vote_count']]
    filmes.rename(columns={
        'id': 'ID_FILME',
        'original_title': 'TITULO',
        'original_language': 'LINGUAGEM',
        'vote_count': 'QT_AVALIACOES',
        'imdb_id': 'IMDB_ID'
    }, inplace=True)
    filmes.dropna(inplace=True)
    filmes = filmes.loc[(filmes['QT_AVALIACOES'] > 999) & (filmes['LINGUAGEM'] == 'en')]
    filmes['ID_FILME'] = filmes['ID_FILME'].astype(int)

    avaliacoes.rename(columns={
        'userId': 'ID_USUARIO',
        'movieId': 'ID_FILME',
        'rating': 'AVALIACAO'
    }, inplace=True)
    qt_avaliacoes = avaliacoes['ID_USUARIO'].value_counts() > 999
    y = qt_avaliacoes[qt_avaliacoes].index
    avaliacoes = avaliacoes[avaliacoes['ID_USUARIO'].isin(y)]

    avaliacoes_e_filmes = avaliacoes.merge(filmes, on='ID_FILME')
    avaliacoes_e_filmes.drop_duplicates(['ID_USUARIO', 'ID_FILME'], inplace=True)

    filmes_pivot = avaliacoes_e_filmes.pivot_table(
        columns='ID_USUARIO', index='TITULO', values='AVALIACAO'
    ).fillna(0)
    return filmes, avaliacoes_e_filmes, filmes_pivot


# ======================================
#  FUNÇÃO: TREINAMENTO DO MODELO (CACHE)
# ======================================
@st.cache_resource(show_spinner=False)
def build_model(filmes_pivot):
    filmes_sparse = csr_matrix(filmes_pivot.values)
    modelo = NearestNeighbors(algorithm='brute')
    modelo.fit(filmes_sparse)
    return modelo


# ======================================
#  FUNÇÃO: BUSCA NA API IMDB
# ======================================
def get_imdb_data(titulo):
    try:
        titulo_enc = requests.utils.requote_uri(titulo)
        # url = f"https://imdbapi.dev/api?title={titulo_enc}"
        # titulo_enc = requests.utils.requote_uri()
        url = f"https://api.imdbapi.dev/titles/{titulo_enc}"

        resp = requests.get(url, timeout=10)
        data = resp.json()

        # a API imdbapi.dev retorna uma lista — vamos pegar o primeiro resultado
        if isinstance(data, list) and len(data) > 0:
            movie = data[0]
        else:
            movie = data

        if movie.get("title"):
            return {
                "title": movie.get("title"),
                "year": movie.get("year"),
                "poster": movie.get("poster"),
                "plot": movie.get("plot")
            }
        return None
    except Exception as e:
        st.error(f"Erro ao buscar dados do IMDb: {e}")
        return None


# ======================================
#  INICIALIZAÇÃO
# ======================================
filmes, avaliacoes_e_filmes, filmes_pivot = load_data()
modelo = build_model(filmes_pivot)

# ======================================
#  INTERFACE STREAMLIT
# ======================================
st.set_page_config(page_title="Recomendador de Filmes", page_icon="🎬", layout="wide")
st.title("🎬 Sistema de Recomendação de Filmes")
st.caption("Baseado em similaridade de avaliações (k-NN) + dados do IMDb")

titles = list(filmes_pivot.index)
chosen_from_list = st.selectbox("Escolha um filme (opcional)", options=[""] + titles)
nome_filme = st.text_input("Ou digite o nome do filme")

nome_filme_final = chosen_from_list if chosen_from_list else nome_filme.strip()

if st.button("Buscar recomendações"):
    if not nome_filme_final:
        st.warning("Por favor digite ou escolha um filme antes de buscar.")
    elif nome_filme_final not in filmes_pivot.index:
        st.error(f"O filme '{nome_filme_final}' não foi encontrado na base de dados.")
    else:
        try:
            query_vec = filmes_pivot.filter(items=[nome_filme_final], axis=0).values.reshape(1, -1)
            distances, sugestions = modelo.kneighbors(query_vec, n_neighbors=6)
            recommended = filmes_pivot.index[sugestions[0][1:]]

            st.subheader("Filmes recomendados:")
            for i, movie in enumerate(recommended, start=1):
                print(movie)
                info = get_imdb_data(movie)

                if info:
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        if info.get("poster"):
                            st.image(info["poster"], width=150)
                        else:
                            st.write("📷 Sem imagem")
                    with col2:
                        st.markdown(f"### {i}. {info['title']} ({info.get('year', '')})")
                        st.caption(info.get("plot", ""))
                else:
                    st.markdown(f"### {i}. {movie}")

        except Exception as e:
            st.error(f"Erro ao buscar recomendações: {e}")
