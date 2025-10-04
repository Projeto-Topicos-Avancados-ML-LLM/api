import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import requests
import difflib
import logging

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
#  HELPERS PARA EXTRACAO DE PLOT E POSTER
# ======================================
def _extract_plot_from_obj(obj):
    """
    Tenta extrair texto de plot/summary/plots/plotSummary de um objeto retornado pela API.
    Retorna string ou None.
    """
    if not isinstance(obj, dict):
        return None

    # 1) plot direto (string)
    p = obj.get('plot')
    if isinstance(p, str) and p.strip():
        return p.strip()

    # 2) summary / description / overview
    for key in ('summary', 'description', 'overview'):
        v = obj.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()

    # 3) plots: lista com dicts ou strings
    plots = obj.get('plots') or obj.get('plotText') or obj.get('plotTexts')
    if isinstance(plots, list) and len(plots) > 0:
        first = plots[0]
        if isinstance(first, str) and first.strip():
            return first.strip()
        if isinstance(first, dict):
            for candidate in ('text', 'plainText', 'plotText', 'summary', 'plot'):
                tv = first.get(candidate)
                if isinstance(tv, str) and tv.strip():
                    return tv.strip()

    # 4) plotSummary: possivel dict com 'text' ou 'summaryText'
    ps = obj.get('plotSummary') or obj.get('plot_summary')
    if isinstance(ps, dict):
        for candidate in ('text', 'summaryText', 'summary'):
            tv = ps.get(candidate)
            if isinstance(tv, str) and tv.strip():
                return tv.strip()

    return None


def _extract_poster_from_obj(obj):
    """
    Extrai URL do poster/primaryImage/url etc.
    """
    if not isinstance(obj, dict):
        return None

    # primaryImage.url
    primary = obj.get('primaryImage')
    if isinstance(primary, dict):
        url = primary.get('url')
        if isinstance(url, str) and url.strip():
            return url.strip()

    # outras chaves que podem conter imagem
    for key in ("poster", "image", "poster_url", "posterLink", "image_url"):
        val = obj.get(key)
        if isinstance(val, dict):
            url = val.get('url')
            if isinstance(url, str) and url.strip():
                return url.strip()
        elif isinstance(val, str) and val.strip():
            return val.strip()

    return None


# ======================================
#  FUNÇÃO: BUSCA NA API IMDB (EXTRAI POSTER, PLOT E RATING)
#  - tenta primeiro a resposta de search/titles
#  - se não encontrar 'plot' faz fallback para titles/{id} para obter descrição detalhada
# ======================================
def get_imdb_data(titulo):
    try:
        titulo_enc = requests.utils.requote_uri(titulo)
        url = f"https://api.imdbapi.dev/search/titles?query={titulo_enc}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        # Normaliza para lista de títulos
        if isinstance(data, dict) and isinstance(data.get("titles"), list):
            titles = data["titles"]
        elif isinstance(data, list):
            titles = data
        elif isinstance(data, dict):
            titles = [data]
        else:
            titles = []

        # Escolhe o primeiro item com primaryImage (se existir), senão o primeiro item
        movie = None
        for item in titles:
            if not isinstance(item, dict):
                continue
            primary = item.get("primaryImage")
            if isinstance(primary, dict) and primary.get("url"):
                movie = item
                break
        if movie is None and titles:
            movie = titles[0]
        if movie is None:
            return None

        # Extrai poster
        poster = _extract_poster_from_obj(movie)

        # Extrai plot diretamente do objeto retornado pelo search (se houver)
        plot = _extract_plot_from_obj(movie)

        # Se não encontrou plot, tenta consultar detalhes via titles/{id} (fallback)
        if not plot:
            movie_id = movie.get('id')
            if movie_id:
                try:
                    detail_url = f"https://api.imdbapi.dev/titles/{movie_id}"
                    resp2 = requests.get(detail_url, timeout=8)
                    resp2.raise_for_status()
                    detail_obj = resp2.json()
                    # detail_obj pode ter 'plot', 'plots', 'plotSummary', etc.
                    plot = _extract_plot_from_obj(detail_obj)
                    # caso detail traga poster melhor
                    if not poster:
                        poster = _extract_poster_from_obj(detail_obj)
                except Exception:
                    # falha no detalhe não é crítica; mantemos plot como None
                    pass

        # Title / year
        title = movie.get("primaryTitle") or movie.get("title") or movie.get("originalTitle") or titulo
        year = movie.get("startYear") or movie.get("year")

        # Tenta extrair rating do objeto
        imdb_rating = None
        imdb_votes = None
        rating_obj = movie.get("rating") or movie.get("ratings") or movie.get("aggregateRating")
        if isinstance(rating_obj, dict):
            imdb_rating = rating_obj.get("aggregateRating") or rating_obj.get("average") or rating_obj.get("value")
            imdb_votes = rating_obj.get("voteCount") or rating_obj.get("votes") or rating_obj.get("count")
        # Conversões seguras
        try:
            if imdb_rating is not None:
                imdb_rating = float(imdb_rating)
        except Exception:
            imdb_rating = None
        try:
            if imdb_votes is not None:
                imdb_votes = int(imdb_votes)
        except Exception:
            imdb_votes = None

        return {
            "title": title,
            "year": year,
            "poster": poster,
            "plot": plot or "Sem descrição disponível.",
            "imdb_rating": imdb_rating,
            "imdb_votes": imdb_votes
        }
    except Exception as e:
        st.error(f"Erro ao buscar dados do IMDb: {e}")
        return None


# ======================================
#  FUNÇÃO: BUSCA CASE-INSENSITIVE / SUBSTRING / FUZZY
# ======================================
def find_best_title_match(query, titles, filmes_df):
    """
    Retorna o título da lista 'titles' que melhor corresponde a 'query',
    ignorando case. Estratégia:
     - igualdade exata (case-insensitive)
     - títulos que contenham a query (case-insensitive), escolhendo o mais popular por QT_AVALIACOES
     - correspondência aproximada via difflib como fallback
    """
    q = (query or "").strip()
    if not q:
        return None
    q_low = q.lower()

    # 1) igualdade exata (case-insensitive)
    for t in titles:
        if t.lower() == q_low:
            return t

    # 2) contains (case-insensitive)
    contains = [t for t in titles if q_low in t.lower()]
    if contains:
        # se tivermos o DataFrame filmes com coluna 'TITULO' e 'QT_AVALIACOES', escolhe o mais popular
        if 'TITULO' in filmes_df.columns and 'QT_AVALIACOES' in filmes_df.columns:
            subset = filmes_df[filmes_df['TITULO'].isin(contains)]
            if not subset.empty:
                best = subset.sort_values('QT_AVALIACOES', ascending=False).iloc[0]['TITULO']
                return best
        # senão, retorna o primeiro encontrado
        return contains[0]

    # 3) correspondência aproximada (fuzzy) como fallback
    close = difflib.get_close_matches(q, titles, n=1, cutoff=0.6)
    if close:
        return close[0]

    return None


# ======================================
#  INICIALIZAÇÃO
# ======================================
filmes, avaliacoes_e_filmes, filmes_pivot = load_data()
modelo = build_model(filmes_pivot)

# resumo das avaliações (do dataset ratings.csv, após merges) para mostrar notas/contagens internas
ratings_summary = avaliacoes_e_filmes.groupby('TITULO')['AVALIACAO'].agg(['mean', 'count'])
ratings_summary.rename(columns={'mean': 'avg_rating', 'count': 'num_ratings'}, inplace=True)
ratings_summary['avg_rating'] = ratings_summary['avg_rating'].round(2)

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
    else:
        # Encontra a melhor correspondência ignorando case
        best_match = find_best_title_match(nome_filme_final, titles, filmes)
        if not best_match:
            st.error(f"O filme '{nome_filme_final}' não foi encontrado na base de dados.")
        else:
            if best_match != nome_filme_final:
                st.info(f"Usando correspondência encontrada: '{best_match}'")

            try:
                # Mostra informações do filme pesquisado
                info_main = get_imdb_data(best_match)
                # busca stats internas
                stats_main = None
                if best_match in ratings_summary.index:
                    stats_main = ratings_summary.loc[best_match]

                st.subheader("Filme pesquisado")
                col1, col2 = st.columns([1, 3])
                with col1:
                    if info_main and info_main.get("poster"):
                        st.image(info_main["poster"], width=180)
                    else:
                        st.write("📷 Sem imagem")
                with col2:
                    title_str = info_main['title'] if info_main and info_main.get('title') else best_match
                    year_str = f" ({info_main.get('year')})" if info_main and info_main.get('year') else ""
                    st.markdown(f"### {title_str}{year_str}")
                    # IMDb rating
                    if info_main and info_main.get("imdb_rating") is not None:
                        imdb_votes = info_main.get("imdb_votes")
                        votes_text = f" ({imdb_votes} votos)" if imdb_votes else ""
                        st.write(f"**Nota IMDb:** {info_main['imdb_rating']}{votes_text}")
                    # Interno (dataset) rating
                    if stats_main is not None:
                        st.write(f"**Nota média (usuários da base):** {stats_main['avg_rating']} / 5 ({int(stats_main['num_ratings'])} avaliações)")
                    # Plot / descrição
                    if info_main and info_main.get("plot"):
                        st.caption(info_main.get("plot"))

                # Recomendados
                query_vec = filmes_pivot.filter(items=[best_match], axis=0).values.reshape(1, -1)
                distances, sugestions = modelo.kneighbors(query_vec, n_neighbors=6)
                recommended = filmes_pivot.index[sugestions[0][1:]]

                st.subheader("Filmes recomendados:")
                for i, movie in enumerate(recommended, start=1):
                    info = get_imdb_data(movie)
                    stats = None
                    if movie in ratings_summary.index:
                        stats = ratings_summary.loc[movie]

                    if info:
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            if info.get("poster"):
                                st.image(info["poster"], width=150)
                            else:
                                st.write("📷 Sem imagem")
                        with col2:
                            title_str = info.get('title') or movie
                            year_str = f" ({info.get('year')})" if info.get('year') else ""
                            st.markdown(f"### {i}. {title_str}{year_str}")
                            # IMDb rating
                            if info.get("imdb_rating") is not None:
                                imdb_votes = info.get("imdb_votes")
                                votes_text = f" ({imdb_votes} votos)" if imdb_votes else ""
                                st.write(f"**Nota IMDb:** {info['imdb_rating']}{votes_text}")
                            # Interno (dataset) rating
                            if stats is not None:
                                st.write(f"**Nota média (usuários da base):** {stats['avg_rating']} / 5 ({int(stats['num_ratings'])} avaliações)")
                            # Plot
                            if info.get("plot"):
                                st.caption(info.get("plot"))
                    else:
                        st.markdown(f"### {i}. {movie}")

            except Exception as e:
                st.error(f"Erro ao buscar recomendações: {e}")
