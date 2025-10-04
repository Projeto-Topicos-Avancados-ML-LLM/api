import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import requests
import difflib
import logging
import os
import re # Módulo para expressões regulares, usado para limpeza de títulos

# --- Configurações de Caminho e Desempenho ---
# Assumimos que os arquivos .csv estão no diretório pai ('../')
DATA_PATH = os.path.join(os.path.dirname(__file__), '')
METADATA_FILE = os.path.join(DATA_PATH, 'movies_metadata.csv')
RATINGS_FILE = os.path.join(DATA_PATH, 'ratings.csv')

# Variável de configuração: Define o número máximo de linhas a serem lidas do ratings.csv
MAX_RATINGS_ROWS = 5000000 

# ======================================
#  FUNÇÃO: CARREGAMENTO DOS DADOS (CACHE)
# ======================================
@st.cache_data(show_spinner="Carregando e processando dados...")
def load_data():
    try:
        filmes = pd.read_csv(METADATA_FILE, low_memory=False)
        # Linha removida: st.info(f"Carregando no máximo {MAX_RATINGS_ROWS:,} linhas do arquivo de avaliações para evitar timeouts.".replace(",", "."))
        avaliacoes = pd.read_csv(RATINGS_FILE, nrows=MAX_RATINGS_ROWS)
    except FileNotFoundError as e:
        st.error(f"Erro: Arquivo não encontrado. Verifique se {e.filename} está no local correto (deve estar em {DATA_PATH}).")
        st.stop()
    
    # Processamento de Filmes
    filmes = filmes[['id', 'original_title', 'original_language', 'vote_count']]
    filmes.rename(columns={
        'id': 'ID_FILME',
        'original_title': 'TITULO',
        'original_language': 'LINGUAGEM',
        'vote_count': 'QT_AVALIACOES',
    }, inplace=True)
    filmes.dropna(inplace=True)
    filmes = filmes.loc[(filmes['QT_AVALIACOES'] > 999) & (filmes['LINGUAGEM'] == 'en')]
    
    # Conversão segura de ID_FILME
    filmes['ID_FILME'] = filmes['ID_FILME'].astype(str)
    filmes = filmes[filmes['ID_FILME'].str.isnumeric()]
    filmes['ID_FILME'] = filmes['ID_FILME'].astype(int)

    # Processamento de Avaliações
    avaliacoes.rename(columns={
        'userId': 'ID_USUARIO',
        'movieId': 'ID_FILME',
        'rating': 'AVALIACAO'
    }, inplace=True)
    qt_avaliacoes = avaliacoes['ID_USUARIO'].value_counts() > 999
    y = qt_avaliacoes[qt_avaliacoes].index
    avaliacoes = avaliacoes[avaliacoes['ID_USUARIO'].isin(y)]

    # Merge e Pivot
    avaliacoes_e_filmes = avaliacoes.merge(filmes, on='ID_FILME')
    avaliacoes_e_filmes.drop_duplicates(['ID_USUARIO', 'ID_FILME'], inplace=True)

    if avaliacoes_e_filmes.empty:
        st.error("Erro: Base de dados vazia após filtragem. Verifique seus critérios de corte.")
        st.stop()
        
    filmes_pivot = avaliacoes_e_filmes.pivot_table(
        columns='ID_USUARIO', index='TITULO', values='AVALIACAO'
    ).fillna(0)
    
    if filmes_pivot.shape[0] < 2:
        st.error("Erro: Poucos filmes restaram para construir o modelo.")
        st.stop()

    return filmes, avaliacoes_e_filmes, filmes_pivot


# ======================================
#  FUNÇÃO: TREINAMENTO DO MODELO (CACHE)
# ======================================
@st.cache_resource(show_spinner="Treinando o modelo de K-Vizinhos Mais Próximos (KNN)...")
def build_model(filmes_pivot):
    filmes_sparse = csr_matrix(filmes_pivot.values)
    modelo = NearestNeighbors(algorithm='brute')
    modelo.fit(filmes_sparse)
    return modelo


# ======================================
#  HELPERS PARA EXTRACAO DE PLOT E POSTER
# ======================================
def _extract_plot_from_obj(obj):
    """ Tenta extrair texto de plot/summary/plots/plotSummary de um objeto retornado pela API. """
    if not isinstance(obj, dict): return None
    p = obj.get('plot')
    if isinstance(p, str) and p.strip(): return p.strip()
    for key in ('summary', 'description', 'overview'):
        v = obj.get(key)
        if isinstance(v, str) and v.strip(): return v.strip()
    plots = obj.get('plots') or obj.get('plotText') or obj.get('plotTexts')
    if isinstance(plots, list) and len(plots) > 0:
        first = plots[0]
        if isinstance(first, str) and first.strip(): return first.strip()
        if isinstance(first, dict):
            for candidate in ('text', 'plainText', 'plotText', 'summary', 'plot'):
                tv = first.get(candidate)
                if isinstance(tv, str) and tv.strip(): return tv.strip()
    ps = obj.get('plotSummary') or obj.get('plot_summary')
    if isinstance(ps, dict):
        for candidate in ('text', 'summaryText', 'summary'):
            tv = ps.get(candidate)
            if isinstance(tv, str) and tv.strip(): return tv.strip()
    return None


def _extract_poster_from_obj(obj):
    """ Extrai URL do poster/primaryImage/url etc. """
    if not isinstance(obj, dict): return None
    primary = obj.get('primaryImage')
    if isinstance(primary, dict):
        url = primary.get('url')
        if isinstance(url, str) and url.strip(): return url.strip()
    for key in ("poster", "image", "poster_url", "posterLink", "image_url"):
        val = obj.get(key)
        if isinstance(val, dict):
            url = val.get('url')
            if isinstance(url, str) and url.strip(): return url.strip()
        elif isinstance(val, str) and val.strip():
            return val.strip()
    return None


# ======================================
#  FUNÇÃO: BUSCA NA API IMDB 
# ======================================
@st.cache_data(show_spinner=False)
def get_imdb_data(titulo):
    try:
        titulo_enc = requests.utils.requote_uri(titulo)
        url = f"https://api.imdbapi.dev/search/titles?query={titulo_enc}" 
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()

        titles = data.get("titles", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])

        movie = next((item for item in titles if isinstance(item.get("primaryImage"), dict) and item["primaryImage"].get("url")), titles[0] if titles else None)
        
        if movie is None: return None
        
        poster = _extract_poster_from_obj(movie)
        plot = _extract_plot_from_obj(movie)

        if not plot:
            movie_id = movie.get('id')
            if movie_id:
                try:
                    detail_url = f"https://api.imdbapi.dev/titles/{movie_id}"
                    resp2 = requests.get(detail_url, timeout=8)
                    resp2.raise_for_status()
                    detail_obj = resp2.json()
                    plot = _extract_plot_from_obj(detail_obj)
                    if not poster: poster = _extract_poster_from_obj(detail_obj)
                except Exception:
                    pass

        title = movie.get("primaryTitle") or movie.get("title") or movie.get("originalTitle") or titulo
        year = movie.get("startYear") or movie.get("year")
        
        imdb_rating = None
        imdb_votes = None
        rating_obj = movie.get("rating") or movie.get("ratings") or movie.get("aggregateRating")
        if isinstance(rating_obj, dict):
            imdb_rating = rating_obj.get("aggregateRating") or rating_obj.get("average") or rating_obj.get("value")
            imdb_votes = rating_obj.get("voteCount") or rating_obj.get("votes") or rating_obj.get("count")
        
        try:
            if imdb_rating is not None: imdb_rating = float(imdb_rating)
        except Exception: imdb_rating = None
        try:
            if imdb_votes is not None: imdb_votes = int(imdb_votes)
        except Exception: imdb_votes = None

        return {
            "title": title,
            "year": year,
            "poster": poster,
            "plot": plot or "Sem descrição disponível.",
            "imdb_rating": imdb_rating,
            "imdb_votes": imdb_votes
        }
    except Exception as e:
        return None

# ======================================
#  FUNÇÃO: BUSCA CASE-INSENSITIVE / SUBSTRING / FUZZY
# ======================================
def find_best_title_match(query, titles, filmes_df):
    """
    Retorna o título da lista 'titles' que melhor corresponde a 'query',
    ignorando case.
    """
    q = (query or "").strip()
    if not q: return None
    q_low = q.lower()

    # 1) igualdade exata (case-insensitive)
    for t in titles:
        if t.lower() == q_low: return t

    # 2) contains (case-insensitive)
    contains = [t for t in titles if q_low in t.lower()]
    if contains:
        if 'TITULO' in filmes_df.columns and 'QT_AVALIACOES' in filmes_df.columns:
            subset = filmes_df[filmes_df['TITULO'].isin(contains)]
            if not subset.empty:
                best = subset.sort_values('QT_AVALIACOES', ascending=False).iloc[0]['TITULO']
                return best
        return contains[0]

    # 3) correspondência aproximada (fuzzy) como fallback
    close = difflib.get_close_matches(q, titles, n=1, cutoff=0.6)
    if close: return close[0]

    return None
    
    
# ======================================
#  FUNÇÃO: BUSCA POR FILMES DA MESMA SÉRIE/TRILOGIA (NOVO e MELHORADO)
# ======================================
def get_trilogy_suggestion(main_title, all_titles):
    """
    Encontra outros filmes na base que compartilham um prefixo de título comum,
    melhorando a heurística para séries e trilogias.
    """
    if not main_title or len(main_title) < 5:
        return []

    base_title_cleaned = main_title
    
    # 1. Tenta remover números no final (Ex: "Toy Story 3" -> "Toy Story")
    base_title_cleaned = re.sub(r'\s+\d+$', '', base_title_cleaned).strip()
    
    # 2. Tenta remover subtítulos marcados por delimitadores comuns
    # Ex: "The Matrix: Reloaded" -> "The Matrix"
    delimiters = [':', ' Part ', ' Vol. ', ' - ']
    
    for sep in delimiters:
        if sep in base_title_cleaned:
            potential_base = base_title_cleaned.split(sep)[0].strip()
            # Se a base potencial for razoável (>= 4 caracteres) e não for a palavra inteira
            if len(potential_base) >= 4 and potential_base != base_title_cleaned:
                base_title_cleaned = potential_base
                break
    
    # Se o prefixo encontrado for muito curto ou não mudou, e o título principal é complexo,
    # pode não ser uma série (e.g. "The" de "The Movie").
    if len(base_title_cleaned) < 4:
        return []

    base_title_low = base_title_cleaned.lower()
    
    related_titles = []
    
    for t in all_titles:
        t_low = t.lower()
        
        # Critério 1: O título deve começar *exatamente* com o prefixo limpo
        if t_low.startswith(base_title_low):
            
            # Critério 2: A seguir ao prefixo, deve haver um separador ou ser o título exato.
            # (t_low == base_title_low) -> É o filme base, ex: "Star Wars"
            # (len > len and t_low[len] in (separators)) -> É a continuação, ex: "Star Wars: A New Hope"
            is_continuation = len(t_low) > len(base_title_low) and t_low[len(base_title_low)] in (' ', ':', '-', '(', '[')
            
            if t_low == base_title_low or is_continuation:
                 related_titles.append(t)

    # Garante que encontrou pelo menos dois filmes para chamar de "série"
    if len(related_titles) < 2:
        return []
        
    # Ordena alfabeticamente
    return sorted(list(set(related_titles)))


# ======================================
#  INICIALIZAÇÃO
# ======================================
with st.spinner("Iniciando o sistema de recomendação..."):
    filmes, avaliacoes_e_filmes, filmes_pivot = load_data()
    modelo = build_model(filmes_pivot)

# resumo das avaliações (do dataset ratings.csv, após merges) para mostrar notas/contagens internas
ratings_summary = avaliacoes_e_filmes.groupby('TITULO')['AVALIACAO'].agg(['mean', 'count'])
ratings_summary.rename(columns={'mean': 'avg_rating', 'count': 'num_ratings'}, inplace=True)
ratings_summary['avg_rating'] = ratings_summary['avg_rating'].round(2)

# Inicialização do estado da sessão (necessário para o Streamlit)
if 'run_recommendation' not in st.session_state:
    st.session_state['run_recommendation'] = False
if 'best_match' not in st.session_state:
    st.session_state['best_match'] = None

# ======================================
#  INTERFACE STREAMLIT
# ======================================
st.set_page_config(page_title="Recomendador de Filmes", page_icon="🎬", layout="wide")
st.title("🎬 Sistema de Recomendação de Filmes")
st.caption("Baseado em similaridade de avaliações (k-NN) + dados do IMDb")

titles = list(filmes_pivot.index)
chosen_from_list = st.selectbox("Escolha um filme (opcional)", options=[""] + titles)
nome_filme = st.text_input("Ou digite o nome do filme")

nome_filme_final = chosen_from_list if chosen_from_list else nome_filme.strip()

if st.button("Buscar recomendações", type="primary"):
    
    if not nome_filme_final:
        st.warning("Por favor digite ou escolha um filme antes de buscar.")
        st.session_state['run_recommendation'] = False
        st.session_state['best_match'] = None
    else:
        st.session_state['run_recommendation'] = True
        
        # Encontra a melhor correspondência ignorando case
        best_match = find_best_title_match(nome_filme_final, titles, filmes)
        
        if not best_match:
            st.error(f"O filme '{nome_filme_final}' não foi encontrado na base de dados.")
            st.session_state['run_recommendation'] = False
            st.session_state['best_match'] = None
        else:
            if best_match != nome_filme_final:
                # Linha removida para não exibir a mensagem de correspondência encontrada.
                pass 
            st.session_state['best_match'] = best_match

    
if st.session_state.get('run_recommendation', False) and st.session_state.get('best_match'):
    
    best_match = st.session_state['best_match']
    
    try:

        # --- 1. MOSTRA INFORMAÇÕES DO FILME PESQUISADO (Original) ---
        
        info_main = get_imdb_data(best_match)
        stats_main = ratings_summary.loc[best_match] if best_match in ratings_summary.index else None

        st.subheader("Filme pesquisado")
        col1, col2 = st.columns([1, 3])
        with col1:
            if info_main and info_main.get("poster"):
                # ALTERADO: Reduzido o tamanho da imagem principal de 180px para 120px
                st.image(info_main["poster"], width=180) 
            else:
                st.write("📷 Sem imagem")
        with col2:
            title_str = info_main['title'] if info_main and info_main.get('title') else best_match
            year_str = f" ({info_main.get('year')})" if info_main and info_main.get('year') else ""
            st.markdown(f"### {title_str}{year_str}")
            
            if info_main and info_main.get("imdb_rating") is not None:
                imdb_votes = info_main.get("imdb_votes")
                votes_text = f" ({imdb_votes:,} votos)".replace(",", ".") if imdb_votes else ""
                st.write(f"**Nota IMDb:** {info_main['imdb_rating']}{votes_text}")
            
            if stats_main is not None:
                st.write(f"**Nota média (usuários da base):** {stats_main['avg_rating']} / 5 ({int(stats_main['num_ratings']):,} avaliações)".replace(",", "."))
            
            if info_main and info_main.get("plot"):
                st.caption(info_main.get("plot"))

        # --- 2. EXIBIÇÃO DA TRILOGIA / SÉRIE (NOVO) ---
        trilogy_titles = get_trilogy_suggestion(best_match, titles)
        
        if trilogy_titles:
            st.markdown("---")
            # Tenta encontrar um nome base limpo para o cabeçalho
            base_name = re.sub(r'(:| Part| Vol\.| - ).*', '', trilogy_titles[0]).strip()
            # Usa o nome base se for razoável, senão usa o prefixo de todos os títulos
            base_name = base_name if len(base_name) > 3 else "Série Relacionada"
            st.subheader(f"🎞️ Filmes da Série/Trilogia '{base_name}'")
            
            # Exibe todos os filmes relacionados
            trilogy_cols = st.columns(len(trilogy_titles))
            for i, movie in enumerate(trilogy_titles):
                # O Streamlit só suporta até 10 colunas, limita o máximo para evitar erro no layout
                if i < 10: 
                    with trilogy_cols[i]:
                        info = get_imdb_data(movie)
                        movie_title = info['title'] if info and info.get('title') else movie
                        movie_year = f" ({info.get('year')})" if info and info.get('year') else ""
                        
                        st.markdown(f"**{i+1}. {movie_title}{movie_year}**")
                        if info and info.get("poster"):
                            # ALTERADO: Reduzido o tamanho da imagem da trilogia para 120px
                            st.image(info["poster"], width=180) 
                        else:
                            st.write("📷 Sem imagem")
                        
                        if movie == best_match:
                            st.markdown("**_Filme de partida_**")
                        
                        # Exibe uma prévia do plot
                        plot_text = info['plot'] if info and info.get("plot") and info['plot'] != "Sem descrição disponível." else "_Sem descrição._"
                        st.caption(f"_{plot_text[:100]}..._")
            
            st.markdown("---") # Separador antes do filme pesquisado/recomendação


        # --- 3. RECOMENDADOS POR SIMILARIDADE (KNN) (Original) ---
        
        query_vec = filmes_pivot.filter(items=[best_match], axis=0).values.reshape(1, -1)
        distances, sugestions = modelo.kneighbors(query_vec, n_neighbors=6)
        recommended = filmes_pivot.index[sugestions[0][1:]]

        st.subheader("Recomendados por Similaridade (k-NN):")
        cols = st.columns(len(recommended))
        
        for i, movie in enumerate(recommended):
            with cols[i]:
                info = get_imdb_data(movie)
                stats = ratings_summary.loc[movie] if movie in ratings_summary.index else None
                
                movie_title = info['title'] if info and info.get('title') else movie
                movie_year = f" ({info.get('year')})" if info and info.get('year') else ""
                
                st.markdown(f"**{i+1}. {movie_title}{movie_year}**")
                
                if info and info.get("poster"):
                    # ALTERADO: Reduzido o tamanho das imagens recomendadas para 120px
                    st.image(info["poster"], width=120)
                else:
                    st.write("📷 Sem imagem")

                st.caption(f"Distância: **{distances[0][i+1]:.4f}** (Menor é melhor)")
                
                if info and info.get("plot"):
                    st.markdown(f"*{info['plot'][:100]}...*")

    except Exception as e:
        st.error(f"Erro ao buscar recomendações: {e}")

st.markdown("""
---
*Nota: Este sistema funciona com dados filtrados para filmes populares e usuários ativos. A busca parcial ajuda a encontrar o título exato para o modelo de recomendação.*
""")
