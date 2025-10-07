# 🎬 Documentação: Sistema de Recomendação de Filmes

Bem-vindo à documentação do Sistema de Recomendação de Filmes! Este guia foi criado para explicar de forma clara e amigável como o código `movie_recommender.py` funciona.

## 1. Visão Geral do Projeto

Este projeto é uma aplicação web, construída com **Streamlit**, que recomenda filmes para os usuários. A lógica de recomendação principal é baseada no método de filtragem colaborativa, usando o algoritmo **k-Vizinhos Mais Próximos (k-NN)**.

O sistema funciona da seguinte maneira:
1.  **Carrega e processa** dados de filmes e avaliações de usuários.
2.  **Treina um modelo** que aprende a "distância" entre os filmes com base nos padrões de avaliação dos usuários.
3.  **Recebe o nome de um filme** do usuário através de uma interface web.
4.  **Encontra recomendações** buscando os filmes mais "próximos" (similares) ao filme inserido.
5.  **Busca dados adicionais**, como pôsteres e sinopses, de uma API externa (IMDb) para enriquecer a exibição.
6.  **Identifica e exibe** outros filmes da mesma saga ou trilogia.

---

## 2. Estrutura do Código

O código é dividido em seções lógicas:
- **Configurações Globais**: Define caminhos de arquivos e parâmetros.
- **Carregamento de Dados**: Funções para ler e preparar os dados.
- **Treinamento do Modelo**: Função para construir o modelo de recomendação.
- **Funções de Apoio (Helpers)**: Funções para buscar dados em APIs e fazer buscas inteligentes de títulos.
- **Interface do Usuário (Streamlit)**: Onde a mágica acontece e a aplicação é exibida.

Vamos detalhar cada parte.

### 2.1. Funções Principais

#### `load_data()`
Esta é a primeira função executada e é responsável por toda a preparação dos dados. O decorador `@st.cache_data` garante que os dados sejam carregados apenas uma vez, tornando a aplicação muito mais rápida após a primeira inicialização.

**O que ela faz?**
1.  **Carrega os Arquivos**: Lê `movies_metadata.csv` (informações dos filmes) e `ratings.csv` (avaliações dos usuários).
2.  **Limpeza de Filmes**:
    - Mantém apenas as colunas importantes (`id`, `título`, `idioma`, etc.).
    - Filtra os filmes para manter apenas os mais relevantes (mais de 999 avaliações e em inglês).
3.  **Limpeza de Avaliações**:
    - Filtra as avaliações para manter apenas as de usuários muito ativos (que avaliaram mais de 999 filmes). Isso ajuda a criar um modelo mais robusto.
4.  **Cria a Tabela Pivot**: Esta é a etapa mais importante. A função transforma os dados em uma grande matriz onde:
    - As **linhas** são os títulos dos filmes.
    - As **colunas** são os IDs dos usuários.
    - Os **valores** são as notas que cada usuário deu para cada filme.

O resultado final é uma tabela `filmes_pivot` que serve de base para o modelo.

#### `build_model(filmes_pivot)`
Com os dados prontos, esta função treina o modelo de recomendação. O decorador `@st.cache_resource` armazena o modelo treinado em cache, evitando o re-treinamento a cada interação do usuário.

**O que ela faz?**
1.  **Converte para Matriz Esparsa**: A tabela pivot é convertida para uma `csr_matrix`, um formato eficiente para matrizes com muitos zeros (já que um usuário não avaliou todos os filmes).
2.  **Cria e Treina o Modelo**: Instancia o `NearestNeighbors` com o algoritmo `brute` (força bruta), que calculará a distância entre todos os filmes. O método `.fit()` treina o modelo com os dados dos filmes.

#### `get_imdb_data(titulo)`
Esta função busca informações extras sobre um filme para deixar a interface mais bonita e informativa. Ela usa uma API pública do IMDb.

**O que ela faz?**
1.  **Busca na API**: Envia o título do filme para a API.
2.  **Extrai Dados**: Procura por informações úteis no retorno da API, como:
    - Pôster do filme (`poster`).
    - Sinopse (`plot`).
    - Ano de lançamento (`year`).
    - Nota no IMDb (`imdb_rating`).
3.  **Tratamento de Erros**: Se a busca falhar ou a API não retornar dados, ela retorna `None` para que a aplicação não quebre.

#### `find_best_title_match(query, titles, filmes_df)`
Esta função melhora a experiência do usuário. Como os usuários podem digitar o nome de um filme de várias maneiras ("Pulp Fiction", "pulp fiction", "pulp"), esta função encontra o título exato que existe na nossa base de dados.

**Como ela funciona (em ordem de prioridade)?**
1.  **Busca Exata**: Procura por um título que seja exatamente igual ao digitado (ignorando maiúsculas/minúsculas).
2.  **Busca por "Contém"**: Se não encontrar, procura por títulos que contenham o texto digitado. Se houver vários, retorna o mais popular (com mais avaliações).
3.  **Busca por Aproximação (Fuzzy)**: Como último recurso, usa a biblioteca `difflib` para encontrar o título mais "parecido" com o que foi digitado.

#### `get_trilogy_suggestion(main_title, all_titles)`
Uma função inteligente para encontrar outros filmes de uma mesma franquia.

**O que ela faz?**
1.  **Limpa o Título**: Pega um título como "The Dark Knight Rises" e tenta extrair a base, como "The Dark Knight". Ela remove números no final ou subtítulos (como ": Reloaded").
2.  **Busca por Prefixo**: Procura na lista de todos os filmes por outros que comecem com essa base.
3.  **Retorna a Lista**: Se encontrar mais de um filme relacionado, retorna a lista organizada, pronta para ser exibida.

---

### 2.2. Interface do Usuário (Streamlit)

Esta é a última parte do código, onde tudo se junta para criar a página web.

**Principais componentes:**
- `st.set_page_config()`: Configura o título e o ícone da aba do navegador.
- `st.title()` e `st.caption()`: Exibem o título principal da página.
- `st.selectbox()` e `st.text_input()`: Criam os campos para o usuário escolher ou digitar um filme.
- `st.button()`: O botão "Buscar recomendações" que dispara a lógica principal.
- `st.session_state`: Uma forma de "memória" do Streamlit. Usamos para saber se o botão foi clicado e qual filme foi encontrado, para que a página não perca essas informações se for recarregada.

**Quando o botão é clicado:**
1.  O `find_best_title_match()` encontra o filme na base.
2.  As informações do filme pesquisado são exibidas usando `get_imdb_data()`.
3.  A função `get_trilogy_suggestion()` é chamada para mostrar filmes da mesma saga.
4.  O modelo `kneighbors()` é usado para encontrar os 5 filmes mais similares.
5.  As recomendações são exibidas em colunas (`st.columns`), cada uma com seu pôster, título e uma breve sinopse.

---

## 3. Conclusão

Este código é um excelente exemplo de como combinar análise de dados (Pandas), Machine Learning (Scikit-learn) e desenvolvimento web (Streamlit) para criar uma aplicação funcional e interativa. As funções de cache (`@st.cache_data` e `@st.cache_resource`) são cruciais para garantir um bom desempenho.

Esperamos que esta documentação tenha ajudado a entender melhor o funcionamento do projeto!
