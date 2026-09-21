import os
import re
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore')

ROOT          = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / 'data' / 'processed'
FEATURES_DIR  = ROOT / 'data' / 'features'
FEATURES_DIR.mkdir(parents=True, exist_ok=True)


def clean_for_nlp_series(text_series):
    return (
        text_series.astype(str).str.lower()
        .str.replace(r'https?://\S+|www\.\S+', '', regex=True)
        .str.replace(r'@\w+', '', regex=True)
        .str.replace(r'#', '', regex=False)
        .str.replace(r'[^\w\s]', ' ', regex=True)
        .str.split().str.join(' ')
    )


def compute_stylometrics_vectorized(tweets_df):
    t = tweets_df['text'].astype(str)
    char_len = t.str.len()
    POSITIVE = ['good','great','awesome','happy','love','best','wonderful','fantastic','amazing','enjoy','glad','cool','perfect','thank']
    NEGATIVE = ['bad','terrible','horrible','worst','hate','fake','scam','shame','fail','awful','corrupt','crisis','fraud','lies']
    tl = t.str.lower()
    pos = sum(tl.str.contains(w, regex=False).astype(float) for w in POSITIVE)
    neg = sum(tl.str.contains(w, regex=False).astype(float) for w in NEGATIVE)
    tot = pos + neg
    return pd.DataFrame({
        'user_id':         tweets_df['user_id'],
        'char_len':        char_len,
        'word_count':      t.str.count(r'\w+'),
        'uppercase_ratio': t.str.count(r'[A-Z]') / (char_len + 1e-9),
        'punct_count':     t.str.count(r'[!?]'),
        'digit_ratio':     t.str.count(r'\d') / (char_len + 1e-9),
        'url_count':       t.str.count(r'https?://\S+'),
        'hashtag_count':   t.str.count(r'#\w+'),
        'mention_count':   t.str.count(r'@\w+'),
        'sentiment':       ((pos - neg) / tot.replace(0, np.nan)).fillna(0.0),
        'clean_text':      clean_for_nlp_series(t),
    })


def compute_user_lexical_features(tweets_df):
    print('  Computing per-tweet stylometrics...', flush=True)
    enriched = compute_stylometrics_vectorized(tweets_df)
    user_agg = enriched.groupby('user_id').agg(
        avg_tweet_chars       =('char_len',        'mean'),
        avg_word_count        =('word_count',       'mean'),
        uppercase_ratio       =('uppercase_ratio',  'mean'),
        punctuation_intensity =('punct_count',      'mean'),
        digit_ratio           =('digit_ratio',      'mean'),
        url_density           =('url_count',        'mean'),
        hashtag_density       =('hashtag_count',    'mean'),
        mention_density       =('mention_count',    'mean'),
        sentiment_polarity    =('sentiment',        'mean'),
    ).reset_index()
    print('  Computing lexical diversity per user...', flush=True)

    def vocab_stats(texts):
        n, u = len(texts), len(set(texts))
        tokens = ' '.join(texts).split()
        return pd.Series({
            'lexical_diversity':   len(set(tokens)) / max(len(tokens), 1),
            'tweet_repetition_rate': 1.0 - u / max(n, 1),
        })

    vocab_df = enriched.groupby('user_id')['clean_text'].apply(vocab_stats).unstack().reset_index()
    return user_agg.merge(vocab_df, on='user_id', how='left')


def build_user_docs(tweets_df):
    tweets_df = tweets_df.copy()
    tweets_df['clean_text'] = clean_for_nlp_series(tweets_df['text'])
    return (tweets_df.groupby('user_id')['clean_text']
            .apply(lambda s: ' '.join(s.dropna()))
            .reset_index()
            .rename(columns={'clean_text': 'user_doc'}))


def compute_tfidf_lsa(user_docs, n_topics=8, n_emb=16):
    print(f'  Fitting TF-IDF on {len(user_docs):,} user documents...', flush=True)
    tfidf = TfidfVectorizer(max_features=10_000, min_df=2, max_df=0.95,
                            ngram_range=(1,2), sublinear_tf=True)
    X = tfidf.fit_transform(user_docs['user_doc'])
    print(f'  TF-IDF matrix: {X.shape}', flush=True)

    print(f'  LSA topics ({n_topics} components)...', flush=True)
    svd_t     = TruncatedSVD(n_components=n_topics, random_state=42)
    topic_v   = svd_t.fit_transform(X)
    topics_df = pd.DataFrame(topic_v, columns=[f'tfidf_topic_{i}' for i in range(n_topics)])
    topics_df['user_id'] = user_docs['user_id'].values

    print(f'  LSA embeddings ({n_emb} dims)...', flush=True)
    svd_e  = TruncatedSVD(n_components=n_emb, random_state=42)
    emb_v  = svd_e.fit_transform(X)
    emb_df = pd.DataFrame(emb_v, columns=[f'text_emb_{i}' for i in range(n_emb)])
    emb_df['user_id'] = user_docs['user_id'].values

    return topics_df, emb_df, tfidf, user_docs


def compute_intra_user_sim(tweets_df, tfidf, max_sample=3):
    print('  Computing intra-user TF-IDF cosine similarities...', flush=True)
    tweets_df = tweets_df.copy()
    tweets_df['clean_text'] = clean_for_nlp_series(tweets_df['text'])
    sampled = (tweets_df.groupby('user_id')['clean_text']
               .apply(lambda s: list(s.dropna().head(max_sample)))
               .reset_index().rename(columns={'clean_text': 'texts'}))
    user_sims = {}
    for _, row in sampled.iterrows():
        uid, texts = row['user_id'], row['texts']
        if len(texts) < 2:
            user_sims[uid] = 0.0
            continue
        vecs = tfidf.transform(texts)
        sims = cosine_similarity(vecs)
        n    = len(texts)
        user_sims[uid] = float(sims[np.triu_indices(n, k=1)].mean())
    return pd.DataFrame(list(user_sims.items()), columns=['user_id', 'intra_user_semantic_sim'])


def analyze_top_discriminative_words(user_docs, nodes_df, tfidf, top_n=15):
    merged = user_docs.merge(nodes_df[['user_id','account_type']], on='user_id', how='left')
    names  = np.array(tfidf.get_feature_names_out())
    bm = np.asarray(tfidf.transform(merged[merged['account_type']=='bot']['user_doc']).mean(axis=0)).flatten()
    hm = np.asarray(tfidf.transform(merged[merged['account_type']=='human']['user_doc']).mean(axis=0)).flatten()
    diff = bm - hm
    return ([(names[i], float(diff[i]))  for i in np.argsort(diff)[-top_n:][::-1]],
            [(names[i], float(-diff[i])) for i in np.argsort(diff)[:top_n]])


def run_nlp_feature_pipeline():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print('=' * 70, flush=True)
    print('Module 5 - NLP Text Feature Extraction Pipeline (sklearn-only)', flush=True)
    print('=' * 70, flush=True)

    nodes_df  = pd.read_csv(PROCESSED_DIR / 'nodes_clean.csv',  low_memory=False)
    tweets_df = pd.read_csv(PROCESSED_DIR / 'tweets_clean.csv', low_memory=False)
    print(f'Loaded {len(nodes_df):,} nodes and {len(tweets_df):,} tweets.\n', flush=True)

    print('[Step 1/4] Stylometrics & Lexical Diversity', flush=True)
    lexical_df = compute_user_lexical_features(tweets_df)
    print(f'  done: {lexical_df.shape}\n', flush=True)

    print('[Step 2/4] TF-IDF + LSA Topics + LSA Dense Embeddings', flush=True)
    user_docs = build_user_docs(tweets_df)
    topics_df, emb_df, tfidf, user_docs = compute_tfidf_lsa(user_docs, n_topics=8, n_emb=16)
    print(f'  done: topics={topics_df.shape}  emb={emb_df.shape}\n', flush=True)

    print('[Step 3/4] Intra-User Semantic Similarity (TF-IDF cosine)', flush=True)
    sim_df = compute_intra_user_sim(tweets_df, tfidf)
    print(f'  done: {sim_df.shape}\n', flush=True)

    print('[Step 4/4] Merge, scale & save', flush=True)
    final_df = nodes_df[['user_id','account_type','label']].copy()
    for df in [lexical_df, sim_df, topics_df, emb_df]:
        final_df = final_df.merge(df, on='user_id', how='left')

    fcols = [c for c in final_df.columns if c not in ['user_id','account_type','label']]
    final_df[fcols] = final_df[fcols].fillna(0.0)
    final_df[fcols] = StandardScaler().fit_transform(final_df[fcols].astype(float))

    out_csv = FEATURES_DIR / 'text_features.csv'
    final_df.to_csv(out_csv, index=False)
    print(f'  Saved: {out_csv.name}  shape={final_df.shape}  {out_csv.stat().st_size//1024} KB', flush=True)

    with open(FEATURES_DIR / 'text_feature_columns.json', 'w') as f:
        json.dump({'text_feature_columns': fcols, 'n_text_features': len(fcols),
                   'groups': {'stylometric_lexical': fcols[:11],
                              'semantic_similarity': ['intra_user_semantic_sim'],
                              'lsa_topics': [f'tfidf_topic_{i}' for i in range(8)],
                              'lsa_embeddings': [f'text_emb_{i}' for i in range(16)]}}, f, indent=2)

    # Plots
    print('\nGenerating visualisations...', flush=True)
    tc = {'human': '#4CAF50', 'bot': '#F44336', 'suspicious': '#FF9800'}

    top_bot, top_human = analyze_top_discriminative_words(user_docs, nodes_df, tfidf)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    b_w, b_s = zip(*top_bot);   ax1.barh(b_w[::-1], b_s[::-1], color='#F44336', alpha=0.85)
    h_w, h_s = zip(*top_human); ax2.barh(h_w[::-1], h_s[::-1], color='#4CAF50', alpha=0.85)
    ax1.set_title('Top Bot-Discriminating Words', fontsize=11, fontweight='bold')
    ax2.set_title('Top Human-Discriminating Words', fontsize=11, fontweight='bold')
    plt.suptitle('Module 5 - Discriminative Vocabulary', fontsize=13, fontweight='bold')
    plt.tight_layout(); plt.savefig(FEATURES_DIR / 'bot_vs_human_keywords.png', dpi=120); plt.close()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    for atype, color in tc.items():
        sub = final_df[final_df['account_type'] == atype]
        ax1.hist(sub['lexical_diversity'],       bins=30, alpha=0.55, color=color, label=atype, density=True)
        ax2.hist(sub['intra_user_semantic_sim'], bins=30, alpha=0.55, color=color, label=atype, density=True)
    ax1.set_title('Lexical Diversity', fontsize=10, fontweight='bold'); ax1.legend()
    ax2.set_title('Intra-User Similarity', fontsize=10, fontweight='bold'); ax2.legend()
    plt.suptitle('Module 5 - Stylometric Distributions', fontsize=13, fontweight='bold')
    plt.tight_layout(); plt.savefig(FEATURES_DIR / 'lexical_and_similarity.png', dpi=120); plt.close()

    fig, ax = plt.subplots(figsize=(9, 6.5))
    for lbl, name, color in [(0,'Human','#4CAF50'),(1,'Bot','#F44336'),(2,'Suspicious','#FF9800')]:
        sub = final_df[final_df['label'] == lbl]
        ax.scatter(sub['text_emb_0'], sub['text_emb_1'], s=8, alpha=0.35, color=color, label=name)
    ax.set_title('LSA Text Embedding - Top 2 Components', fontsize=12, fontweight='bold')
    ax.legend(markerscale=3); plt.tight_layout()
    plt.savefig(FEATURES_DIR / 'nlp_embeddings_tsne.png', dpi=120); plt.close()

    assert final_df['user_id'].nunique() == len(final_df)
    assert final_df[fcols].isna().sum().sum() == 0
    print('\nAll assertions passed!', flush=True)
    print(f'  {len(fcols)} features x {len(final_df):,} users -> text_features.csv', flush=True)
    print('Module 5 NLP Pipeline Completed Successfully!', flush=True)


if __name__ == '__main__':
    run_nlp_feature_pipeline()
