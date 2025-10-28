# Vercel Python Serverless Function
#  /api/recommend?list=1                  -> returns list of titles
#  /api/recommend?title=Inception&k=10    -> returns recommendations

import json, os, glob
import pandas as pd
import numpy as np
from collections import Counter
import math

_df = None
_titles = None
_title_col = None
_vocab = None
_idf = None
_doc_tf = None
_index_by_title = None

def _find_csv_path():
    candidates = glob.glob("data/**/*.csv", recursive=True) + glob.glob("*.csv")
    if not candidates:
        raise RuntimeError("No CSV files found. Put a movies CSV under /data or repo root.")
    pref = [c for c in candidates if "movie" in os.path.basename(c).lower()] or candidates
    return pref[0]

def _tokenize(s):
    return [w for w in "".join(ch.lower() if ch.isalnum() else " " for ch in s).split() if w]

def _build_text(row, title_col, df):
    text_cols = [c for c in ["overview","genres","keywords","cast","crew","tagline","description"]
                 if c in df.columns]
    if not text_cols:
        text_cols = [c for c in df.select_dtypes(include="object").columns if c != title_col]
    parts = [str(row.get(c, "")) for c in text_cols]
    return " ".join(parts)

def _ensure_loaded():
    global _df, _titles, _title_col, _vocab, _idf, _doc_tf, _index_by_title
    if _df is not None:
        return

    csv_path = _find_csv_path()
    df = pd.read_csv(csv_path, low_memory=False)

    title_col = next((c for c in ["title","original_title","movie","name"] if c in df.columns), None)
    if title_col is None:
        obj = df.select_dtypes(include="object").columns.tolist()
        if not obj:
            raise RuntimeError("No suitable title column found.")
        title_col = obj[0]
    df = df.dropna(subset=[title_col]).reset_index(drop=True)

    docs = []
    for _, row in df.iterrows():
        docs.append(_build_text(row, title_col, df))

    token_docs = [Counter(_tokenize(t)) for t in docs]

    df_count = Counter()
    for c in token_docs:
        for term in c.keys():
            df_count[term] += 1

    N = len(token_docs)
    vocab = {term:i for i, term in enumerate(df_count.keys())}
    idf = {term: math.log((N + 1) / (dfreq + 1)) + 1.0 for term, dfreq in df_count.items()}

    _df = df
    _titles = df[title_col].astype(str).tolist()
    _title_col = title_col
    _vocab = vocab
    _idf = idf
    _doc_tf = token_docs
    _index_by_title = {t:i for i,t in enumerate(_titles)}

def _cosine_sim_tfidf(i, j):
    ai = _doc_tf[i]; aj = _doc_tf[j]
    si = {t:(1+math.log(f))*_idf.get(t,0.0) for t,f in ai.items()}
    sj = {t:(1+math.log(f))*_idf.get(t,0.0) for t,f in aj.items()}
    dot = 0.0
    if len(si) < len(sj):
        for t, w in si.items(): dot += w * sj.get(t, 0.0)
    else:
        for t, w in sj.items(): dot += w * si.get(t, 0.0)
    ni = math.sqrt(sum(w*w for w in si.values()))
    nj = math.sqrt(sum(w*w for w in sj.values()))
    if ni == 0 or nj == 0: return 0.0
    return dot / (ni * nj)

def handler(request):
    try:
        _ensure_loaded()
        qs = request.get("queryStringParameters") or {}
        if "list" in qs:
            return {"statusCode":200, "headers":{"Content-Type":"application/json"},
                    "body": json.dumps({"titles": _titles[:2000]})}

        title = qs.get("title"); k = int(qs.get("k","10"))
        if not title or title not in _index_by_title:
            return {"statusCode":400, "headers":{"Content-Type":"application/json"},
                    "body": json.dumps({"error":"Provide ?title=<movie in list>&k=10"})}

        i = _index_by_title[title]
        sims = [(j, _cosine_sim_tfidf(i,j)) for j in range(len(_titles)) if j != i]
        sims.sort(key=lambda x: -x[1])
        recs = [_titles[j] for j,_ in sims[:k]]

        return {"statusCode":200, "headers":{"Content-Type":"application/json"},
                "body": json.dumps({"title": title, "recommendations": recs})}
    except Exception as e:
        return {"statusCode":500, "headers":{"Content-Type":"application/json"},
                "body": json.dumps({"error": str(e)})}
