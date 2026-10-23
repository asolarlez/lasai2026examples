"""Construye un índice BM25 de las páginas HTML del directorio Wikipedia.

El índice guarda, para cada palabra, su peso BM25 en cada documento donde
aparece, de modo que buscar es solo sumar pesos.

Uso:
    python createDB.py
"""

import json
import math
import os
import re
from collections import Counter
from html.parser import HTMLParser

WIKI_DIR = "Wikipedia"
INDEX_FILE = "index.json"
K1 = 1.5
B = 0.75

# Common English words (the pages are from English Wikipedia) that carry no meaning
# for search: articles, pronouns, prepositions, conjunctions and auxiliary verbs.
STOPWORDS = set("""
a an the this that these those
i me my we our us you your he him his she her it its they them their
and or but nor so if then than because while as
of in on at to from by for with about into onto over under between through
during before after above below up down out off again further
is are was were be been being am do does did done have has had having
will would shall should can could may might must
not no only own same too very also just more most other some such
what which who whom whose when where why how all any both each few
there here s t
""".split())

# Content ends where the reference sections and page footer begin.
END_MARKERS = ['id="References"', 'id="Notes"', 'id="Citations"', 'id="Footnotes"',
               'class="printfooter"']
# Elements whose text is not article content (citation marks, [edit] links, ...).
SKIP_TAGS = {"script", "style", "sup"}
SKIP_CLASSES = ("mw-editsection", "navbox", "reference", "noprint")


class TextExtractor(HTMLParser):
    """Collect visible text, skipping SKIP_TAGS and elements with SKIP_CLASSES."""

    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip_tag = None
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if self.skip_tag:
            self.skip_depth += tag == self.skip_tag
            return
        classes = dict(attrs).get("class") or ""
        if tag in SKIP_TAGS or any(c in classes for c in SKIP_CLASSES):
            self.skip_tag, self.skip_depth = tag, 1

    def handle_endtag(self, tag):
        if self.skip_tag == tag:
            self.skip_depth -= 1
            if self.skip_depth == 0:
                self.skip_tag = None

    def handle_data(self, data):
        if not self.skip_tag:
            self.parts.append(data)


def extract_text(path):
    """Return the article text of a saved Wikipedia HTML page."""
    with open(path, encoding="utf-8", errors="ignore") as f:
        html = f.read()
    # Start right after the opening tag of the content <div>.
    start = html.find(">", max(html.find('id="mw-content-text"'), 0)) + 1
    ends = [i for i in (html.find(m, start) for m in END_MARKERS) if i != -1]
    parser = TextExtractor()
    parser.feed(html[start:min(ends, default=len(html))])
    return re.sub(r"\s+", " ", " ".join(parser.parts)).strip()


def tokenize(text):
    """Lowercase words of the text, without stopwords or single characters."""
    return [w for w in re.findall(r"\w+", text.lower()) if len(w) > 1 and w not in STOPWORDS]


def title_from_file(filename):
    return os.path.splitext(filename)[0].replace("_", " ")


def bm25_weights(term_counts):
    """Map each term to [[doc_id, bm25 weight], ...] given per-document term counts."""
    n_docs = len(term_counts)
    lengths = [sum(c.values()) for c in term_counts]
    avg_len = sum(lengths) / n_docs
    doc_freq = Counter(term for counts in term_counts for term in counts)

    weights = {}
    for doc_id, (counts, length) in enumerate(zip(term_counts, lengths)):
        norm = K1 * (1 - B + B * length / avg_len)
        for term, tf in counts.items():
            df = doc_freq[term]
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            weights.setdefault(term, []).append([doc_id, round(idf * tf * (K1 + 1) / (tf + norm), 3)])
    return weights


def build_index():
    files = sorted(f for f in os.listdir(WIKI_DIR) if f.endswith(".html"))
    docs, term_counts = [], []
    for i, filename in enumerate(files, 1):
        path = os.path.join(WIKI_DIR, filename)
        title = title_from_file(filename)
        # Title words count twice so a page ranks high for its own subject.
        term_counts.append(Counter(tokenize(title) * 2 + tokenize(extract_text(path))))
        docs.append({"file": path, "title": title})
        if i % 200 == 0:
            print(f"  {i}/{len(files)} documentos procesados")
    return {"docs": docs, "weights": bm25_weights(term_counts)}


def main():
    print(f"Indexando {WIKI_DIR}/ ...")
    index = build_index()
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, separators=(",", ":"))
    print(f"Listo: {len(index['docs'])} documentos, {len(index['weights'])} palabras -> {INDEX_FILE}")


if __name__ == "__main__":
    main()
