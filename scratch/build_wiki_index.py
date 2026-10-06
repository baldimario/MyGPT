# indice BM25 (SQLite FTS5) sugli incipit di tutti gli articoli di it.wikipedia -> data/it-wiki-leads.sqlite
# ponytail: solo i primi LEAD caratteri di ogni articolo; per i fatti nel corpo dell'articolo servono passaggi su tutto il testo
import sqlite3
import time

import pyarrow.parquet as pq

LEAD = 1000
db = sqlite3.connect("data/it-wiki-leads.sqlite")
db.execute("drop table if exists leads")
# unicode61 senza diacritici: "città" e "citta" sono lo stesso termine
db.execute("create virtual table leads using fts5(title, body, tokenize='unicode61 remove_diacritics 2')")
t, n = time.time(), 0
for i in range(10):
    pf = pq.ParquetFile(f"data/it-wiki-{i:02d}.parquet")
    for batch in pf.iter_batches(batch_size=5000, columns=["title", "text"]):
        rows = []
        for title, text in zip(batch.column("title").to_pylist(), batch.column("text").to_pylist()):
            cut = text.rfind(". ", 0, LEAD)  # taglia a fine frase se possibile
            rows.append((title, text[: cut + 1] if cut > LEAD // 2 else text[:LEAD]))
        db.executemany("insert into leads values (?, ?)", rows)
        n += len(rows)
    db.commit()
    print(f"shard {i}: {n:,} articoli ({time.time() - t:.0f}s)", flush=True)
db.execute("insert into leads(leads) values('optimize')")
db.commit()
print(f"fatto: {n:,} articoli in {time.time() - t:.0f}s")
