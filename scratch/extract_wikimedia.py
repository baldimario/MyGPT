import pyarrow.parquet as pq

LIMIT = 500 * 1024**2  # tetto sul testo scritto, non sul parquet letto
pf = pq.ParquetFile("data/it-wiki-00.parquet")
written = n = 0
with open("data/input-it.txt", "w", encoding="utf-8") as out:
    for batch in pf.iter_batches(batch_size=1000, columns=["title", "text"]):
        for title, text in zip(
            batch.column("title").to_pylist(), batch.column("text").to_pylist()
        ):
            # il titolo come intestazione: da' al modello un segnale di struttura
            # del documento, come faceva "NOME:" con le battute di Shakespeare
            doc = f"= {title} =\n\n{text}\n\n"
            out.write(doc)
            written += len(doc.encode())
            n += 1
        if written >= LIMIT:
            break
print(f"{n:,} articoli, {written / 1024**2:.0f} MB -> data/input-it.txt")
