LayerNorm fa due operazioni, centra e riscala

```
LN(x) = γ · (x − μ) / √(σ² + ε) + β
```

la RMSNorm tiene solo la seconda

```
RMS(x) = √(mean(x²) + ε)  RMSNorm(x) = γ · x / RMS(x)
```
non c'é media da sottrarre E non c'é beta, notiamo che se la media di x é zero RMS e deviazione standard coincidono, le due norme differiscono solo per quanto il vettore é spostato dallo zero

cosa perdiamo geometricamente togliendo la centratura?

consideriamo il vettore del token come un punto in R^c

la sottrazione della media é una proiezione, rimuove la componente lungo la direzione (1, 1, ..., 1), dopo la Layernorm il vettore vive in un sottospazio di C-1 dimensioni e la rete ha perso una direzione

la divisione per l'RMS porta il cettore sulla sfera di raggio sqrt(C), la direzione resta intatta, cambia solo la lunghezza

la RMSNorm fa solo il secondo passo, se diamo in input x+3 la LayerNorm cancellerebbe lo spostamento mentre la RMSNorm lascia una media di 0.95 quindi la direzione "tutti uno" sopravvive

ma perché funziona uguale?

l'ipotesi di Zhang & Sennrich (2019) é che il merito della LayerNorm stia nell'invarianza di scala non nella centratura

```
RMSNorm(α·x) = RMSNorm(x)  per ogni α > 0
```

é la proprietà che ci serve nel pre-LN, il residual stream cresce di blocco in blocco perché ogni blocco ci somma sopra le proprie attivazioni, chi legge dal residual deve vedere un segnale sempre alla stessa scala qualunque sia la profondità, la centratura non contribuisce a questo

c'é anche un effetto collaterale sul gradiente, una funzione invariante alla scala di c ha gradiente ortogonale a x, spingere x più lontano nella sua stessa direzione non cambia niente, questo rende l'addestramento meno sensibile alla grandezza delle attivazoni, la LayerNorm ha la stessa proprietà quindi da questo lato non perdiamo nulla

Ma perché togliere anche β?

Nel pre-LN l'output della norm entra sempre in Un Linear

```
W(γ·x̂ + β) + b = W(γ·x̂) + (Wβ + b)
```

β passa attraverso W e diventa un bias costante, dove il Linear ha già un bias, come fc dell'MLP, β é ridondante. Dove non ce l'ha, come c_attn con bias=False, β faceva da bias nascosto di q, k e v ma é un contributo marginale


