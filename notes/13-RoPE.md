Implementiamo RoPe che rende esatta la KV cache oltre block_size, il limite riscontrato con la KV cache semplice, con la wpe assoluta, tagliare i token vecchi falsa le psoizioni di quelli rimasti, con RoPE l'attention vede solo differenze di posizione e nella finestra tagliata le differenze restano quelle giuste. Il compromesso sparisce da solo

Invece di sommare un vettore di posizione all'input, ruotiamo q e k di un angolo proporzionale alla loro psoizione, il prodotto scalare fra una query ruotata di m e una key ruotata di n dipende solo da m-n, la rotazione assoluta si cancella nella differenza, la posizione smette di essere un'informazione sommata al contenuto e diventa una proprietà geometrica di come i token si guardano

prendiamo dunque delle decisioni
- wpe sparisce, -98304 parametri, RoPE non ne aggiunge nessuno, cos e sin si calcolano al volo, quindi passiamo da 10764288 a 10665984 parametri- si ruotano q e k non v, la rotazione deve agire dove si misura l asomiglianza, il valore trasportato non c'entra, ruotarlo sposterebbe il contenuto non la psoizione
t0 lo legge l'attention dalla sua cache, non lo passa GPT.foward, ogni layer ha esattamente la stessa lunghezza di cache, quindi sa da se da dove ripartono i token nuovi, quindi non ci sono cambi di firma in nel transformer Block
- il test di equivalenza fuso-vs-native va tolto perché adesso la matematica è cambiata apposta, resta però il test di causalità che per regola vale per ogni variante di attention

Questo ha una conseguenza pratica, out/ckpt.pt non si ricarica più, mancando la wpe nello state_dict quindi va riaddestrato, bisogna rilanciare e dobbiamo rieseguire i benchmark su cache e attention, bisogna rilanciare train.py e confrontare con 1.4485, non ci aspettiamo un grande miglioramento a block_size 256 su test char-level, la wpe appresa fa il suo lavoro benissimo, RoPe qui è un investimento per avere un contesto lungo e per la cache esatta, non è un trucco per diminuire la loss, se scende a ~1.43 già è tanto, se resta uguale ho 98k parametri in meno e un modello che estrapola oltre la finestra su cui è stato addestrato

Il vero salto lo faremo con l'implementazione di BPE, il vocabolario sarà ~1024 costruito su un corpus, non tiktoken, 50k CPT-2 su 1MB di shapespeare ci darebbero 19M di paraemtri di sola tabella di embedding, più del modello stesso, il corpus passa da 1M token a 350k, i nostri 256 token di contesto coprono 3-4 volte più testo, quando ci arriviamo la val loss per token non potremo più confrontarla con 1.,4485, andrà convertita in bit/byte che è l'unica metrica onesta fra tokenizzatori diversi

Tornando a rope e al problema che risolve
l'attention da sol anon sa cosa sia l'ordine, guardando il punteggio tra due token q_i*k_j, q_i e k_k dipendono solo dai rispettivi vettori di nput, se permutiamo le righe della sequenza i puntewggi seguono la permutaizone  el'output esce permutato ugualmetne, per un'attention pura "cane morde uomo" e "uomo morde cane" sono lo stesso insieme di token è un set non una sequenza

quindi la posizione va iniettata a mano, finora l'abbiamo fatto nel modo GPT-2, che è il modo ovvio

```
x = wte(token) + wpe(posizione)
```

una tabella di 256 vettori appresi, uno per slot, sommata al contenuto. Funziona, il nostro 1.4485 di val è la dimostrazione ma ha dei difetti
- occupa parametri, 98304 del nosro caso, ok è l'1% sulla scala del nostro modello ma è concreto
- mescola posizione e contenuto nello stesso vettore, il residual stream porta la somma e i layer successivi devono districarla, non è fatale ma è lavoro sprecato
- la finestra è cablata nell'architettura, wpe ha 256 righe, alla posizione 256 non c'è nessuna riga da leggere, i lmodello non può in linea di principio vedere un token più in là e ogni riga impara da sola, la 255 ha visto molti meno esempi della riga 3 perché le finestre di training partono da offset casuali, la psoizione 250 è addestrata peggio della 5
- non è traslabile, ch eè quello che ci frega con la KV cache, se butto via i token vecchi, quelli che restano si portano dietro la wpe della psoizione che avevano quando sono entrati, il token che era in posizione 100 continua a credersi in psoziione 100 anche quando la finestra è scivolata e lui è diventato il 10imo token

In particolar el'ultim,o punto non è un bug, è che al modello della posizione assoluta non importa nulla, quello che conta per capire una frase è quanto distano due parole, non a che offset assoluto del'opera omnia di shakespeare si trovano, stiamo dando al modello un'informazione di cui nonm ha bisogno e nascondendogli quella che serve, deve dedurre m-n confrontando due encoding assoluti attraverso le due matrici q e k e se la cava ma è un giro assurdo da apprendere


l'idea di RoPE

vogliamo una funzione f(x, m) che prende un vettore e la sua posizione tale che

```
< f(q, m), f(k, n) > = g(q, k, m - n)
```

il punteggio di attention deve dipendere adlle posizioni solo attraverso la loro differenza, se troviamo questa f la posizione assoluta sparisce dalla matematica dell'attention e ogni difetto elencato prima sparisce con lei

la f esiste ed è banale, UNA ROTAZIONE

prendi due sole dimensioni, un piano, sia R(alpha) la rotazione di angolo alpha, vale per ogni coppia di vettori ("'" = trasposta)

```
< R(m*theta)q, R(n*theta)k > = q' * R(m*theta)' R(n*theta) k = q' R((n-m)theta)k
```

perché le rotazioni sono ortogonali (R(alpha)' = R(-a)) e si compongono sommando gli angoli (R(-m*theta)R(n*theta) = R((n-m)theta)), il risultato dipende da n-m e basta, la posizione assoluta si cancella da se nel prodotto scalare, non perché l'abbiamo tolta ma perché è la stessa rotazione applicata ad entrambi

non sommiamo la posizione al contenuto, ruotiamo il contenuto di un angolo che dipende dalla posizione, la somma inquina il vettore, la rotazione ne preserva la norma e cambia solo l'orientamento, cioè esatamente la parte che il prodotto scalare misura


Da 2 dimensioni a 64

una head ha hs = 64 dimensioni, le spezziamo in 32 piani indipendenti e ogni piano ruiota ad una velocità diversa

```
theta_i = 100000^(-2i/hs) con i = 0 .. 31
```

quindi

```
coppia 0    theta=1.0       giro completo ogni ~6 posizioni
coppia 8    theta=0.1       un giro ogni ~63 psoizioni
coppia 16   theta=0.01      un giro ogni ~628 posizioni
coppia 31   theta=0.0001    quasi ferma su tutta la finestra
```

è un tachimetro con 32 lancette di velocità diversa in progressione geometrica, ma ci sono motivi ovvi per cui serve questa varietà
- se fossero tutte uguali e veloci, la funzione sarebbe periodica con periodo corto, distanza 3 e distanza 9 darebbero la stessa rotazione e sarebbero indistinguibili
- se fossero tutte lentissime, due posizioni vicine sarebbero qusi identiche, il modello non distinguerebbe il token precedente da quello cinque indietro
- con lo spettro geometrico, le lancette veloci risolvono le distanze corte con precisione, quelle lente danno il senso del lontano enza mai riavvolgersi, insieme ogni distanza della finestra ha una firma unica

come la somma di sinusoidi a frequenza diversa della serie di fourier

Il punteggio finale di attention diventa una somma di 32 coseni di (m-n)*theta_i pesati dal contenuto, il modello scegliendo i pesi di q e k può costruirsi il filtro sulla distanza che preferisce, "guarda 2 indietro", "guarda l'inizio del verso", "ignora tutto oltre 40", non sono io a programmarlo è lui che lo compone da quella base di frequenze

l'idea delle sinusoidi viene dal paper originale del 2017 con la differenza decisiva che lì venivano sommate, qui invece moltiplicate, ma solo moltiplicando la proprietà di traslazione vale esattamente, RoPE è di Su et al. 2021 oggi lo usano LLaMA, Mistral, Qwen, praticamente tutti

Per chiarezza scriviamo la matrice di rotazione per esteso

```
return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)

[ cos  -sin ] [ x1 ]     [ x1·cos − x2·sin ]
[ sin   cos ] [ x2 ]  =  [ x1·sin + x2·cos ]
```

la scelta di quali diensioni accoppiare è arbitraria, il paper originale accoppia le adiacenti (0, 1), (2, 3), (4, 5), per semplicità noi dividiamo l'intervallo in due accoppiando (0, 32), (1, 33), (2, 34), è la stessa cosa ma facciamo a meno di una permutazione fissa dei canali, tanto prima c'è una matrice appresa c_attn, quindi i lmodello impara la permutazione gratis se gli serve

ma cerchiamo la risposta e rispondiamo ad alcune domande che frullano nella mia testa

perché non si ruota v? perché la rotazione serve a modulare quanto due token si guardano e quello è il prodotto q*k, v è il contenuto che inviene viene trasportato una volta decis ochi guarda chi, ruotarlo signficia alterare l'informazione trasmessa in fuzione della posizione, cosa di cui vogliamo liberarci perche è un'informazione ch ei blocchi successivi dovrebber imparare ad eliminare

perché ogni layer e non una volta sola? perché ogni layer calcola le sue q e k fresche dal residual stream con la propria c_attn, non c'è un posto unico dove va messa, è un vantaggio, la posizione non si degrada strato dopo strato, ogni layer la riceve pulita, con wpe la iniettavam una volta ll'ingresso e speravamo sopravvivesse ai sei blocchi.

e oltre la finestra addestrata? in teoria RoPE non ha limiti, non è una tabella, t0 cresce quatno vogliamo, in pratica un modello addestrato a 256 peggiora a 2000 perché vede combinazioni di angoli mai incontrate, è un problema aperto e molto studiato, position interpolation, NTK-aware scaling, YaRN, sono tutte tecniche per allungare la finestra di contesto di un modello già addestrato rimappando le frequenze, ma non ci servono per quello che stiamo facendo è de un rabbit hole studiarle ora

Un bel regalino per la KV cache

questo è il pezzo che chiude il discorso di prima, la cache oltre block_size, talgia i k/v più vecchi. Con wpe quel taglio era finto, i sopravvissuti mantenevano le vecchie psoizioni, con RoPE le rotazioni nella cache restano assolute, t0 continua a crecere, e l'attention vede solo differenze, che nella finestra tagliata sono esattamente quelle giuste, sempre nell'intervalloi [0, block_size) su cui i lmodello è addestrato, il compromesso che avevamo dato per approssimazione diventa esatto e non abbiamo dovuto scrivere una riga per ottenerlo è puramente matematico

Sul training
| | val loss | ms/it | |
|---|---|---|---|
| wpe | 1.4485 | 49.9 | |
| RoPE | 1.4581 | 56.6 | -0.7% qualità +13% tempo |

RoPE qui ci costa, ma lo sapevamo infatti va un pochino peggio ed è più lento, perché ricalcolo cos/sin ogni layer a ogni passo invece di tenerli in u buffer, a block_size 256 su char-level una wpe appresa è semplicemente adeguata al problema

Quello che abbiamo guadagnato sono 98k parametri in meno, la cache esatta oltre la finestra, un modello che estrapol ainvece di sbattere contro una tabella di 256 righe, sono proprietà che contano quando il contsto è lungo, dopo che abbiamo implementato BPE, quando quei 256 oken copriranno 3-4 volte piuù testo e potremo alzare la finestra.

run di prova senza cache

```
uv run src/mygpt/generate.py --prompt "EDWARD" --tokens 160 --temperature 0.6 --seed 1 --no-cache 
# out/ckpt.pt: iter 1250, val loss 1.4581
# 160 token in 0.47s = 340 tok/s (no cache, batch 1)
============================================================
EDWARD IV:
Can your heart to be a tribune of them.

LADY GREY:
Bring thee, though they were not a man.

QUEEN MARGARET:
Ay, what she is the gentleman in the ground?


```

cache

```
uv run src/mygpt/generate.py --prompt "EDWARD" --tokens 160 --temperature 0.6 --seed 1 --samples 8
# out/ckpt.pt: iter 1250, val loss 1.4581
# 1280 token in 0.47s = 2747 tok/s (kv cache, batch 8)
============================================================
EDWARD IV:
Can your heart to be a tribune of them.

LADY GREY:
Bring thee, though they were not a man.

QUEEN MARGARET:
Ay, what she is the gentleman in the ground?


============================================================
EDWARD:
Then shall you say, that I may be shaped.

DUKE OF AUMERLE:
Why, sir, I am too still as I would be so,
And then I am a better.

DUKE OF AUMERLE:
What noise mu
============================================================
EDWARD IV:
They seem to bear the king of Hereford,
The senate of Marcius heads the Duke of York,
The loathsome house of York and York,
Brought her light and despised 
============================================================
EDWARD IV:
Now music of all my lordship lies and will.

KING HENRY VI:
We must be married; and many more hands
Than this same lands that he did supply her
Committed t
============================================================
EDWARD IV:
Why, this is not the company?

Second Gentleman:
What news we say?

Second Citizen:
Ay, noble letters and the which is no dear thing?

ROMEO:
It would not 
============================================================
EDWARD IV:
I do beseech your honour with you good villain.

GLOUCESTER:
Why, sir, we have met your lordship come to say.

KING EDWARD IV:
Ay, there is no more tongue 
============================================================
EDWARD IV:
Welcome, what comfort I am done to speak.

LADY ANNE:
Why then a man of man?

ROMEO:
At the duke? Why, when we'll have deceived grace.

DUKE OF AUMERLE:
My
============================================================
EDWARD IV:
Richard, as I am set by the truth of realm;
And then that I would be deposed to shame.

RICHARD:
So do your grace is cause to your leave and tears.

BENVOL
```

