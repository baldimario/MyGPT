import torch
import torch.nn.functional as F

torch.manual_seed(1337)
B, T, C = 4, 8, 2
x = torch.randn(B, T, C)

# il loop esplicito, lento e ovvio
xbow = torch.zeros(B, T, C)
for b in range(B):
    for t in range(T):
        xbow[b, t] = x[b, : t + 1].mean(dim=0)  # media di tutto il passato incluso t

# la stessa cosa come moltiplicazione di matrici
wei = torch.tril(torch.ones(T, T))  # triangolare inferiore di 1
wei = wei / wei.sum(dim=1, keepdim=True)  # ogni riga somma a 1
xbow2 = wei @ x  # (T,T) @ (B,T,C) -> (B,T,C)

# la stessa cosa passando da una softmax
tril = torch.tril(torch.ones(T, T))
wei3 = torch.zeros(T, T)  # "affinita'" tutte uguali
wei3 = wei3.masked_fill(tril == 0, float("-inf"))  # il futuro e' proibito
wei3 = F.softmax(wei3, dim=-1)  # -> stessa matrice di v2
xbow3 = wei3 @ x

assert torch.allclose(xbow, xbow2, atol=1e-6)
assert torch.allclose(xbow, xbow3, atol=1e-6)

torch.set_printoptions(precision=3, sci_mode=False)
print(wei)
