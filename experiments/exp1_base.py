import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. Estructura del modelo
# =========================

faces = ["+x", "-x", "+y", "-y", "+z", "-z"]
n_faces = len(faces)

# Grafo simple: cadena 0-1-2-3
nodes = [0, 1, 2, 3]
edges = [(0, 1), (1, 2), (2, 3)]

# Asignación componente-componente por arista
# Simplificación: todas las aristas conectan +x de i con -x de j
phi = {
    (0, 1): ("+x", "-x"),
    (1, 2): ("+x", "-x"),
    (2, 3): ("+x", "-x"),
}

# Parámetros
beta = 0.35
mu = 1.0
lam = 0.15
T_steps = 60

# =========================
# 2. Estado inicial
# =========================

rng = np.random.default_rng(7)

# X[node][face] = valor escalar
X = {i: {f: rng.uniform(-1, 1) for f in faces} for i in nodes}

# =========================
# 3. Utilidades
# =========================

def face_neighbors(i, f):
    """Devuelve vecinos j tales que el componente f de i interactúa con j."""
    out = []
    for (a, b) in edges:
        if (a, b) in phi:
            fa, fb = phi[(a, b)]
            if a == i and fa == f:
                out.append((b, fb))
        if (a, b) in phi:
            fa, fb = phi[(a, b)]
            # interacción inversa
            if b == i and fb == f:
                out.append((a, fa))
    return out

def global_tension(X):
    ext = 0.0
    for (i, j) in edges:
        fi, fj = phi[(i, j)]
        ext += (X[i][fi] - X[j][fj])**2

    internal = 0.0
    for i in nodes:
        for a in range(n_faces):
            for b in range(a + 1, n_faces):
                f1, f2 = faces[a], faces[b]
                internal += lam * (X[i][f1] - X[i][f2])**2

    return ext + mu * internal

def psi(i, f, X):
    neigh = face_neighbors(i, f)

    num = 0.0
    den = 0.0

    # parte externa
    for j, fj in neigh:
        num += X[j][fj]
        den += 1.0

    # parte interna
    for g in faces:
        if g != f:
            num += mu * lam * X[i][g]
            den += mu * lam

    # si no hubiera términos, devolver estado actual
    if den == 0:
        return X[i][f]

    return num / den

def one_step(X):
    """Actualización secuencial"""
    X_new = {i: dict(X[i]) for i in nodes}

    for i in nodes:
        for f in faces:
            p = psi(i, f, X_new)
            X_new[i][f] = (1 - beta) * X_new[i][f] + beta * p

    return X_new

def flatten_state(X):
    vals = []
    for i in nodes:
        for f in faces:
            vals.append(X[i][f])
    return np.array(vals)

# =========================
# 4. Simulación
# =========================

tension_values = []
step_sizes = []

for t in range(T_steps):
    tension_values.append(global_tension(X))
    X_next = one_step(X)
    step_sizes.append(np.linalg.norm(flatten_state(X_next) - flatten_state(X)))
    X = X_next

tension_values.append(global_tension(X))

# =========================
# 5. Gráficos
# =========================

plt.figure(figsize=(8, 4))
plt.plot(tension_values, linewidth=2)
plt.xlabel("Iteración t")
plt.ylabel("Tensión global T(X(t))")
plt.title("Disipación de tensión en VEHICLE base")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(step_sizes, linewidth=2)
plt.xlabel("Iteración t")
plt.ylabel("||X(t+1)-X(t)||")
plt.title("Tamaño del paso dinámico")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()