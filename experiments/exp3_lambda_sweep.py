import numpy as np
import matplotlib.pyplot as plt

# ====================================
# 1. Configuración general
# ====================================

faces = ["+x", "-x", "+y", "-y", "+z", "-z"]
nodes = [0, 1, 2, 3]
edges = [(0, 1), (1, 2), (2, 3)]

phi = {
    (0, 1): ("+x", "-x"),
    (1, 2): ("+x", "-x"),
    (2, 3): ("+x", "-x"),
}

beta = 0.35
mu = 1.0
T_steps = 100
seed = 7

lambda_values = [0.0, 0.05, 0.15, 0.30, 0.60]

# ====================================
# 2. Utilidades
# ====================================

def init_state(seed=7):
    rng = np.random.default_rng(seed)
    return {i: {f: rng.uniform(-1, 1) for f in faces} for i in nodes}

def face_neighbors(i, f):
    out = []
    for (a, b) in edges:
        fa, fb = phi[(a, b)]
        if a == i and fa == f:
            out.append((b, fb))
        if b == i and fb == f:
            out.append((a, fa))
    return out

def global_tension(X, lam):
    ext = 0.0
    for (i, j) in edges:
        fi, fj = phi[(i, j)]
        ext += (X[i][fi] - X[j][fj])**2

    internal = 0.0
    for i in nodes:
        for a in range(len(faces)):
            for b in range(a + 1, len(faces)):
                f1, f2 = faces[a], faces[b]
                internal += lam * (X[i][f1] - X[i][f2])**2

    return ext + mu * internal

def internal_raw_tension(X):
    total = 0.0
    for i in nodes:
        for a in range(len(faces)):
            for b in range(a + 1, len(faces)):
                f1, f2 = faces[a], faces[b]
                total += (X[i][f1] - X[i][f2])**2
    return total

def internal_dispersion(X):
    total = 0.0
    for i in nodes:
        vals = np.array([X[i][f] for f in faces])
        mean_val = vals.mean()
        total += np.mean((vals - mean_val)**2)
    return total / len(nodes)

def psi(i, f, X, lam):
    neigh = face_neighbors(i, f)
    num = 0.0
    den = 0.0

    for j, fj in neigh:
        num += X[j][fj]
        den += 1.0

    for g in faces:
        if g != f:
            num += mu * lam * X[i][g]
            den += mu * lam

    if den == 0:
        return X[i][f]

    return num / den

def one_step(X, lam):
    X_new = {i: dict(X[i]) for i in nodes}
    for i in nodes:
        for f in faces:
            p = psi(i, f, X_new, lam)
            X_new[i][f] = (1 - beta) * X_new[i][f] + beta * p
    return X_new

def flatten_state(X):
    vals = []
    for i in nodes:
        for f in faces:
            vals.append(X[i][f])
    return np.array(vals)

def run_experiment(lam, seed=7):
    X = init_state(seed)
    T_vals = []
    step_vals = []
    internal_vals = []
    dispersion_vals = []

    for _ in range(T_steps):
        T_vals.append(global_tension(X, lam))
        internal_vals.append(internal_raw_tension(X))
        dispersion_vals.append(internal_dispersion(X))

        X_next = one_step(X, lam)
        step_vals.append(np.linalg.norm(flatten_state(X_next) - flatten_state(X)))
        X = X_next

    T_vals.append(global_tension(X, lam))
    internal_vals.append(internal_raw_tension(X))
    dispersion_vals.append(internal_dispersion(X))

    return {
        "T": np.array(T_vals),
        "step": np.array(step_vals),
        "internal": np.array(internal_vals),
        "dispersion": np.array(dispersion_vals),
    }

# ====================================
# 3. Ejecutar barrido en lambda
# ====================================

results = {}
for lam in lambda_values:
    results[lam] = run_experiment(lam, seed=seed)

# ====================================
# 4. Gráficos comparativos
# ====================================

plt.figure(figsize=(8, 4))
for lam in lambda_values:
    plt.plot(results[lam]["T"], label=f"λ = {lam}")
plt.xlabel("Iteración t")
plt.ylabel("Tensión global T(X(t))")
plt.title("Experimento 3 — Disipación global para distintos λ")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
for lam in lambda_values:
    plt.plot(results[lam]["dispersion"], label=f"λ = {lam}")
plt.xlabel("Iteración t")
plt.ylabel("Dispersión interna promedio")
plt.title("Experimento 3 — Coherencia interna vs λ")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
for lam in lambda_values:
    plt.plot(results[lam]["step"], label=f"λ = {lam}")
plt.xlabel("Iteración t")
plt.ylabel("||X(t+1)-X(t)||")
plt.title("Experimento 3 — Tamaño del paso dinámico")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ====================================
# 5. Resumen final numérico
# ====================================

print("Resumen final:")
for lam in lambda_values:
    print(f"λ={lam:>4}: "
          f"T_final={results[lam]['T'][-1]:.6f}, "
          f"Disp_final={results[lam]['dispersion'][-1]:.6f}, "
          f"Step_final={results[lam]['step'][-1]:.6f}")