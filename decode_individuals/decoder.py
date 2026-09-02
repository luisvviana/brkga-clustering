import numpy as np

def gen_pop(X, PSIZE): # Gera população inicial
    n, d = X.shape
    p = []
    for _ in range(PSIZE):
        A = []
        k = int(np.random.uniform(2, n/2))
        A.append(k)
        for __ in range(d):
            A.append(np.random.binomial(n=1, p=0.5))
        p.append(A)

    return p

def encoder(pop, kmax):
    new_p = []
    for ind in pop:
        new_ind = []
        k = ind[0]
        k = (k-2)/max(1, kmax - 3) # kmax - 2 - 1
        new_ind.append(k)
        for i in range(1, len(ind)):
            if ind[i] == 0:
                rand = np.random.uniform(0, 0.49999)
            else:
                rand = np.random.uniform(0.5, 1)
            new_ind.append(rand)
        new_p.append(new_ind)
            
    return new_p

def decode_2(X, chromosome, metric="sil", lambda_k=0.0):
    n_samples, n_features = X.shape

    percentage = chromosome[1]
    percentage = max(0.0, min(1.0, percentage))

    selected_vars = int(round(n_features * percentage))
    selected_vars = min(selected_vars, n_features)

    ranking = [(chromosome[i + 2], i) for i in range(n_features)]
    ranking.sort(key=lambda x: x[0])

    selected_cols = [ranking[i][1] for i in range(selected_vars)]

    X_sel = X[:, selected_cols]

    unique_samples = np.unique(X_sel, axis=0).shape[0]

    k_real = chromosome[0]
    k_real = max(0.0, min(1.0, k_real))

    kmax = unique_samples // 2

    k = 2 + int(round((kmax - 2) * k_real))
    k = min(k, kmax)

    flags = []
    for i in range(n_features):
        if i in selected_cols:
            flags.append(1)
        else:
            flags.append(0)

    return k, flags