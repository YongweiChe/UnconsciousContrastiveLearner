import torch

from ucl import direct_scores, mc_scores, recall_at_k


@torch.no_grad()
def evaluate(model, val, bridge_pool, n_candidates=32):
    """Recall@1 for retrieving c from a among groups of n_candidates held-out triplets.

    val:         (A, B, C) tensors of held-out triplets
    bridge_pool: B samples drawn independently of val, used for the Monte Carlo estimate
    Returns {"direct": float, "mc": float or None}.
    """
    model.eval()
    a, _, c = val
    phi_a = model.embed("A", a)
    phi_c = model.embed("C", c)
    if model.bridge is not None:
        (left_a, left_b), (right_b, right_c) = model.pairs
        bridge_ab = model.embed(model.bridge[0], bridge_pool)
        bridge_bc = model.embed(model.bridge[1], bridge_pool)
        scale_ab, scale_bc = model.scale(left_a, left_b), model.scale(right_b, right_c)

    direct, mc = [], []
    for start in range(0, len(a) - n_candidates + 1, n_candidates):
        sl = slice(start, start + n_candidates)
        direct.append(recall_at_k(direct_scores(phi_a[sl], phi_c[sl], model.critic), 1))
        if model.bridge is not None:
            scores = mc_scores(
                phi_a[sl], phi_c[sl], bridge_ab, bridge_bc, model.critic, model.critic, scale_ab, scale_bc
            )
            mc.append(recall_at_k(scores, 1))
    mean = lambda xs: sum(xs) / len(xs)
    return {"direct": mean(direct), "mc": mean(mc) if mc else None}


def train(
    model, train_data, val, bridge_pool, *, pair_size, epochs, batch_size, lr, lr_step, lr_gamma, norm_penalty, generator
):
    """Trains model on its pairs and evaluates after every epoch (index 0 = before training).

    train_data: (A, B, C) tensors with at least pair_size * len(model.pairs) triplets.
    Pair i trains on rows [i * pair_size, (i + 1) * pair_size), so e.g. the A-B and B-C
    models see independent samples and neither ever sees a full (A, B, C) triplet.
    """
    n = pair_size
    index = {"A": 0, "B": 1, "B1": 1, "B2": 1, "C": 2}
    tasks = [
        (left, right, train_data[index[left]][i * n : (i + 1) * n], train_data[index[right]][i * n : (i + 1) * n])
        for i, (left, right) in enumerate(model.pairs)
    ]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)

    history = {"direct": [], "mc": [], "loss": []}

    def log_eval():
        result = evaluate(model, val, bridge_pool)
        history["direct"].append(result["direct"])
        history["mc"].append(result["mc"])

    log_eval()
    for _ in range(epochs):
        model.train()
        perms = [torch.randperm(n, generator=generator) for _ in tasks]
        total = 0.0
        n_steps = n // batch_size
        for step in range(n_steps):
            optimizer.zero_grad()
            loss = 0.0
            for (left, right, x_l, x_r), perm in zip(tasks, perms):
                idx = perm[step * batch_size : (step + 1) * batch_size]
                loss = loss + model.pair_loss(left, right, x_l[idx], x_r[idx], norm_penalty)
            loss = loss / len(tasks)
            loss.backward()
            optimizer.step()
            total += loss.item()
        scheduler.step()
        history["loss"].append(total / n_steps)
        log_eval()
    return history
