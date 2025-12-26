def DSM(self, features, target, temperature: float = 0.07, beta: float = 0.1, eta: float = 0.2):

    features = F.normalize(features, dim=1)
    logits = torch.matmul(features, features.T) / temperature
    logits_max, _ = logits.max(dim=1, keepdim=True)
    logits = logits - logits_max.detach()
    labels = target.view(-1, 1)
    dro_weights = torch.exp(
        -0.5 * ((logits - eta) / eta) ** 2
    )
    dro_weights = dro_weights / dro_weights.sum(
        dim=1, keepdim=True
    ).clamp(min=1e-8)
    exp_logits = torch.exp(logits).sum(
        dim=1, keepdim=True
    ).clamp(min=1e-8)

    log_prob = logits - torch.log(exp_logits)
    weighted_log_prob = dro_weights * log_prob


    contrastive_loss = -(
        positive_mask * weighted_log_prob
    ).sum(dim=1) / positive_mask.sum(
        dim=1
    ).clamp(min=1e-8)
    feature_cov = torch.cov(features.T) + \
        1e-6 * torch.eye(
            features.shape[1],
            device=features.device
        )

    ib_penalty = torch.trace(feature_cov)

    return contrastive_loss.mean() - beta * ib_penalty
