

def get_optimizer_with_weight_decay(model,
                                    optimizer,
                                    learning_rate,
                                    weight_decay):
    """

    Args:
        model:
        optimizer:
        learning_rate:
        weight_decay:

    Returns:

    """
    no_decay = ["bias", "LayerNorm.weight"]
    params = [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
    params_nd = [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]
    optimizer_grouped_parameters = [{"params": params, "weight_decay": weight_decay},
                                    {"params": params_nd, "weight_decay": 0.0}]

    return optimizer(optimizer_grouped_parameters, lr=learning_rate)