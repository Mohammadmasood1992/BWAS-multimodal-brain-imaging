"""Small inspection utilities."""

import torch


def count_parameters_by_layer(model, input_shapes=None, covariate_dim=None, device='cuda'):
    """Prints a per-layer parameter count and the model's total trainable parameters.

    If `input_shapes` is given, runs one dummy forward pass first so a sanity
    check of the model's wiring happens before real (expensive) training starts.
    """
    if input_shapes is not None:
        model = model.to(device)
        dummy_inputs = [torch.randn(1, *shape).to(device) for shape in input_shapes]
        dummy_covariates = torch.randn(1, covariate_dim).to(device) if covariate_dim else None

        with torch.no_grad():
            output = model(dummy_inputs, covariates=dummy_covariates)
        print(f"Forward pass check succeeded. Output shape(s): "
              f"{[o.shape for o in output] if isinstance(output, list) else output.shape}\n")

    print(f"{'Layer':<50}{'Trainable Params':>20}{'Total Params':>20}")
    print("-" * 90)

    total_trainable, total_params = 0, 0
    for name, module in model.named_modules():
        if len(list(module.children())) > 0:
            continue  # only leaf modules hold parameters directly
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        total = sum(p.numel() for p in module.parameters())
        if total > 0:
            print(f"{name:<50}{trainable:>20,}{total:>20,}")
            total_trainable += trainable
            total_params += total

    print("-" * 90)
    print(f"{'TOTAL':<50}{total_trainable:>20,}{total_params:>20,}")
    return total_trainable, total_params
