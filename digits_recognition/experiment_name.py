import hashlib

def get_experiment_name(model, args):
    model_id = hashlib.sha256(str(model).encode()).hexdigest()[:8]
    args_id = hashlib.sha256(str(args).encode()).hexdigest()[:8]

    return f"{model._get_name()} (arch: {model_id}; args: {args_id})"