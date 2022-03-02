from src.core.checkpointers import CustomCheckpointer

def build(save_dir, logger, model, optimizer, scheduler, eval_standard):
    checkpointer = CustomCheckpointer(
        save_dir, logger, model, optimizer, scheduler, eval_standard)
    return checkpointer

