setup.py: Defines a Setup class for model/tokenizer initialization, data loading, formatting, and tokenization.

train.py: Defines a ModelTrainer class encapsulating the HuggingFace Trainer logic and GPU‐stat reporting.

run.py: A simple entry point that ties together Setup and ModelTrainer with your chosen hyperparameters/PEFT settings.

utils.py: Contains formatting_prompts_func and tokenize_function as standalone utilities.

