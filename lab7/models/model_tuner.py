from models.model_lstm import build_model_hp as build_lstm_hp


def build_model_hp(hp, model_type="lstm"):
    """Wrapper dla Keras Tuner"""
    if model_type == "lstm":
        return build_lstm_hp(hp)
    raise ValueError(f"Nieznany typ modelu: {model_type}. Użyj 'lstm'.")
