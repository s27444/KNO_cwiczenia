from models.model_cnn import build_model_hp as build_cnn_hp
from models.model_fc import build_model_hp as build_fc_hp


def build_model_hp(hp, model_type="fc"):
    """
    Funkcja wrapper dla Keras Tuner
    model_type: 'fc' dla fully connected, 'cnn' dla convolutional
    """
    if model_type == "fc":
        return build_fc_hp(hp)
    elif model_type == "cnn":
        return build_cnn_hp(hp)
    else:
        raise ValueError(f"Nieznany typ modelu: {model_type}. Użyj 'fc' lub 'cnn'.")
