import torch
import torch.nn as nn
from transformers import BertModel


class DPRModel(nn.Module):
    def __init__(self, Q_encoder, P_encoder, drop_out):
        super().__init__()
        self.Q_encoder = Q_encoder
        self.P_encoder = P_encoder
        self.dropout = nn.Dropout(drop_out)
    def forward(self, question_inputs, passage_inputs):
        question_embeddings = self.Q_encoder(**question_inputs).pooler_output
        question_embeddings = self.dropout(question_embeddings)
        passage_embeddings = self.P_encoder(**passage_inputs).pooler_output
        passage_embeddings = self.dropout(passage_embeddings)

        return question_embeddings, passage_embeddings


def build_dpr_model(Q_encoder_config, P_encoder_config, device, drop_out):
    Q_encoder = BertModel.from_pretrained(
        Q_encoder_config.pretrained_model_name).to(device)
    P_encoder = BertModel.from_pretrained(
        P_encoder_config.pretrained_model_name).to(device)

    dpr_model = DPRModel(Q_encoder=Q_encoder, P_encoder=P_encoder, drop_out=drop_out).to(device)
    return dpr_model


def load_dpr_model(model_path, Q_encoder_config, P_encoder_config, device, drop_out):
    dpr_model = build_dpr_model(Q_encoder_config, P_encoder_config, device, drop_out)
    dpr_model.load_state_dict(torch.load(model_path, map_location=device))
    dpr_model.to(device)
    return dpr_model
