import torch.nn as nn
from transformers import AutoModel


class JigsawModel(nn.Module):
    def __init__(self, model_name, config):
        super(JigsawModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.2)
        self.fc = nn.Linear(768, config["num_classes"])

    def forward(self, ids, mask):
        out = self.model(input_ids=ids, attention_mask=mask, output_hidden_states=False)
        out = self.drop(out[1])
        outputs = self.fc(out)
        return outputs
