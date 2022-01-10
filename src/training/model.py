import torch.nn as nn
from transformers import AutoModel


class JigsawModel(nn.Module):
    """
        A wrapper around the model being used
    """

    def __init__(self, model_name, num_classes):

        # Create the model
        super(JigsawModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)

        # Add a dropout layer
        self.drop = nn.Dropout(p=0.2)

        # Add a linear output layer
        self.fc = nn.Linear(768, num_classes)

    def forward(self, ids, mask):
        """
            Perform a forward feed
            :param ids: The input
            :param mask: The attention mask
        """
        out = self.model(input_ids=ids, attention_mask=mask, output_hidden_states=False)
        out = self.drop(out[1])
        outputs = self.fc(out)
        return outputs
