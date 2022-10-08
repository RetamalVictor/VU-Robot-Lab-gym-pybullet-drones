from turtle import forward
import torch
import torch.nn as nn
import numpy as np
import os


class BasicNet(nn.Module):
    def __init__(self):
        super(BasicNet, self).__init__()
        # Define Layers:
        self.l1 = nn.Linear(3, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3_magnitude = nn.Linear(64, 1)
        self.l4_position = nn.Linear(64, 1)

        # Define Activation functions:
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, state):

        state_tensor = torch.Tensor(state, dtype=torch.float32)
        x = self.l1(state_tensor)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)

        magnitude = self.l3_magnitude(x)
        magnitude = self.sigmoid(magnitude)
        position = self.l4_position(x)
        position = self.sigmoid(position)
        return magnitude

    def save(self, output_path):
        torch.save(
            {
                "model_state_dict": self.state_dict(),
            },
            os.path.join(output_path, "agent_model.pt"),
        )
        print(f"Saved agent_model.pt")
