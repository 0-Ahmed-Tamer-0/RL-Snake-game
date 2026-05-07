import os
import torch
import torch.nn as nn

class QNetwork(nn.Module):


    def __init__(
        self,
        input_size: int = 11,
        output_size: int = 3,
        use_deep: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.use_deep = use_deep
        self.dropout_prob = dropout

        if use_deep:
            self.network = self._build_deep_network(input_size, output_size, dropout)
        else:
            self.network = self._build_base_network(input_size, output_size, dropout)

    @staticmethod
    def _build_base_network(
        input_size: int, output_size: int, dropout: float
    ) -> nn.Sequential:

        layers = [
            nn.Linear(input_size, 256),
            nn.ReLU(),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))

        layers += [
            nn.Linear(256, 128),
            nn.ReLU(),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))

        layers.append(nn.Linear(128, output_size))

        return nn.Sequential(*layers)

    @staticmethod
    def _build_deep_network(
        input_size: int, output_size: int, dropout: float
    ) -> nn.Sequential:

        layers = [
            nn.Linear(input_size, 512),
            nn.ReLU(),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))

        layers += [
            nn.Linear(512, 256),
            nn.ReLU(),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))

        layers += [
            nn.Linear(256, 128),
            nn.ReLU(),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))

        layers.append(nn.Linear(128, output_size))

        return nn.Sequential(*layers)



    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self._prepare_input(x)
        return self.network(x)


    def predict(self, state) -> torch.Tensor:

        with torch.no_grad():
            tensor = self._to_tensor(state)           
            q_values = self.network(tensor)            
            return q_values.squeeze(0)                 



    def save(self, path: str) -> None:

        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": {
                    "input_size": self.input_size,
                    "output_size": self.output_size,
                    "use_deep": self.use_deep,
                    "dropout": self.dropout_prob,
                },
            },
            path,
        )

    def load(self, path: str) -> None:

        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location="cpu")
        self.load_state_dict(checkpoint["state_dict"])



    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        else:
            x = x.float()

        if x.dim() == 1:
            x = x.unsqueeze(0)

        return x

    @staticmethod
    def _to_tensor(state) -> torch.Tensor:

        if isinstance(state, torch.Tensor):
            tensor = state.float()
        else:
            tensor = torch.tensor(state, dtype=torch.float32)

        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)

        return tensor



    def __repr__(self) -> str:
        mode = "deep" if self.use_deep else "base"
        return (
            f"QNetwork(input={self.input_size}, output={self.output_size}, "
            f"mode={mode}, dropout={self.dropout_prob})\n{self.network}"
        )




def build_qnetwork(
    input_size: int = 11,
    output_size: int = 3,
    use_deep: bool = False,
    dropout: float = 0.0,
    device: str = "cpu",
) -> QNetwork:

    model = QNetwork(
        input_size=input_size,
        output_size=output_size,
        use_deep=use_deep,
        dropout=dropout,
    )
    return model.to(device)

if __name__ == "__main__":
    # Base architecture
    base_model = build_qnetwork()
    print("=== Base Model ===")
    print(base_model)

    # Deep architecture with dropout
    deep_model = build_qnetwork(use_deep=True, dropout=0.2)
    print("\n=== Deep Model ===")
    print(deep_model)

    # Single-state inference
    sample_state = [1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0]
    q_vals = deep_model.predict(sample_state)
    print(f"\nSingle-state Q-values : {q_vals}")
    print(f"Selected action       : {q_vals.argmax().item()}")

    # Batch forward pass
    batch = torch.zeros(32, 11)
    batch_q = deep_model(batch)
    print(f"Batch Q-values shape  : {batch_q.shape}")   # (32, 3)