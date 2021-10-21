from typing import Optional
from dataclasses import dataclass, field

@dataclass
class Arguments:
    lr: float = field(
        default=0.0001, metadata={"help": "learning rate"}
    )
    batch_size: Optional[int] = field(
        default=64, metadata={"help": "batch size"}
    )
    n_epochs: Optional[int] = field(
        default=100, metadata={"help": "training epochs"}
    )
    data_split_dim: Optional[int] = field(
        default=2, metadata={"help": "split mnist pictures to sub blocks"}
    )
    data_dimension: Optional[int] = field(
        default=8, metadata={"help": "data dimensions, for our mnist is 8"}
    )
    n_heads: Optional[int] = field(
        default=4, metadata={"help": "multi head"}
    )
    num_classes: Optional[int] = field(
        default=10, metadata={"help": "classes for mnist is 10"}
    )