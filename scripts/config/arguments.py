from typing import Optional
from dataclasses import dataclass, field

@dataclass
class Arguments:
    ### Hyper parameters
    lr: float = field(
        default=0.0001, metadata={"help": "learning rate"}
    )
    batch_size: Optional[int] = field(
        default=256, metadata={"help": "batch size"}
    )
    n_epochs: Optional[int] = field(
        default=100, metadata={"help": "training epochs"}
    )
    num_classes: Optional[int] = field(
        default=10, metadata={"help": "classes for mnist is 10"}
    )
    ### Data parameter
    data_root: Optional[str] = field(
        default="./cache/", metadata={"help": "data root"}
    )
    data_split_dim: Optional[int] = field(
        default=7, metadata={"help": "split mnist pictures to sub blocks"}
    )
    data_dimension: Optional[int] = field(
        default=8, metadata={"help": "data dimensions, for our mnist is 8"}
    )
    ### Model parameters
    n_heads: Optional[int] = field(
        default=4, metadata={"help": "multi head"}
    )
    model_type: Optional[str] = field(
        default='transformer', metadata={"help": "model type choices: transformer or linear"}
    )
    model_save_root: Optional[str] = field(
        default="./cache/checkpoint.pth.tar", metadata={"help": "model save root"}
    )
    gama_scale: float = field(
        default=0.001, metadata={"help": "rate to scale noise weight"}
    )