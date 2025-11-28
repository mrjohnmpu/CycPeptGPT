from pydantic import BaseModel


class ArchitectureConfig(BaseModel):
    #Setup
    name: str
    training_data_path: str
    validation_data_path: str
    save_directory: str
    batch_size: int = 16
    num_epoch: int = 20
    starting_epoch: int = 1
    padding_value: int = 0
    max_sequence_length: int = 500
    #Model architecture
    N: int = 6
    H: int = 8
    d_model: int = 256
    d_ff: int = 2048
    #Regularization
    dropout: float = 0.1
    label_smoothing : float = 0.0
    #Optimization
    factor: float = 1.0
    warmup_steps: int = 4000
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_eps: float = 1e-9
    lr: float = 5.0
    eval_batch_size: int = 10
    bptt: int = 35
    embedding_size: int = 200  # embedding dimension, emsize
    dimension_hid: int = 200  # dimension of the feedforward network model in nn.TransformerEncoder, d_hid
    n_layers: int = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    n_head: int = 2  # number of heads in nn.MultiheadAttention
    best_val_loss: float = float('inf')
    shuffle_each_epoch: bool = True
    drop_last_batch: bool = True
    use_cuda: bool = True
    run_type: str = 'training'
