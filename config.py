class Config:
    # Model parameters
    image_size = 224
    patch_size = 16
    hidden_dim = 768
    num_heads = 12
    num_layers = 12
    text_max_length = 77
    
    # GAS parameters
    top_k = 45
    
    # CCL parameters
    lambda_i = 2.0
    lambda_t = 2.0
    temperature = 0.07
    queue_size = 57600
    momentum = 0.995
    
    # Training parameters
    batch_size = 4
    num_epochs = 30
    learning_rate = 1e-4
    weight_decay = 0.05
    
    # Loss weights
    weight_itm = 1.0
    weight_itc = 1.0
    weight_lm = 1.0
    weight_icc = 1.0
    weight_tcc = 1.0
    
    # Dataset paths
    dataset_name = "CUHK-PEDES"
    data_root = "data/CUHK_PEDES_images"

config = Config()
