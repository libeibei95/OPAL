params = {
    'Caser': {
        'nh': 16,
        'nv': 8,
        'reg_weight': 1e-4,
    },
    'NextItNet': {
        'kernel_size': 3,
        'block_num': 5,
        'dilations': [1, 4],
        'reg_weight': 1e-5,
    },
    'BERT4Rec': {
        'n_layers': 2,
        'n_heads': 2,
        'hidden_size': 64,
        'inner_size': 256,
        'hidden_dropout_prob': 0.5,
        'attn_dropout_prob': 0.5,
        'hidden_act': 'gelu',
        'layer_norm_eps': 1e-12,
        'initializer_range': 0.02,
        'mask_ratio': 0.2
    }
}

