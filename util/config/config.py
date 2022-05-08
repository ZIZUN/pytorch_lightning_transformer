from sacred import Experiment

EXPERIMENT_NAME = 'Intent_CLS'

ex = Experiment(EXPERIMENT_NAME, save_git_info=False)

@ex.config
def config():
    exp_name = EXPERIMENT_NAME
    
    seed = 1841
    
    # BERT hyper-parameter setting
    vocab_size = 50265
    bert_hidden_size = 768
    multi_head_num = 12
    qkv_hidden_size = 64
    bert_layer_num = 12  
    position_encoding_maxlen = 512  
    
    # GPU, CPU Environment Setting
    num_nodes = 1
    gpus = [0,1,2,3]
    batch_size = 32
    per_gpu_batch_size = 8    # Note that -> batch_size % (per_gpu_batch_size * len(gpus) == 0
    num_workers = 5
 
    # Main Setting   
    input_seq_len = 50
    max_steps = 5000
    warmup_steps = 500
    val_check_interval = 0.5
    model_name = 'roberta-base'
    
    # Path Setting
    load_path = ''  # 
    log_dir = 'result'
    train_dataset_path = 'data/HWU64/train'
    val_dataset_path = 'data/HWU64/valid'
    test_dataset_path = 'data/HWU64/test'

# text encoder
@ex.named_config
def text_roberta_base():
    model_name = "roberta-base"
    vocab_size = 50265
    bert_hidden_size = 768
    