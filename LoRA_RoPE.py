from multitask_classifier import *

class DotDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

TQDM_DISABLE = False

args_dict = {
    "sst_train": "data/ids-sst-train.csv",
    "sst_dev": "data/ids-sst-dev.csv",
    "sst_test": "data/ids-sst-test-student.csv",

    "para_train": "data/quora-train.csv",
    "para_dev": "data/quora-dev.csv",
    "para_test": "data/quora-test-student.csv",

    "sts_train": "data/sts-train.csv",
    "sts_dev": "data/sts-dev.csv",
    "sts_test": "data/sts-test-student.csv",

    "seed": 11711,
    "epochs": 10,
    "option": "finetune",
    "use_gpu": True,  # Default is False, set to True if required

    "sst_dev_out": "predictions/sst-dev-output-LoRA-RoPE.csv",
    "sst_test_out": "predictions/sst-test-output-LoRA-RoPE.csv",

    "para_dev_out": "predictions/para-dev-output-LoRA-RoPE.csv",
    "para_test_out": "predictions/para-test-output-LoRA-RoPE.csv",

    "sts_dev_out": "predictions/sts-dev-output-LoRA-RoPE.csv",
    "sts_test_out": "predictions/sts-test-output-LoRA-RoPE.csv",

    # hyperparameters
    "batch_size": 32,  # 64 for sst, 8 for cfimdb can fit a 12GB GPU
    "hidden_dropout_prob": 0.3,
    "lr": 1e-5  # 1e-3 for pretrain, 1e-5 for fine-tuning
}

args = DotDict(args_dict)

# args = get_args()
args.filepath = f'{args.option}-{args.epochs}-{args.lr}-multitask-LoRA-RoPE.pt' # save path
seed_everything(args.seed)  # fix the seed for reproducibility

save_metrics = {
    "batch_size": args.batch_size,
    "lr": args.lr,
    "hidden_dropout_prob": args.hidden_dropout_prob,
    "option": args.option,
    "epoch": [],
    "train_loss": [],
    "train_sentiment_acc": [],
    "train_paraphrase_acc": [],
    "train_sts_corr": [],
    "train_avg_normalized_score": [],
    "test_sentiment_accuracy": [], 
    "test_paraphrase_accuracy": [],
    "test_sts_corr": []
}

train_multitask(args, save_metrics, model_name='RoPE')
test_model(args, save_metrics, model_name = 'RoPE')

MULTITASK_RoPE = "./stats/multitask-LoRA-RoPE_saved_metrics.json"

# Save save_metrics to a JSON file
with open(MULTITASK_RoPE, 'w') as f:
    json.dump(save_metrics, f, indent=4)

print(f'Metrics saved to {MULTITASK_RoPE}')

from utils import *

MULTITASK_RoPE = "./stats/multitask-LoRA-RoPE_saved_metrics.json"
test_sentiment_accuracy, test_paraphrase_accuracy, test_sts_corr = visualize_multitask(MULTITASK_RoPE)

print(f"test sentiment accuracy: {test_sentiment_accuracy}")
print(f"test paraphrase accuracy: {test_paraphrase_accuracy}")
print(f"semantic similarity regression accuracy: {test_sts_corr}")