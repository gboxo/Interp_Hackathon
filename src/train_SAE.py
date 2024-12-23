import os
import re
import torch
import datasets
from sae_lens import LanguageModelSAERunnerConfig, SAETrainingRunner
from transformer_lens import HookedTransformer

# Set device based on availability
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print("Using device:", device)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load the model
model = HookedTransformer.from_pretrained("gemma-2-2b-it")

# Load the dataset
dataset = datasets.load_dataset("gboxo/control_dataset_gemma2_tokenized")

# Argument key definitions for start and end
start_argument_keys = ["**Argument**", "**Argument:**", "## Argument", "##  Argument"]
end_argument_keys = ["**End Argument**", "**End Argument:**", "## End Argument", "\nEnd Argument"]

# Function to create masks based on argument positions
def create_masks(dataset, model, start_keys, end_keys, max_examples=2000):
    masks = []
    for i in range(max_examples):
        example = dataset["train"][i]
        prompt = example["prompt"]
        input_ids = example["tokens"]

        # Find the start and end positions
        start_positions = [m.start() for key in start_keys for m in re.finditer(re.escape(key), prompt)]
        end_positions = [m.start() for key in end_keys for m in re.finditer(re.escape(key), prompt)]
        
        # Sort and filter positions
        start_positions.sort()
        end_positions.sort()

        # Log potential issues with positions
        if len(start_positions) == 0:
            print(f"No start positions found in example {i}")
            continue
        if len(end_positions) == 0:
            print(f"No end positions found in example {i}")
            continue
        if len(start_positions) > 1:
            print(f"Multiple start positions found in example {i}, using first.")
            start_positions = start_positions[1:]  # Skip subsequent starts
        if len(end_positions) > 1:
            print(f"Multiple end positions found in example {i}, using first.")
            end_positions = end_positions[1:]  # Skip subsequent ends
        
        # Extract the argument substring
        argument = prompt[start_positions[0]:end_positions[0]]

        # Tokenize the argument (without adding BOS)
        tokenized_argument = model.to_tokens(argument, prepend_bos=False).squeeze().tolist()

        # Attempt to create a mask around the tokenized argument
        success = False
        l, r = 0, len(tokenized_argument)
        p = 0
        while not success and p <= 10:
            # Adjust token boundaries to find a match
            new_tokenized_argument = tokenized_argument[l:r]
            mask = torch.zeros(len(input_ids), dtype=torch.float)
            
            # Search for the tokenized argument in input_ids
            for j in range(len(input_ids) - len(new_tokenized_argument) + 1):
                if input_ids[j:j + len(new_tokenized_argument)] == new_tokenized_argument:
                    mask[j:j + len(new_tokenized_argument)] = 1
                    success = True
                    break
            
            # Adjust the boundaries (l and r) based on the iteration
            if not success:
                if p % 2 == 0:
                    l += 1  # Move start boundary
                else:
                    r -= 1  # Move end boundary
            p += 1

        if not success:
            print(f"Could not find a match for example {i}.")
        else:
            masks.append(mask)

    return masks

# Execute the mask creation
masks = create_masks(dataset, model, start_argument_keys, end_argument_keys)







# ===========================






total_training_steps = 500
batch_size = 4096
total_training_tokens = total_training_steps * batch_size
lr_warm_up_steps = 100
lr_decay_steps = 400
l1_warm_up_steps = 100

cfg = LanguageModelSAERunnerConfig(
    # Data Generating Function (Model + Training Distibuion)
    from_pretrained_path="",
    model_name="gemma-2-2b-it", 
    hook_name="blocks.10.hook_res_post",    hook_layer=0,  # Only one layer in the model.
    d_in=2304,  # the width of the mlp output.
    dataset_path="",  # this is a tokenized language dataset on Huggingface for the Tiny Stories corpus.
    is_dataset_tokenized=True,
    streaming=True,  # we could pre-download the token dataset if it was small.
    # SAE Parameters
    mse_loss_normalization=None,  # We won't normalize the mse loss,
    expansion_factor=7,  # the width of the SAE. Larger will result in better stats but slower training.
    b_dec_init_method="zeros",  # The geometric median can be used to initialize the decoder weights.
    apply_b_dec_to_input=False,  # We won't apply the decoder weights to the input.
    normalize_sae_decoder=False,
    scale_sparsity_penalty_by_decoder_norm=True,
    decoder_heuristic_init=True,
    init_encoder_as_decoder_transpose=True,
    normalize_activations="expected_average_only_in",
    # Training Parameters
    lr=3e-6,  # lower the better, we'll go fairly high to speed up the tutorial.
    adam_beta1=0.9,  # adam params (default, but once upon a time we experimented with these.)
    adam_beta2=0.999,
    lr_scheduler_name="constant",  # constant learning rate with warmup. Could be better schedules out there.
    lr_warm_up_steps=lr_warm_up_steps,  # this can help avoid too many dead features initially.
    lr_decay_steps=lr_decay_steps,  # this will help us avoid overfitting.
    l1_coefficient=5,  # will control how sparse the feature activations are
    l1_warm_up_steps=l1_warm_up_steps,  # this can help avoid too many dead features initially.
    lp_norm=1.0,  # the L1 penalty (and not a Lp for p < 1)
    train_batch_size_tokens=batch_size,
    context_size=512,  # will control the lenght of the prompts we feed to the model. Larger is better but slower. so for the tutorial we'll use a short one.
    # Activation Store Parameters
    n_batches_in_buffer=64,  # controls how many activations we store / shuffle.
    training_tokens=total_training_tokens,  # 100 million tokens is quite a few, but we want to see good stats. Get a coffee, come back.
    store_batch_size_prompts=16,
    # Resampling protocol
    use_ghost_grads=False,  # we don't use ghost grads anymore.
    feature_sampling_window=1000,  # this controls our reporting of feature sparsity stats
    dead_feature_window=1000,  # would effect resampling or ghost grads if we were using it.
    dead_feature_threshold=1e-4,  # would effect resampling or ghost grads if we were using it.
    # WANDB
    log_to_wandb=True,  # always use wandb unless you are just testing code.
    wandb_project="SAE finetuning",
    wandb_log_frequency=30,
    eval_every_n_wandb_logs=20,
    # Misc
    device=device,
    seed=42,
    n_checkpoints=0,
    checkpoint_path="checkpoints",
    dtype="float32",
)

# look at the next cell to see some instruction for what to do while this is running.
sparse_autoencoder = SAETrainingRunner(cfg).run()
