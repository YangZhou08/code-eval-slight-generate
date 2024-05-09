from transformers import (
    LlamaTokenizer,
    LlamaForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
) 
from core import filter_code, run_eval, fix_indents
import os
import torch
import argparse 

from transformers.models.llama.modeling_llama import LlamaWeirdLargeTest 

# TODO: move to python-dotenv
# add hugging face access token here
TOKEN = ""

parser = argparse.ArgumentParser() 
# parser.add_argument("--model_name", type=str, help="", required = True) 
parser.add_argument("--loading_from_checkpoint", type = str, help = "", required = True) 
parser.add_argument("--experiment_setting", type = str, help = "", required = True) 
parser.add_argument("--kernelsize", type = int, help = "", required=True) 

args = parser.parse_args() 

potential_modelsnames = ["TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", "Cheng98/llama-160m"] 
model_labels = ["tinyllama", "smalllama"] 
labels = None 

# for idx, model_name in enumerate(potential_modelsnames): 
#     if args.model_name == model_name: 
#         labels = model_labels[idx] 
# print(model_name, labels) 

@torch.inference_mode()
def generate_batch_completion(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompt, batch_size
) -> list[str]:
    input_batch = [prompt for _ in range(batch_size)]
    inputs = tokenizer(input_batch, return_tensors="pt").to(model.device)
    input_ids_cutoff = inputs.input_ids.size(dim=1)

    generated_ids = model.generate(
        **inputs,
        use_cache=True,
        max_new_tokens=512,
        temperature=0.2,
        top_p=0.95,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,  # model has no pad token
    )

    batch_completions = tokenizer.batch_decode(
        [ids[input_ids_cutoff:] for ids in generated_ids],
        skip_special_tokens=True,
    )

    return [filter_code(fix_indents(completion)) for completion in batch_completions]


if __name__ == "__main__":
    # adjust for n = 10 etc
    num_samples_per_task = 10
    if labels is not None: 
        out_path = "results/{}/eval.jsonl".format(labels) 
        os.makedirs("results/{}".format(labels), exist_ok = True) 
    else: 
        os.makedirs("results/llama", exist_ok=True)
        out_path = "results/llama/eval.jsonl" 

    tokenizer = LlamaTokenizer.from_pretrained(
        # "huggyllama/llama-7b",
        "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", 
        # args.model_name, 
    ) 
    
    '''
    model = torch.compile(
        LlamaForCausalLM.from_pretrained(
            # "huggyllama/llama-7b",
            # "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", 
            args.model_name, 
            torch_dtype=torch.bfloat16,
        ) 
        .eval()
        .to(torch.bfloat16) 
        .to("cuda")
    ) 
    ''' 
    large_model = LlamaWeirdLargeTest.from_pretrained(args.loading_from_checkpoint).to(torch.bfloat16) 
    large_model.set_sliding_window_length(args.kernelsize) 
    large_model.addonsmallmodel.set_criticalpath(hostname = "lovelace") 
    large_model.set_msece_loss(use_mse_loss = False, ce_loss_only = True) 
    large_model.to(torch.bfloat16) 
    large_model.set_inference_setting(args.experiment_setting) 
    large_model.set_walpha(0.5) 
    large_model.set_slidingwindowlength(args.kernelsize) 
    large_model.set_tokenizer_bos_id(bos_id = tokenizer.bos_token_id, pad_id = tokenizer.pad_token_id) 
    large_model.set_cosinesimilarity(False) 
    large_model.config.pad_token_id = tokenizer.pad_token_id 
    large_model.addonsmallmodel.config.pad_token_id = tokenizer.pad_token_id 
    # large_model.model.eval() 
    # large_model.addonsmallmodel.eval() 
    
    model = torch.compile(
        large_model.eval().to(torch.bfloat16).to("cuda") 
    ) 
    
    run_eval(
        model,
        tokenizer,
        num_samples_per_task,
        out_path,
        generate_batch_completion,
        True,
    )
