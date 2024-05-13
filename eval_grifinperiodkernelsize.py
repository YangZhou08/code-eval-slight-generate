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
import datetime 
hash_of_time = str(datetime.datetime.now()).split('.')[-1] 
print("the hash of time is {}".format(hash_of_time)) 

from transformers.models.llama.modeling_llama import LlamaWeirdLargeTest 
from transformers.models.llama.modeling_llama import LlamaForCausalLM2 
from transformers import AutoModelForCausalLM 
from transformers import AutoConfig 
from transformers.models.llama.modeling_llama import LlamaConfig 
from griffin.llama import get_llama_griffin 
from griffin.llama_chunk_redirecting import get_llama_griffintwo 

# TODO: move to python-dotenv
# add hugging face access token here
TOKEN = ""

parser = argparse.ArgumentParser() 
parser.add_argument("--model_name", type=str, help="", required = True) 
parser.add_argument("--experiment", type = str, choices = ["plain", "griffin_plain", "griffin_period"], required = True) 

args = parser.parse_args() 

# labels = "kernelsize{}_experimentsetting{}_finetuned{}_{}".format(args.kernelsize, args.experiment_setting, args.largefinetuned, hash_of_time) 

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
    out_path = "results/{}/eval.jsonl".format("Llama2_7B_{}".format(args.experiment)) 
    os.makedirs("results/{}".format("Llama2_7B_{}".format(args.experiment), exist_ok = True)) 

    tokenizer = LlamaTokenizer.from_pretrained(
        # "huggyllama/llama-7b", 
        # "meta-llama/Llama-2-7b-hf", 
        args.model_name, 
    ) 
    
    density = 0.5 
    if args.experiment == "griffin_plain": 
        config = AutoConfig.from_pretrained(args.model_name) 
        model = AutoModelForCausalLM.from_pretrained(args.model_name).to(torch.float16).to("cuda:0") 
        schedule = [density for _ in range(config.num_hidden_layers)] 
        model.config.mode = "gen" 
        # large_model.config.chunksize = 8 
        model.config.selection_method = "topk" 
        model = get_llama_griffin(model, schedule) 
    elif args.experiment == "griffin_period": 
        config = LlamaConfig.from_pretrained(args.model_name) 
        model = LlamaForCausalLM2.from_pretrained(args.model_name).to(torch.float16) 
        schedule = [density for _ in range(config.num_hidden_layers)] 
        model.config.chunksize = 8 
        
        model.config.mode = "gen" 
        # large_model.config.chunksize = 8 
        model.config.selection_method = "topk" 
    
        model = get_llama_griffintwo(model, schedule) 
    else: 
        raise NotImplementedError("Not implemented yet") 
    
    # self._model = get_llama_griffin(model, schedule) 
    model.eval() 
    
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
    ''' 
    run_eval(
        model,
        tokenizer,
        num_samples_per_task,
        out_path,
        generate_batch_completion,
        True,
    )
