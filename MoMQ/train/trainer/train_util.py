import torch
from model.modeling_qwen2_moe import Qwen2MoeForCausalLM
from model.configuration_qwen2_moe import Qwen2MoeConfig
from model.modeling_qwen2 import Qwen2ForCausalLM, Qwen2RMSNorm
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoConfig,
    DataCollatorForTokenClassification
)
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch.nn as nn
from .argument import TrainingArguments
from .moe import MoELinear
import time

class DataCollatorForGeneration(DataCollatorForTokenClassification):
    def torch_call(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None

        no_labels_features = [{k: v for k, v in feature.items() if k != label_name} for feature in features]

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            no_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if labels is None:
            return batch

        sequence_length = batch["input_ids"].shape[1]
        padding_side = self.tokenizer.padding_side

        def to_list(tensor_or_iterable):
            if isinstance(tensor_or_iterable, torch.Tensor):
                return tensor_or_iterable.tolist()
            return list(tensor_or_iterable)

        if padding_side == "right":
            batch[label_name] = [
                to_list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels
            ]
        else:
            
            # pad_labels = []
            # for label in labels:
            #     if sequence_length - len(label) > 0:
            #         pad_labels.append([to_list(label)[0]] + [self.label_pad_token_id] * (sequence_length - len(label) - 1) + to_list(label))
            #     else:
            #         pad_labels.append([self.label_pad_token_id] * (sequence_length - len(label)) + to_list(label))
            # batch[label_name] = pad_labels
            

            batch[label_name] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + to_list(label) for label in labels
            ]
             

            
        batch[label_name] = torch.tensor(batch[label_name], dtype=torch.int64)
        return batch


def _replace_module(parent, child_name, new_module, child):
    setattr(parent, child_name, new_module)
    # It's not necessary to set requires_grad here, as that is handled by
    # _mark_only_adapters_as_trainable

    # child layer wraps the original module, unpack it
    if hasattr(child, "base_layer"):
        child = child.base_layer

    if not hasattr(new_module, "base_layer"):
        new_module.weight = child.weight
        if hasattr(child, "bias"):
            new_module.bias = child.bias

    if getattr(child, "state", None) is not None:
        if hasattr(new_module, "base_layer"):
            new_module.base_layer.state = child.state
        else:
            new_module.state = child.state
        new_module.to(child.weight.device)

    # dispatch to correct device
    prefix = 'lora_'
    for name, module in new_module.named_modules():
        if (prefix in name) or ("ranknum" in name):
            weight = (
                child.qweight
                if hasattr(child, "qweight")
                else child.W_q
                if hasattr(child, "W_q")
                else child.weight
            )
            module.to(weight.device)

def _get_submodules(model: nn.Module, key):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name

def _create_new_module(config, target: nn.Linear, target_name, moe_lora_target_modules):
    
    # print(target_name, moe_lora_target_modules)
    if target_name in moe_lora_target_modules:
        new_module = MoELinear(
            config,
            in_features=target.in_features,
            out_features=target.out_features,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias=False,
        )
    else:
        tmp = config.num_experts
        config.num_experts = 1
        new_module = MoELinear(
            config,
            in_features=target.in_features,
            out_features=target.out_features,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias=False,
        )
        config.num_experts = tmp

    return new_module

def is_target_module(key, lora_args):
    # target_list = [
    #     "q_proj",
    #     "k_proj",
    #     "v_proj",
    #     "o_proj",
    #     "up_proj",
    #     "gate_proj",
    #     "down_proj",
    # ]
    for target in lora_args.lora_target_modules:
        if key.find(target)>=0:
            return True
    return False


def replace_module(model: nn.Module, config, training_args, lora_args):
    key_list = [key for key, _ in model.named_modules()]
    for key in key_list:
        if not is_target_module(key, lora_args):
            continue
        parent, target, target_name = _get_submodules(model, key)
        new_module =_create_new_module(config, target, target_name, training_args.moe_lora_target_modules)
        _replace_module(parent, target_name, new_module, target)


def modify_model_after_init(model: nn.Module, config, training_args: TrainingArguments, lora_args):
    
    if training_args.use_moe_lora:
        print('Using MoE Lora')
        replace_module(model, config, training_args, lora_args)
        # Freeze model
        for par in model.parameters():
            par.requires_grad = False
        
        # Unfreezes LoRA
        for n, p in model.named_parameters():
            if 'lora_' in n:
                p.requires_grad = True
            if 'fusion_' in n:
                p.requires_grad = True

        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

        if training_args.unfreeze_layer_norms:
            for name, sub_module in model.named_modules():
                if isinstance(sub_module, (Qwen2RMSNorm)):
                    for param_name, param in sub_module.named_parameters():
                        param.requires_grad = True
        
        # Print trainable params
        for n,p in model.named_parameters():
            if p.requires_grad and n.find('.1.') >= 0:
                print(f"{n}, {p.shape}, {p.dtype}")
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total trainable parameters {total_trainable_params}")
        print(f"Total parameters {total_params}")

    if training_args.use_lora:
        print('Using Single Lora')
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

        model = get_peft_model(model, lora_config)

        # Print peft trainable params
        model.print_trainable_parameters()

        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()
        
        # Print trainable params
        for n,p in model.named_parameters():
            if p.requires_grad and n.find('.1.') >= 0:
                print(f"{n}, {p.shape}, {p.dtype}")
    
    return model


def load_tokenizer_and_model(model_args, training_args: TrainingArguments, lora_args):

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        trust_remote_code=True
    )
    # print(model_args.model_name_or_path,training_args.cache_dir,training_args.model_max_length)
    # assert 1>2
    # tokenizer.padding_side = "left"
    # if tokenizer.pad_token is None:
    #     tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
    )
    config.use_cache = False
        

    # Load model and tokenizer
    if model_args.model_type == 'moe':
        model = Qwen2MoeForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )
        config.output_router_logits = training_args.output_router_logits
        config.router_aux_loss_coef = training_args.router_aux_loss_coef
    elif model_args.model_type == 'qwen':
        # init moe lora config
        config.use_moe_lora = training_args.use_moe_lora
        config.use_moe_expert = training_args.use_moe_expert
        config.output_router_logits = training_args.output_router_logits
        config.router_aux_loss_coef = training_args.router_aux_loss_coef
        config.num_experts = training_args.num_experts
        config.num_experts_per_tok = training_args.num_experts_per_tok
        config.norm_topk_prob = False
        config.moe_intermediate_size = training_args.moe_intermediate_size

        if training_args.use_moe_lora:
            config.lora_r = lora_args.lora_r
            config.lora_alpha = lora_args.lora_alpha
            config.lora_dropout = lora_args.lora_dropout
            config.lora_route_type = training_args.lora_route_type

            config.dialect_num = training_args.dialect_num
            config.dialect_router_loss_coef = training_args.dialect_router_loss_coef
            config.enable_dialect_router = training_args.enable_dialect_router
            config.enable_label_smooth = training_args.enable_label_smooth
            config.smooth_factor = training_args.smooth_factor
            config.share_expert_num = training_args.share_expert_num
            config.add_moe_fusion = training_args.add_moe_fusion
            config.use_in_group_balance = training_args.use_in_group_balance
            config.hard_dialect_router = training_args.hard_dialect_router

        model = Qwen2ForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            # load_in_8bit=True,
            # quantization_config=BitsAndBytesConfig(
            #     load_in_4bit=True,
            #     bnb_4bit_use_double_quant=True,
            #     bnb_4bit_quant_type="nf4",
            #     bnb_4bit_compute_dtype=torch.bfloat16,
            # )
            # if training_args.use_lora and lora_args.q_lora
            # else None,
            # device_map="auto"
        )

    model = modify_model_after_init(model, config, training_args, lora_args)
    
    print(config)
    print('load done')


    return tokenizer, model
