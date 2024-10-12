import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/mnt/data/ckpt/linzhisheng/cache'
import torch
from transformers import Qwen2ForCausalLM, Qwen2MoeForCausalLM, AutoModelForCausalLM, AutoTokenizer, AutoConfig
from model.modeling_qwen2_moe import Qwen2MoeForCausalLM
from model.configuration_qwen2_moe import Qwen2MoeConfig

#base = 'Qwen/Qwen1.5-MoE-A2.7B'
# base = 'Qwen/Qwen2-1.5B-Instruct'
base = 'model/Qwen/CodeQwen1___5-7B-Chat'
output_dir = 'model/moe_from_codeqwen'

def init_from_qwen_dense():
    base_model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(base)
    tokenizer.save_pretrained(output_dir)

    for n,p in base_model.named_parameters():
        print(n,p.shape)

    # config = Qwen2MoeConfig()
    config = AutoConfig.from_pretrained(base)
    # moe config

    split_num = 16

    moe_intermediate_size=config.intermediate_size // split_num
    shared_expert_intermediate_size=config.intermediate_size
    num_experts_per_tok=4
    num_experts=8
    decoder_sparse_step=1
    norm_topk_prob=False
    output_router_logits=True
    router_aux_loss_coef=0.001
    mlp_only_layers=None

    config.decoder_sparse_step = decoder_sparse_step
    config.moe_intermediate_size = moe_intermediate_size
    config.shared_expert_intermediate_size = shared_expert_intermediate_size
    config.num_experts_per_tok = num_experts_per_tok
    config.num_experts = num_experts
    config.norm_topk_prob = norm_topk_prob
    config.output_router_logits = output_router_logits
    config.router_aux_loss_coef = router_aux_loss_coef
    config.mlp_only_layers = [] if mlp_only_layers is None else mlp_only_layers

    print(config)
    target_model = Qwen2MoeForCausalLM(config)

    target_model.lm_head.weight.data = base_model.lm_head.weight.data.clone()
    target_model.model.embed_tokens.weight.data = base_model.model.embed_tokens.weight.data.clone()
    target_model.model.norm.weight.data = base_model.model.norm.weight.data.clone()


    for i in range(config.num_hidden_layers):
        print('init layer {}'.format(i))
        # attention
        base_attn = base_model.model.layers[i].self_attn
        target_attn = target_model.model.layers[i].self_attn
        
        target_attn.q_proj.weight.data = base_attn.q_proj.weight.data.clone()
        target_attn.k_proj.weight.data = base_attn.k_proj.weight.data.clone()
        target_attn.v_proj.weight.data = base_attn.v_proj.weight.data.clone()
        target_attn.o_proj.weight.data = base_attn.o_proj.weight.data.clone()
        if base_attn.q_proj.bias is not None:
            target_attn.q_proj.bias.data = base_attn.q_proj.bias.data.clone()
        if base_attn.k_proj.bias is not None:
            target_attn.k_proj.bias.data = base_attn.k_proj.bias.data.clone()
        if base_attn.v_proj.bias is not None:
            target_attn.v_proj.bias.data = base_attn.v_proj.bias.data.clone()
        # norm
        base_input_ln = base_model.model.layers[i].input_layernorm
        base_post_attn_ln = base_model.model.layers[i].post_attention_layernorm
        target_input_ln = target_model.model.layers[i].input_layernorm
        target_post_attn_ln = target_model.model.layers[i].post_attention_layernorm

        target_input_ln.weight.data = base_input_ln.weight.data.clone()
        if hasattr(base_input_ln, 'bias') and base_input_ln.bias is not None:
            target_input_ln.bias.data = base_input_ln.bias.data.clone()

        target_post_attn_ln.weight.data = base_post_attn_ln.weight.data.clone()
        if hasattr(base_post_attn_ln, 'bias') and base_post_attn_ln.bias is not None:
            target_post_attn_ln.bias.data = base_post_attn_ln.bias.data.clone()

        
        base_mlp = base_model.model.layers[i].mlp
        # share expert
        if shared_expert_intermediate_size>0:
            target_expert = target_model.model.layers[i].mlp.shared_expert
            target_expert.gate_proj.weight.data = base_mlp.gate_proj.weight.data.clone()
            target_expert.up_proj.weight.data = base_mlp.up_proj.weight.data.clone()
            target_expert.down_proj.weight.data = base_mlp.down_proj.weight.data.clone()

            if hasattr(base_mlp.gate_proj, 'bias') and base_mlp.gate_proj.bias is not None:
                target_expert.gate_proj.bias.data = base_mlp.gate_proj.bias.data.clone()
            if hasattr(base_mlp.up_proj, 'bias') and base_mlp.up_proj.bias is not None:
                target_expert.up_proj.bias.data = base_mlp.up_proj.bias.data.clone()
            if hasattr(base_mlp.down_proj, 'bias') and base_mlp.down_proj.bias is not None:
                target_expert.down_proj.bias.data = base_mlp.down_proj.bias.data.clone()


        
        # expert group
        gate_proj_splits = torch.chunk(base_mlp.gate_proj.weight.data, split_num, dim=0)
        up_proj_splits = torch.chunk(base_mlp.up_proj.weight.data, split_num, dim=0)
        down_proj_splits = torch.chunk(base_mlp.down_proj.weight.data, split_num, dim=1)

        for expert_group in target_model.model.layers[i].mlp.expert_groups:
            for j in range(len(expert_group.experts)):
                print(j, j%split_num)
                expert_group.experts[j].gate_proj.weight.data = gate_proj_splits[j%split_num].clone()
                expert_group.experts[j].up_proj.weight.data = up_proj_splits[j%split_num].clone()
                expert_group.experts[j].down_proj.weight.data = down_proj_splits[j%split_num].clone()

    print(target_model)
    target_model.save_pretrained(output_dir)
    print(base_model.model.layers[23].mlp.gate_proj.weight[0])
    print(target_model.model.layers[23].mlp.shared_expert.gate_proj.weight[0])


def init_from_qwen_moe():


    base_model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(base)
    tokenizer.save_pretrained(output_dir)

    for n,p in base_model.named_parameters():
        print(n,p.shape)
    
    '''
    config = Qwen2MoeConfig()
    config.intermediate_size=5632
    config.shared_expert_intermediate_size=5632
    config.moe_intermediate_size=1408
    '''
    config = AutoConfig.from_pretrained(base)
    # moe config
    moe_intermediate_size=config.intermediate_size // 4
    shared_expert_intermediate_size=config.intermediate_size
    num_experts_per_tok=4
    num_experts=24
    decoder_sparse_step=1
    norm_topk_prob=False
    output_router_logits=False
    router_aux_loss_coef=0.001
    mlp_only_layers=None
    config.decoder_sparse_step = decoder_sparse_step
    config.moe_intermediate_size = moe_intermediate_size
    config.shared_expert_intermediate_size = shared_expert_intermediate_size
    config.num_experts_per_tok = num_experts_per_tok
    config.num_experts = num_experts
    config.norm_topk_prob = norm_topk_prob
    config.output_router_logits = output_router_logits
    config.router_aux_loss_coef = router_aux_loss_coef
    config.mlp_only_layers = [] if mlp_only_layers is None else mlp_only_layers

    print(config)
    target_model = Qwen2MoeForCausalLM(config)


    target_model.lm_head.weight.data = base_model.lm_head.weight.data.clone()
    target_model.model.embed_tokens.weight.data = base_model.model.embed_tokens.weight.data.clone()
    target_model.model.norm.weight.data = base_model.model.norm.weight.data.clone()


    for i in range(config.num_hidden_layers):
        print('init layer {}'.format(i))
        # attention
        base_attn = base_model.model.layers[i].self_attn
        target_attn = target_model.model.layers[i].self_attn
        
        target_attn.q_proj.weight.data = base_attn.q_proj.weight.data.clone()
        target_attn.k_proj.weight.data = base_attn.k_proj.weight.data.clone()
        target_attn.v_proj.weight.data = base_attn.v_proj.weight.data.clone()
        target_attn.o_proj.weight.data = base_attn.o_proj.weight.data.clone()
        if base_attn.q_proj.bias is not None:
            target_attn.q_proj.bias.data = base_attn.q_proj.bias.data.clone()
        if base_attn.k_proj.bias is not None:
            target_attn.k_proj.bias.data = base_attn.k_proj.bias.data.clone()
        if base_attn.v_proj.bias is not None:
            target_attn.v_proj.bias.data = base_attn.v_proj.bias.data.clone()
        # norm
        base_input_ln = base_model.model.layers[i].input_layernorm
        base_post_attn_ln = base_model.model.layers[i].post_attention_layernorm
        target_input_ln = target_model.model.layers[i].input_layernorm
        target_post_attn_ln = target_model.model.layers[i].post_attention_layernorm

        target_input_ln.weight.data = base_input_ln.weight.data.clone()
        if hasattr(base_input_ln, 'bias') and base_input_ln.bias is not None:
            target_input_ln.bias.data = base_input_ln.bias.data.clone()

        target_post_attn_ln.weight.data = base_post_attn_ln.weight.data.clone()
        if hasattr(base_post_attn_ln, 'bias') and base_post_attn_ln.bias is not None:
            target_post_attn_ln.bias.data = base_post_attn_ln.bias.data.clone()

        # share expert
        base_mlp = base_model.model.layers[i].mlp.shared_expert
        target_expert = target_model.model.layers[i].mlp.shared_expert
        target_expert.gate_proj.weight.data = base_mlp.gate_proj.weight.data.clone()
        target_expert.up_proj.weight.data = base_mlp.up_proj.weight.data.clone()
        target_expert.down_proj.weight.data = base_mlp.down_proj.weight.data.clone()

        if hasattr(base_mlp.gate_proj, 'bias') and base_mlp.gate_proj.bias is not None:
            target_expert.gate_proj.bias.data = base_mlp.gate_proj.bias.data.clone()
        if hasattr(base_mlp.up_proj, 'bias') and base_mlp.up_proj.bias is not None:
            target_expert.up_proj.bias.data = base_mlp.up_proj.bias.data.clone()
        if hasattr(base_mlp.down_proj, 'bias') and base_mlp.down_proj.bias is not None:
            target_expert.down_proj.bias.data = base_mlp.down_proj.bias.data.clone()



        # expert group
        base_experts = base_model.model.layers[i].mlp.experts

        count = 0
        for expert_group in target_model.model.layers[i].mlp.expert_groups:
            for j in range(len(expert_group.experts)):
                print(j, count)
                expert_group.experts[j].gate_proj.weight.data = base_experts[count].gate_proj.weight.data.clone()
                expert_group.experts[j].up_proj.weight.data = base_experts[count].up_proj.weight.data.clone()
                expert_group.experts[j].down_proj.weight.data = base_experts[count].down_proj.weight.data.clone()
                count+=1
    target_model.save_pretrained(output_dir)
    print(base_model.model.layers[23].mlp.shared_expert.gate_proj.weight[0])
    print(target_model.model.layers[23].mlp.shared_expert.gate_proj.weight[0])


init_from_qwen_dense()
#init_from_qwen_moe()
