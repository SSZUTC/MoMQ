import sys

import torch.utils
import torch.utils.data
# sys.path.append('../')
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '../../cache'
os.environ['HF_HOME'] = '/mnt/model_cache/'

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoConfig
from model.modeling_qwen2_moe import Qwen2MoeForCausalLM
from model.modeling_qwen2 import Qwen2ForCausalLM, Qwen2RMSNorm
from utils.common_utils import read_json, write_json
import torch
from transformers import GenerationMixin, GenerationConfig
from safetensors.torch import load_file
import torch.nn as nn
from trainer.moe import MoELinear


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

def is_target_module(key):
    target_list = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "gate_proj",
        "down_proj",
    ]
    for target in target_list:
        if key.find(target)>=0:
            return True
    return False

def replace_module(model: nn.Module, config):
    key_list = [key for key, _ in model.named_modules()]
    for key in key_list:
        if not is_target_module(key):
            continue
        parent, target, target_name = _get_submodules(model, key)
        new_module =_create_new_module(config, target, target_name, ['down_proj'])
        _replace_module(parent, target_name, new_module, target)



device = 'cuda:0'
# load
load_path1 = '/mnt/model_cache/model-00001-of-00002.safetensors'
load_path2 = '/mnt/model_cache/model-00002-of-00002.safetensors'

state_dict_part1 = load_file(load_path1, device=device)
state_dict_part2 = load_file(load_path2, device=device)


combined_state_dict = {**state_dict_part1, **state_dict_part2}

# print(combined_state_dict.keys())



model_path = 'Qwen/Qwen2-7B-Instruct'


config = AutoConfig.from_pretrained(
    model_path,
    trust_remote_code=True,
)
config.use_cache=True
print(config)
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
)
tokenizer.padding_side = "left"

config.use_moe_lora = True
config.use_moe_expert = False
config.output_router_logits = True
config.router_aux_loss_coef = 0
config.num_experts = 32
config.num_experts_per_tok = 2
config.norm_topk_prob = False
config.moe_intermediate_size = 128

config.lora_r = 128
config.lora_alpha = 256
config.lora_dropout = 0.05
config.lora_route_type = 'token'
config.dialect_num = 4
config.dialect_router_loss_coef = 0
config.enable_dialect_router = True
config.enable_label_smooth = False
config.smooth_factor = 0
config.share_expert_num = 2
config.add_moe_fusion = False
config.use_in_group_balance = False

model = Qwen2ForCausalLM.from_pretrained(
    model_path,
    config=config,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="flash_attention_2"
).to(device)

replace_module(model, config)

for n, p in model.named_parameters():
    
    if n in combined_state_dict:
        p.data = combined_state_dict[n].to(device)

# model.load_state_dict(combined_state_dict， strict=False)

for n, p in model.named_parameters():
    if n.find('.1.') >= 0:
        print(n, p.device)
prompt = [
      {
        "content": "你是一名PostgreSQL专家，现在需要阅读并理解下面的【数据库schema】描述，以及可能用到的【参考信息】，并运用PostgreSQL知识生成sql语句回答【用户问题】。\n【用户问题】\nWhich conference published the most publications in the last 15 years? Give the conference name and publication count.\n\n【数据库schema】\n【DB_ID】 academic\n【Schema】\n# Table: author\n[\n  (aid:bigint, Unique identifier for each author. Value examples: ['1', '2', '3'].),\n  (homepage:text, URL of the author's personal website. Value examples: ['www.larry.com', 'www.ashish.com', 'www.noam.com'].),\n  (name:text, Name of the author. Value examples: ['Larry Summers', 'Ashish Vaswani', 'Noam Shazeer'].),\n  (oid:bigint, Foreign key referencing the organization the author belongs to. Value examples: ['2', '3', '3'].),\n]\n# Table: cite\n[\n  (cited:bigint, ID of the publication being cited. Value examples: ['1', '1', '1'].),\n  (citing:bigint, ID of the publication that is citing another publication. Value examples: ['2', '3', '4'].),\n]\n# Table: conference\n[\n  (cid:bigint, Unique identifier for a conference. Value examples: ['1', '2', '3'].),\n  (homepage:text, The homepage URL for the conference. Value examples: ['www.isa.com', 'www.aaas.com', 'www.icml.com'].),\n  (name:text, The name of the conference. Value examples: ['ISA', 'AAAS', 'ICML'].),\n]\n# Table: domain\n[\n  (did:bigint, Unique identifier for a domain. Value examples: ['1', '2', '3'].),\n  (name:text, Name of the domain. Value examples: ['Data Science', 'Natural Sciences', 'Computer Science'].),\n]\n# Table: domain_author\n[\n  (aid:bigint, Foreign key referencing the author table's primary key. Value examples: ['1', '1'].),\n  (did:bigint, Foreign key referencing the domain table's primary key. Value examples: ['2', '4'].),\n]\n# Table: domain_conference\n[\n  (cid:bigint, Foreign key referencing the cid column in the conference table. Value examples: ['1', '2', '3'].),\n  (did:bigint, Foreign key referencing the did column in the domain table. Value examples: ['2', '4', '5'].),\n]\n# Table: domain_journal\n[\n  (did:bigint, Foreign key referencing the domain table's primary key. Value examples: ['1', '2'].),\n  (jid:bigint, Foreign key referencing the journal table's primary key. Value examples: ['2', '3'].),\n]\n# Table: domain_keyword\n[\n  (did:bigint, Foreign key referencing the 'did' column of the 'domain' table. Value examples: ['1', '2'].),\n  (kid:bigint, Foreign key referencing the 'kid' column of the 'keyword' table. Value examples: ['2', '3'].),\n]\n# Table: domain_publication\n[\n  (did:bigint, Foreign key referencing the domain table's primary key column (did). Value examples: ['4', '2', '1'].),\n  (pid:bigint, Foreign key referencing the publication table's primary key column (pid). Value examples: ['1', '2', '3'].),\n]\n# Table: journal\n[\n  (homepage:text, The homepage URL for the journal. Value examples: ['www.aijournal.com', 'www.nature.com', 'www.science.com', 'www.ml.com'].),\n  (jid:bigint, Unique identifier for a journal. Value examples: ['1', '2', '3', '4'].),\n  (name:text, The name of the journal. Value examples: ['Journal of Artificial Intelligence Research', 'Nature', 'Science', 'Journal of Machine Learning Research'].),\n]\n# Table: keyword\n[\n  (keyword:text, The actual keyword. Value examples: ['AI', 'Neuroscience', 'Machine Learning', 'Keyword 4'].),\n  (kid:bigint, Unique identifier for a keyword. Value examples: ['1', '2', '3', '4'].),\n]\n# Table: organization\n[\n  (continent:text, Continent where the organization is located. Value examples: ['Asia', 'North America', 'North America'].),\n  (homepage:text, URL of the organization's homepage. Value examples: ['www.organization1.com', 'www.organization2.com', 'www.organization3.com'].),\n  (name:text, Name of the organization. Value examples: ['Organization 1', 'Organization 2', 'Organization 3'].),\n  (oid:bigint, Unique identifier for the organization. Value examples: ['1', '2', '3'].),\n]\n# Table: publication\n[\n  (abstract:text, The abstract of the publication. Value examples: ['Abstract 1', 'Abstract 2'].),\n  (cid:bigint, The ID of the conference where the publication was presented. Value examples: ['1', '2'].),\n  (citation_num:bigint, The number of citations received by the publication. Value examples: ['4', '2'].),\n  (jid:bigint, The ID of the journal where the publication was published. Value examples: ['1', '2'].),\n  (pid:bigint, The unique ID of the publication. Value examples: ['1', '2'].),\n  (reference_num:bigint, The number of references cited by the publication. Value examples: ['0', '1'].),\n  (title:text, The title of the publication. Value examples: ['The Effects of Climate Change on Agriculture', 'A Study on the Effects of Social Media on Mental Health'].),\n  (year:bigint, The year of publication. Value examples: ['2020', '2020'].),\n]\n# Table: publication_keyword\n[\n  (pid:bigint, Foreign key referencing the publication table's primary key (pid). Value examples: ['1', '2'].),\n  (kid:bigint, Foreign key referencing the keyword table's primary key (kid). Value examples: ['2', '3'].),\n]\n# Table: writes\n[\n  (aid:bigint, Foreign key referencing the author table's primary key. Value examples: ['1', '1', '2', '2'].),\n  (pid:bigint, Foreign key referencing the publication table's primary key. Value examples: ['1', '2', '3', '4'].),\n]\n\n【Foreign keys】\nauthor.aid = domain_author.aid\nauthor.oid = organization.oid\nauthor.aid = writes.aid\ncite.cited = publication.pid\nconference.cid = domain_conference.cid\nconference.cid = publication.cid\ndomain.did = domain_author.did\ndomain.did = domain_conference.did\ndomain.did = domain_journal.did\ndomain.did = domain_keyword.did\ndomain_journal.jid = journal.jid\ndomain_keyword.kid = keyword.kid\ndomain_publication.pid = publication.pid\njournal.jid = publication.jid\nkeyword.kid = publication_keyword.kid\npublication.pid = publication_keyword.pid\npublication.pid = writes.pid\n\n【参考信息】\n\n\n【用户问题】\nWhich conference published the most publications in the last 15 years? Give the conference name and publication count.\n\n```sql",
        "role": "user"
      }
    ]
text = tokenizer.apply_chat_template(
    prompt,
    tokenize=False,
    # chat_template=TEMPLATE,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], padding=True, truncation=True, return_tensors="pt", return_token_type_ids=False).to(device)

generation_config = GenerationConfig(
    pad_token_id = tokenizer.pad_token_id,
    eos_token_id = tokenizer.eos_token_id,
    max_new_tokens = 512,
    temperature = None,
    num_beams = 1,
    top_p = None,
    do_sample = False,
    use_cache = True,
    output_hidden_states=True,
    output_attentions=True,
    return_dict_in_generate=True
)
result = model.generate(
    **model_inputs,
    generation_config = generation_config,
)


token_router_logits_list = []
for layer in range(len(result.hidden_states[0])):
    token_concat_list = []
    for token in range(len(result.hidden_states)):
        token_concat_list.append(result.hidden_states[token][layer])
    token_concat_all = torch.concat(token_concat_list, dim=0)
    token_router_logits_list.append(token_concat_all)
token_router_logits_all = torch.concat([t.unsqueeze(0) for t in token_router_logits_list], dim=0)

dialect_router_logits_list = []
for layer in range(len(result.attentions[0])):
    dialect_concat_list = []
    for token in range(len(result.attentions)):
        dialect_concat_list.append(result.attentions[token][layer].view(-1 ,4))
    dialect_concat_all = torch.concat(dialect_concat_list, dim=0)
    dialect_router_logits_list.append(dialect_concat_all)

dialect_router_logits_all = torch.concat([t.unsqueeze(0) for t in dialect_router_logits_list], dim=0)

print(token_router_logits_all.shape)
print(dialect_router_logits_all.shape)

torch.save(token_router_logits_all, 'expert_logits.pt')
torch.save(dialect_router_logits_all, 'dialect_logits.pt')

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, result.sequences)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

print(response)
