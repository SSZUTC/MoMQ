import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List

class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.share_merged = False
        self.merge_weights = merge_weights

class MoELinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        config,
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, 1, 1, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        self.num_experts = config.num_experts
        self.lora_route_type = config.lora_route_type
        self.top_k = config.num_experts_per_tok
        self.dialect_num = config.dialect_num
        self.enable_dialect_router = config.enable_dialect_router
        self.hard_dialect_router = config.hard_dialect_router
        self.in_features = in_features
        self.out_features = out_features
        self.share_expert_num = config.share_expert_num
        self.use_in_group_balance = config.use_in_group_balance
        # Actual trainable parameters
        if r > 0:
            if self.num_experts == 1:
                self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features), dtype=torch.bfloat16))
                self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r), dtype=torch.bfloat16))
            else:
                self.lora_A = nn.ParameterList()
                self.lora_B = nn.ParameterList()
                for _ in range(self.num_experts):
                    self.lora_A.append(nn.Parameter(self.weight.new_zeros((r, in_features), dtype=torch.bfloat16)))
                    self.lora_B.append(nn.Parameter(self.weight.new_zeros((out_features, r), dtype=torch.bfloat16)))

                if self.lora_route_type == 'token' and self.share_expert_num > 0:
                    self.lora_shared_A = nn.Parameter(self.weight.new_zeros((r * self.share_expert_num, in_features), dtype=torch.bfloat16))
                    self.lora_shared_B = nn.Parameter(self.weight.new_zeros((out_features, r * self.share_expert_num), dtype=torch.bfloat16))

                self.lora_moe_gate = nn.Parameter(torch.zeros(in_features, self.num_experts, dtype=torch.bfloat16), requires_grad=True)

                self.temperature = 1
                self.softmax = nn.Softmax(dim=-1)
                if self.enable_dialect_router:
                    self.lora_dialect_gate = nn.Parameter(torch.zeros(in_features, self.dialect_num, dtype=torch.bfloat16), requires_grad=True)

            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize B the same way as the default for nn.Linear and A to zero
            # this is different than what is described in the paper but should not affect performance
            if self.num_experts == 1:
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B)
            else:
                for i in range(self.num_experts):
                    nn.init.kaiming_uniform_(self.lora_A[i], a=math.sqrt(5))
                    nn.init.zeros_(self.lora_B[i])

        if hasattr(self, 'lora_shared_A'):
            nn.init.kaiming_uniform_(self.lora_shared_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_shared_B)

        if hasattr(self, 'lora_moe_gate'):
            nn.init.kaiming_uniform_(self.lora_moe_gate, a=math.sqrt(5))
            # nn.init.zeros_(self.lora_moe_gate)

        if hasattr(self, 'lora_dialect_gate'):
            nn.init.kaiming_uniform_(self.lora_dialect_gate, a=math.sqrt(5))
            # nn.init.zeros_(self.lora_dialect_gate)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged and self.num_experts == 1:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
            
            if self.merge_weights and self.share_merged and hasattr(self, 'lora_shared_A'):
                if self.r > 0:
                    self.weight.data -= T(self.lora_shared_B @ self.lora_shared_A) * self.scaling
                self.share_merged = False
        else:
            if self.merge_weights and not self.merged and self.num_experts == 1:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True
            
            if self.merge_weights and not self.share_merged and hasattr(self, 'lora_shared_A'):
                if self.r > 0:
                    self.weight.data += T(self.lora_shared_B @ self.lora_shared_A) * self.scaling
                self.share_merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            router_logits = None
            if self.num_experts == 1:        
                result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            else:
                if self.lora_route_type == 'sentence':
                    gate_x = x.mean(1).unsqueeze(1)
                    gete_out = gate_x @ self.lora_moe_gate
                    gating_weights = self.softmax(gete_out/self.temperature)
                    all_results = []
                    x = self.lora_dropout(x)
                    for i in range(self.num_experts):
                        all_results.append((x @ self.lora_A[i].transpose(0, 1) @ self.lora_B[i].transpose(0, 1)) * self.scaling)
                    final_output = torch.stack(all_results, dim=3) @ gating_weights.unsqueeze(3)
                    final_output = final_output.squeeze()
                    if len(final_output.shape) == 2:
                        final_output = final_output.unsqueeze(0)
                    result += final_output
                elif self.lora_route_type == 'token':
                    # add share

                    if not self.share_merged and self.share_expert_num > 0:
                        share_output = (self.lora_dropout(x) @ self.lora_shared_A.transpose(0, 1) @ self.lora_shared_B.transpose(0, 1)) * self.scaling
                        result += share_output

                    # compute expert part
                    batch_size, sequence_length, hidden_dim = x.shape

                    if self.hard_dialect_router and self.enable_dialect_router:
                        dialect_logits = x.mean(dim=1) @ self.lora_moe_gate
                        select_group = F.gumbel_softmax(dialect_logits, tau=1, hard=True, dim=-1).argmax(dim=-1)

                        x = x.view(-1, hidden_dim)
                        router_logits = x @ self.lora_moe_gate
                        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
                        num_experts_per_group = self.num_experts // self.dialect_num

                        routing_weights = routing_weights.view(batch_size, -1, self.num_experts)

                        mask = torch.zeros_like(routing_weights)
                        for idx, group in enumerate(select_group):
                            start_idx = group * num_experts_per_group
                            end_idx = start_idx + num_experts_per_group
                            mask[idx, :, start_idx:end_idx] = 1
                            # routing_weights[idx, :, :start_idx] = 0
                            # routing_weights[idx, :, end_idx:] = 0
                        routing_weights = routing_weights * mask
                        routing_weights = routing_weights.view(-1, self.num_experts)
                        
                    else:
                        x = x.view(-1, hidden_dim)
                        router_logits = x @ self.lora_moe_gate
                        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
                        num_experts_per_group = self.num_experts // self.dialect_num
                        if self.enable_dialect_router:
                            dialect_logits = x @ self.lora_dialect_gate
                            dialect_weights = F.softmax(dialect_logits, dim=1, dtype=torch.float)

                    
                            routing_weights = routing_weights.view(-1, self.dialect_num, num_experts_per_group) * dialect_weights.unsqueeze(2)
                            routing_weights = routing_weights.view(-1, self.num_experts)


                    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1, sorted=False)
                    routing_weights = routing_weights.to(x.dtype)

                    if self.training:
                        final_hidden_states = torch.zeros(
                            (batch_size * sequence_length, self.out_features), dtype=x.dtype, device=x.device
                        )
                        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

                        for expert_idx in range(self.num_experts):
                            idx, top_x = torch.where(expert_mask[expert_idx])
                            current_state = x[None, top_x].reshape(-1, hidden_dim)
                            current_state = self.lora_dropout(current_state)
                            expert_output = (current_state @ self.lora_A[expert_idx].transpose(0, 1) @ self.lora_B[expert_idx].transpose(0, 1)) * self.scaling
                            current_hidden_states = expert_output * routing_weights[top_x, idx, None]
                            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(x.dtype))
                    else:
                        final_hidden_states = self.moe_infer(x, selected_experts, routing_weights)
                        
                    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, self.out_features)
                    if self.use_in_group_balance:
                        router_logits = router_logits.reshape(-1, num_experts_per_group)
                    result += final_hidden_states
                    result = (result, router_logits)

                    if self.enable_dialect_router:
                        dialect_logits = dialect_logits.view(batch_size, -1, self.dialect_num)
                        result += (dialect_logits,)
                        
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)
        
    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], self.num_experts))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        sorted_tokens_shape = sorted_tokens.shape

        tokens_per_expert = tokens_per_expert.cpu().numpy()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            # expert = self.experts[i]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            # expert_out = expert(tokens_for_this_expert)
            expert_out = (self.lora_dropout(tokens_for_this_expert) @ self.lora_A[i].transpose(0, 1) @ self.lora_B[i].transpose(0, 1)) * self.scaling
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out