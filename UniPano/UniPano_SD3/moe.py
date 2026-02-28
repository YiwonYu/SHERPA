import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from diffusers.models.lora import LoRALinearLayer


class TopkRouter(nn.Module):
    def __init__(self, in_channels, num_experts, top_k):
        super(TopkRouter, self).__init__()
        self.top_k = top_k 
        self.linear = nn.Linear(in_channels, num_experts)

    def forward(self, x):
        logits = self.linear(x)
        topk_logits, indices = logits.topk(self.top_k, dim=-1)
        sparse_logits = torch.full_like(logits, float('-inf')).scatter(-1, indices, topk_logits)
        return F.softmax(sparse_logits, dim=-1), indices


class NoisyTopkRouter(nn.Module):
    def __init__(self, in_channels, num_experts, top_k):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k 
        self.linear = nn.Linear(in_channels, num_experts)
        self.noise_linear = nn.Linear(in_channels, num_experts)

    def forward(self, x):
        logits = self.linear(x)
        noise_logits = self.noise_linear(x)
        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noisy_logits = logits + noise
        topk_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        sparse_logits = torch.full_like(logits, float('-inf')).scatter(-1, indices, topk_logits)
        return F.softmax(sparse_logits, dim=-1), indices


class SparseMoELoRA(nn.Module):
    def __init__(self, in_channels, out_channels, num_experts, top_k, rank, alpha, use_noisy_router=True):
        super(SparseMoELoRA, self).__init__()
        self.router = NoisyTopkRouter(in_channels, num_experts, top_k) if use_noisy_router else TopkRouter(in_channels, num_experts, top_k)
        self.experts = nn.ModuleList([LoRALinearLayer(in_channels, out_channels, rank, alpha) for _ in range(num_experts)])
        self.top_k = top_k
        self.out_channels = out_channels

    def forward(self, x):
        b, l, c = x.shape
        gating_out, indices = self.router(x)
        final_out = torch.zeros(b, l, self.out_channels, dtype=x.dtype, device=x.device)

        flat_x = x.view(-1, x.size(-1))
        flat_gating_out = gating_out.view(-1, gating_out.size(-1))

        for i, expert in enumerate(self.experts):
            expert_msk = (indices == i).any(dim=-1)
            flat_msk = expert_msk.view(-1)

            if flat_msk.any():
                expert_inp = flat_x[flat_msk]
                expert_out = expert(expert_inp)

                gating_scores = flat_gating_out[flat_msk, i].unsqueeze(1)
                weighted_out = expert_out * gating_scores

                final_out[expert_msk] += weighted_out.squeeze(1)

        return final_out

        
######### DiT MoE implementation #########
class MoEGate(nn.Module):
    def __init__(self, embed_dim, num_experts=16, num_experts_per_tok=2, aux_loss_alpha=0.01):
        super().__init__()
        self.top_k = num_experts_per_tok
        self.n_routed_experts = num_experts
        self.scoring_func = 'softmax'
        self.alpha = aux_loss_alpha
        self.seq_aux = False
        # topk selection algorithm
        self.norm_topk_prob = False
        self.gating_dim = embed_dim
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init  as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        # print(bsz, seq_len, h)
        ### compute gating score
        hidden_states = hidden_states.reshape(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        ### select top-k experts
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
            
        ### expert-level computation auxiliary loss
        if (torch.is_grad_enabled() or self.training) and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            # always compute aux loss based on the naive greedy topk method
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss, torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim = 1)).sum(dim = 1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha

            z_loss = torch.mean(torch.log(torch.exp(logits).sum(dim=1)) ** 2) * 0.001
            aux_loss = aux_loss + z_loss
        else:
            aux_loss = None
        return topk_idx, topk_weight, aux_loss


class AddAuxiliaryLoss(torch.autograd.Function):
    """
    The trick function of adding auxiliary (aux) loss,
    which includes the gradient of the aux loss during backpropagation.
    """
    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss


class SparseMoeLoRA(nn.Module):
    """
    A mixed of LoRA expert module containing shared experts.
    """
    def __init__(self, embed_dim, out_dim, rank, alpha, num_experts=16, num_experts_per_tok=2, sync_expert=False):
        super().__init__()
        self.num_experts_per_tok = num_experts_per_tok
        # self.experts = nn.ModuleList([MoeMLP(hidden_size = embed_dim, intermediate_size = mlp_ratio * embed_dim, pretraining_tp=pretraining_tp) for i in range(num_experts)])
        self.experts = nn.ModuleList([LoRALinearLayer(embed_dim, out_dim, rank, alpha) for _ in range(num_experts)])
        self.gate = MoEGate(embed_dim=embed_dim, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok)
        self.n_shared_experts = 2
        if self.n_shared_experts is not None:
            self.shared_experts = LoRALinearLayer(embed_dim, out_dim, rank, alpha)
        self.sync_expert = sync_expert
        self.beta = 0.5

    @torch.no_grad()
    def sync_expert_weight(self):
        if not self.sync_expert:
            return
        print("Syncing experts weights...")
        down_weights = [expert.down.weight.data for expert in self.experts]
        up_weights = [expert.up.weight.data for expert in self.experts]
        assert len(down_weights) == len(up_weights)
        for idx in range(len(down_weights)):
            self.experts[idx].down.weight.data = (1 - self.beta) * down_weights[idx] \
                                                + self.beta / (len(down_weights) - 1) * (torch.stack(down_weights, dim=0).sum(dim=0) - down_weights[idx])
            self.experts[idx].up.weight.data = (1 - self.beta) * up_weights[idx] \
                                                + self.beta / (len(up_weights) - 1) * (torch.stack(up_weights, dim=0).sum(dim=0) - up_weights[idx])
    
    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
        # print(topk_idx.tolist(), print(len(topk_idx.tolist())))
        # global selected_ids_list
        # selected_ids_list.append(topk_idx.tolist())

        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if (torch.is_grad_enabled() or self.training):
            hidden_states = hidden_states.repeat_interleave(self.num_experts_per_tok, dim=0)
            y = torch.empty_like(hidden_states, dtype=hidden_states.dtype)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i])#.float()
            y = (y.reshape(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y =  y.reshape(*orig_shape)
            y = AddAuxiliaryLoss.apply(y, aux_loss)
        else:
            y = self.moe_infer(hidden_states, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.num_experts_per_tok
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i-1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            
            # for fp16 and other dtype
            expert_cache = expert_cache.to(expert_out.dtype)
            expert_cache.scatter_reduce_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out, reduce='sum')
        return expert_cache
    
    
class MoLELinear(nn.Module):
    def __init__(self, base_layer, rank, alpha, num_experts=4, num_experts_per_tok=2):
        super().__init__()
        self.base_layer = base_layer
        self.base_layer.requires_grad_(False)
        self.adaptor = SparseMoeLoRA(
            base_layer.in_features, 
            base_layer.out_features, 
            rank, alpha, num_experts=num_experts, num_experts_per_tok=2
        )
    
    def forward(self, hidden_states):
        if self.adaptor is None:
            out = self.base_layer(hidden_states)
            return out
        else:
            out = self.base_layer(hidden_states) + self.adaptor(hidden_states)
            return out