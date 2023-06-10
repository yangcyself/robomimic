import math
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
from typing import Optional, Dict
import uuid

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, kdim=None,vdim=None,
                 dropout=0.0, bias=True):
        super().__init__()
        self.id = str(uuid.uuid4())  # This creates a unique UUID for this instance
        self.embed_dim = embed_dim 
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.num_heads = num_heads
        self.dropout_module = nn.Dropout(dropout)
        self.head_dim = embed_dim//num_heads
        assert self.head_dim*self.num_heads==self.embed_dim, "head_dim*num_heads must equal to embed_dim"
        self.scaling = self.head_dim**(-0.5)
        self.k_proj = nn.Linear(self.kdim,self.embed_dim,bias=bias)
        self.v_proj = nn.Linear(self.vdim,self.embed_dim,bias=bias)
        self.q_proj = nn.Linear(self.embed_dim,self.embed_dim,bias=bias)
        self.out_proj = nn.Linear(self.embed_dim,self.embed_dim,bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.)

    def get_input_buffer(self, incremental_state):
        if self.id in incremental_state:
            return incremental_state[self.id]
        else:
            incremental_state[self.id]=None
            return None

    def set_input_buffer(self, incremental_state, saved_state):
        incremental_state[self.id] = saved_state
        return incremental_state
        
    def forward(self,query,key:Optional[Tensor]=None,value:Optional[Tensor]=None,
            key_padding_mask: Optional[Tensor]=None,attn_mask:Optional[Tensor]=None,
            incremental_state=None):
        tgt_len, bsz, embed_dim = query.size()
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        q *= self.scaling
        q = q.contiguous().view(tgt_len,bsz*self.num_heads, self.head_dim).transpose(0,1)
        k = k.contiguous().view(-1,bsz*self.num_heads, self.head_dim).transpose(0,1)
        v = v.contiguous().view(-1,bsz*self.num_heads, self.head_dim).transpose(0,1)
        if incremental_state is not None:
            saved_state = self.get_input_buffer(incremental_state)
            if saved_state is not None:
                # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
                (prev_key,prev_value) = saved_state
                prev_key = prev_key.view(bsz * self.num_heads, -1, self.head_dim)
                k = torch.cat([prev_key, k], dim=1)
                prev_value = prev_value.view(bsz * self.num_heads, -1, self.head_dim)
                v = torch.cat([prev_value, v], dim=1)
                assert k is not None and v is not None

                saved_state = (k.view(bsz, self.num_heads, -1, self.head_dim),
                                 v.view(bsz, self.num_heads, -1, self.head_dim))
                incremental_state = self.set_input_buffer(incremental_state, saved_state)

        src_len = k.size(1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))

        #attention mask is none is incremental_state is not none
        if attn_mask is not None:
           attn_mask = attn_mask.unsqueeze(0)
           attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf")
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights_float = F.softmax(attn_weights, dim=-1,dtype=torch.float32)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)
        attn = torch.bmm(attn_probs, v)
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        return attn, attn_weights

    def reorder_incremental_state(
        self, incremental_state: Dict[str, Dict[str, Optional[Tensor]]], new_order: Tensor
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self.get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    if self.encoder_decoder_attention and input_buffer_k.size(0) == new_order.size(0):
                        break
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self.set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                # in_proj_weight used to be q + k + v with same dimensions
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim : 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim :]

                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
                        dim : 2 * dim
                    ]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim :]

                    keys_to_remove.append(prefix + "in_proj_bias")

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value


        
