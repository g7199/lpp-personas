"""Loader for the FP8 (compressed-tensors) checkpoint of Gemma-4-26B-A4B-it. The checkpoint stores experts one by one, while
transformers expects fused 3D expert tensors, so the expert modules are replaced by FP8Experts, which keep the FP8 weights and
per-channel scales on the GPU (~23GB over two 24GB GPUs) and dequantize each routed expert to bf16 on the fly. The other linear
layers are dequantized to bf16 at load time, so all activations stay in bf16."""
import os, torch, torch.nn as nn, json
from safetensors import safe_open
from transformers.models.gemma4 import modeling_gemma4 as mg
from transformers.activations import ACT2FN


class FP8Experts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts; self.act_fn = ACT2FN[config.hidden_activation]
        self.gate_up_proj = nn.Parameter(torch.empty(0)); self.down_proj = nn.Parameter(torch.empty(0))  # placeholders for HF init
        self.gu = self.gus = self.dn = self.dns = None  # filled after loading: fp8 [E,2I,H], bf16 [E,2I,1], fp8 [E,H,I], bf16 [E,H,1]
    def forward(self, hidden_states, top_k_index, top_k_weights):
        T, k = top_k_index.shape
        if T * k <= 128:  # decoding / short inputs: gather the routed experts and batch them, without host syncs
            x = hidden_states.unsqueeze(1).expand(T, k, -1).reshape(T * k, 1, -1); idx = top_k_index.reshape(-1)
            w1 = self.gu[idx].to(x.dtype) * self.gus[idx]; w2 = self.dn[idx].to(x.dtype) * self.dns[idx]
            g, u = torch.bmm(x, w1.transpose(1, 2)).chunk(2, dim=-1)
            h = torch.bmm(self.act_fn(g) * u, w2.transpose(1, 2)).reshape(T, k, -1)
            return (h * top_k_weights.unsqueeze(-1).to(h.dtype)).sum(1).to(hidden_states.dtype)
        # prefill: sort (token, slot) pairs by expert, one host sync for the group sizes, then one dense pass per routed expert
        flat = top_k_index.reshape(-1); order = torch.argsort(flat, stable=True); counts = torch.bincount(flat, minlength=self.num_experts).tolist()
        tok = order // k; xs = hidden_states[tok]; ws = top_k_weights.reshape(-1)[order]
        hs = torch.empty_like(xs); s0 = 0
        for e, c in enumerate(counts):
            if c == 0: continue
            x = xs[s0:s0 + c]
            g, u = nn.functional.linear(x, self.gu[e].to(x.dtype) * self.gus[e]).chunk(2, dim=-1)
            hs[s0:s0 + c] = nn.functional.linear(self.act_fn(g) * u, self.dn[e].to(x.dtype) * self.dns[e]); s0 += c
        out = torch.zeros_like(hidden_states)
        out.index_add_(0, tok, (hs * ws[:, None].to(hs.dtype)).to(out.dtype))
        return out

def load(repo, split=15):
    from huggingface_hub import snapshot_download
    PATH = repo if os.path.isdir(repo) else snapshot_download(repo)
    mg.Gemma4TextExperts = FP8Experts  # decoder layers look the class up at construction time
    from transformers import AutoModelForCausalLM
    cfg = json.load(open(PATH + '/config.json')); L = cfg['text_config']['num_hidden_layers']
    dmap = {'model.vision_tower': 0, 'model.embed_vision': 0, 'model.language_model.embed_tokens': 0, 'model.language_model.norm': 1,
            'model.language_model.rotary_emb': 0, 'lm_head': 0}
    for i in range(L): dmap[f'model.language_model.layers.{i}'] = 0 if i < split else 1
    try:
        m = AutoModelForCausalLM.from_pretrained(PATH, dtype=torch.bfloat16, device_map=dmap)
    except ValueError:  # device map missing a module: fall back to listing what exists
        raise
    # compressed-tensors keeps FP8 weights in the non-expert Linear layers and re-quantizes activations on every call (W8A8
    # emulation, ~10k kernel launches per token): replace them by plain bf16 Linear layers with dequantized weights
    for name, mod in list(m.model.language_model.named_modules()):
        for cn, ch in list(mod.named_children()):
            if isinstance(ch, nn.Linear) and ch.weight.dtype == torch.float8_e4m3fn:
                new = nn.Linear(ch.in_features, ch.out_features, bias=False, device=ch.weight.device, dtype=torch.bfloat16)
                new.weight.data = ch.weight.to(torch.bfloat16) * ch.weight_scale.to(torch.bfloat16); setattr(mod, cn, new)
    torch.cuda.empty_cache()
    f = safe_open(PATH + '/model.safetensors', 'pt')
    E = cfg['text_config']['num_experts']
    for i, layer in enumerate(m.model.language_model.layers):
        dev = torch.device('cuda:0' if i < split else 'cuda:1'); p = f'model.language_model.layers.{i}.experts.'
        ex = layer.experts
        ex.gu = torch.stack([torch.cat([f.get_tensor(p + f'{e}.gate_proj.weight'), f.get_tensor(p + f'{e}.up_proj.weight')]) for e in range(E)]).to(dev)
        ex.gus = torch.stack([torch.cat([f.get_tensor(p + f'{e}.gate_proj.weight_scale'), f.get_tensor(p + f'{e}.up_proj.weight_scale')]) for e in range(E)]).to(dev, torch.bfloat16)
        ex.dn = torch.stack([f.get_tensor(p + f'{e}.down_proj.weight') for e in range(E)]).to(dev)
        ex.dns = torch.stack([f.get_tensor(p + f'{e}.down_proj.weight_scale') for e in range(E)]).to(dev, torch.bfloat16)
    # accelerate's dispatch hooks break on this model after compressed-tensors decompression: replace them by plain hooks that move
    # every tensor argument to the device of the module that receives it
    from accelerate.hooks import remove_hook_from_module
    remove_hook_from_module(m, recurse=True)
    def mv(x, d):
        if torch.is_tensor(x): return x.to(d)
        if isinstance(x, tuple): return tuple(mv(y, d) for y in x)
        if isinstance(x, list): return [mv(y, d) for y in x]
        if isinstance(x, dict): return {k: mv(v, d) for k, v in x.items()}
        return x
    def hook(mod, args, kwargs):
        d = next(mod.parameters()).device; return mv(args, d), mv(kwargs, d)
    for mod in list(m.model.language_model.layers) + [m.model.language_model.norm, m.lm_head]:
        mod.register_forward_pre_hook(hook, with_kwargs=True)
    m.eval()
    return m
