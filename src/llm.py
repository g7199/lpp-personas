"""Scorer for open models: next-token probabilities over the answer-option digits (no sampling)."""
import torch, json, hashlib, time, os
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText
# Hugging Face model ids (override with LPP_MODEL_<NAME>, e.g. LPP_MODEL_QWEN3_8_27B=/path/to/local/copy)
MODELS = {
  'qwen3.8-27b': 'Qwen/Qwen3.8-27B',  # bf16 weights, loaded in 4-bit NF4 (one 24GB GPU)
  'gemma4-26b': 'RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic',  # FP8 checkpoint of google/gemma-4-26B-A4B-it, custom loader in gemma26.py (two GPUs)
}
MODELS = {k: os.environ.get('LPP_MODEL_' + k.upper().replace('.', '_').replace('-', '_'), v) for k, v in MODELS.items()}
QUANT4 = {'qwen3.8-27b'}  # 27B bf16 (~54GB) exceeds one 24GB GPU: bitsandbytes NF4 (double quantization, bf16 compute)
class Scorer:
    def __init__(self, name, device='cuda:1'):
        self.name=name; path=MODELS[name]
        self.tok=AutoTokenizer.from_pretrained(path); self.tok.padding_side='left'
        if self.tok.pad_token is None: self.tok.pad_token=self.tok.eos_token
        t=time.time(); kw={}
        if name in QUANT4:
            from transformers import BitsAndBytesConfig
            kw['quantization_config']=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
        if name == 'gemma4-26b':
            import gemma26; self.model = gemma26.load(path); self.load_s = time.time() - t; self.device = 'cuda:0'; return
        try:
            self.model=AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map=device, **kw)
        except Exception as e:
            print('CausalLM load failed, trying ImageTextToText:', str(e)[:200])
            self.model=AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, device_map=device, **kw)
        self.model.eval(); self.load_s=time.time()-t
        self.device=device if device!='auto' else 'cuda:0'
    def chat(self, user, system=None, prefill=''):
        if 'base' in self.name:  # base models: no chat template; plain text + answer cue
            return (system + '\n\n' if system else '') + user + '\nAnswer: ' + prefill
        msgs=([{'role':'system','content':system}] if system else [])+[{'role':'user','content':user}]
        kw={}
        if 'qwen' in self.name: kw['enable_thinking']=False
        s=self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
        return s+prefill
    def option_ids(self, labels):
        ids=[]
        for l in labels:
            t=self.tok.encode(l, add_special_tokens=False)
            ids.append(t[0])
        assert len(set(ids))==len(ids), (labels, ids)
        return ids
    @torch.no_grad()
    def probs(self, prompts, labels, bs=16):
        """prompts: list of fully formatted strings ending right before answer token."""
        ids=self.option_ids(labels); out=[]; mass=[]
        for i in range(0,len(prompts),bs):
            enc=self.tok(prompts[i:i+bs], return_tensors='pt', padding=True, add_special_tokens=False).to(self.device)
            logits=self.model(**enc, logits_to_keep=1).logits[:,-1,:].float()
            p=torch.softmax(logits,-1)
            sub=p[:,ids]; mass+= sub.sum(-1).tolist()
            out+= (sub/sub.sum(-1,keepdim=True)).tolist()
        return out, mass
    @torch.no_grad()
    def probs_prefix(self, prefix, suffixes, labels):
        """Score many suffixes sharing one long prefix: prefix encoded once, KV cache deep-copied per suffix.
        Token boundary: prefix and suffix are tokenized separately (caller must split at a newline)."""
        import copy
        ids = self.option_ids(labels)
        pre = self.tok(prefix, return_tensors='pt', add_special_tokens=False).input_ids.to(self.device)
        out = self.model(input_ids=pre, use_cache=True); cache = out.past_key_values; res = []; mass = []
        for sfx in suffixes:
            s_ids = self.tok(sfx, return_tensors='pt', add_special_tokens=False).input_ids.to(self.device)
            o = self.model(input_ids=s_ids, past_key_values=copy.deepcopy(cache), use_cache=True, logits_to_keep=1)
            p = torch.softmax(o.logits[0, -1].float(), -1)[ids]; mass.append(float(p.sum())); res.append((p / p.sum()).tolist())
        return res, mass
    @torch.no_grad()
    def generate(self, prompts, max_new_tokens=64, bs=8, temperature=0.0, seed=0):
        torch.manual_seed(seed); res=[]
        for i in range(0,len(prompts),bs):
            enc=self.tok(prompts[i:i+bs], return_tensors='pt', padding=True, add_special_tokens=False).to(self.device)
            g=self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=temperature>0, temperature=temperature if temperature>0 else None, top_p=None, top_k=None)
            res+= self.tok.batch_decode(g[:,enc['input_ids'].shape[1]:], skip_special_tokens=True)
        return res
def phash(s): return hashlib.md5(s.encode()).hexdigest()[:10]
