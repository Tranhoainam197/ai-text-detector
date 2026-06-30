import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

_MODEL_NAME = "gpt2"  # small model; fine for CPU


class PerplexityScorer:
    def __init__(self, model_name: str = _MODEL_NAME):
        # On Mac, prefer MPS (Apple Silicon GPU) if available, else CPU
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score(self, text: str, stride: int = 512, max_length: int = 1024) -> float:
        """
        Sliding-window perplexity. Lower = more LM-like (more AI-leaning).
        """
        encodings = self.tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids.to(self.device)
        seq_len = input_ids.size(1)
        if seq_len < 2:
            return float("inf")

        nlls = []
        prev_end = 0
        for begin in range(0, seq_len, stride):
            end = min(begin + max_length, seq_len)
            trg_len = end - prev_end
            chunk = input_ids[:, begin:end]
            target_ids = chunk.clone()
            target_ids[:, :-trg_len] = -100
            outputs = self.model(chunk, labels=target_ids)
            nlls.append(outputs.loss * trg_len)
            prev_end = end
            if end == seq_len:
                break

        avg_nll = torch.stack(nlls).sum() / seq_len
        return float(torch.exp(avg_nll).item())


# Singleton — load model once, reuse across requests
_scorer = None


def get_scorer() -> PerplexityScorer:
    global _scorer
    if _scorer is None:
        _scorer = PerplexityScorer()
    return _scorer