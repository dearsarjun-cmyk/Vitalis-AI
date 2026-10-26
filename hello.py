"""
VITALIS v3 — from-scratch language-model training prototype.

Install:
    uv add torch tokenizers

First train:
    uv run python hello.py train --steps 2000

Continue training:
    uv run python hello.py train --resume --steps 2000

Chat:
    uv run python hello.py chat

Add permitted UTF-8 text files under data/corpus/ before training.
This is an educational prototype, not a reliable medical system.
It does not load a pretrained language model.
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"
TOKENIZER_PATH = ROOT / "tokenizer.json"
CHECKPOINT_PATH = ROOT / "vitalis_model.pt"
LATEST_PATH = ROOT / "vitalis_latest.pt"
SPECIAL = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_corpus():
    """Read every UTF-8 .txt file in data/corpus, including subfolders."""
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(CORPUS_DIR.rglob("*.txt"))
    if not files:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        bootstrap = DATA_DIR / "bootstrap.txt"
        if not bootstrap.exists():
            bootstrap.write_text(
                (
                    "A language model predicts the next token from previous tokens. "
                    "A neural network learns by adjusting weights to reduce prediction loss. "
                    "Biology studies living organisms, cells, genes, and proteins. "
                    "Scientific claims should be supported by reliable evidence. "
                    "A small corpus cannot teach broad language ability. "
                    "中文：生物学研究生命和细胞。 "
                    "हिन्दी: जीव विज्ञान जीवन का अध्ययन है। "
                    "Español: La biología estudia los seres vivos. "
                    "العربية: يدرس علم الأحياء الكائنات الحية. "
                    "Français : La biologie étudie les êtres vivants. "
                    "தமிழ்: உயிரியல் உயிரினங்களை ஆய்வு செய்கிறது。\n\n"
                ) * 200,
                encoding="utf-8",
            )
        files = [bootstrap]
        print("WARNING: using tiny bootstrap text, only to test the pipeline.")

    docs = []
    for path in files:
        try:
            content = path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError as exc:
            print(f"Skipping {path}: {exc}")
            continue
        if content:
            docs.append(content)

    if not docs:
        raise RuntimeError("No readable text files found under data/corpus/.")

    chars = sum(len(d) for d in docs)
    print(f"Loaded {len(docs)} files | {chars:,} characters")
    return files, docs


def train_tokenizer(files, vocab_size: int) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train([str(p) for p in files], trainer)
    tokenizer.save(str(TOKENIZER_PATH))
    print(f"Saved tokenizer with {tokenizer.get_vocab_size():,} tokens")
    return tokenizer


class Brain(nn.Module):
    """Causal Transformer language model. All weights start randomly initialized."""

    def __init__(
        self,
        vocab_size: int,
        width: int = 192,
        heads: int = 4,
        layers: int = 4,
        context: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.context = context
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.position_embedding = nn.Embedding(context, width)

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=4 * width,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.layers = nn.TransformerEncoder(
            transformer_layer,
            num_layers=layers,
        )
        self.final_norm = nn.LayerNorm(width)
        self.lm_head = nn.Linear(width, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, token_ids):
        _, seq_len = token_ids.shape
        if seq_len > self.context:
            raise ValueError(f"Sequence length {seq_len} exceeds context {self.context}")
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        causal_mask = torch.triu(
            torch.ones(
                seq_len, seq_len,
                device=token_ids.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        x = self.layers(x, mask=causal_mask)
        return self.lm_head(self.final_norm(x))


def encode_documents(tokenizer, docs):
    bos = tokenizer.token_to_id("[BOS]")
    eos = tokenizer.token_to_id("[EOS]")
    ids = []
    for doc in docs:
        ids.append(bos)
        ids.extend(tokenizer.encode(doc).ids)
        ids.append(eos)
    return ids


def get_batch(data, batch_size, context, device):
    max_start = len(data) - context - 1
    if max_start <= 0:
        raise ValueError("Dataset too short for this context size.")
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[s:s + context] for s in starts]).to(device)
    y = torch.stack([data[s + 1:s + context + 1] for s in starts]).to(device)
    return x, y


@torch.no_grad()
def evaluate_loss(brain, data, vocab_size, context, batch_size, batches, device):
    if len(data) <= context + 1:
        return float("nan")
    was_training = brain.training
    brain.eval()
    losses = []
    for _ in range(batches):
        x, y = get_batch(data, batch_size, context, device)
        logits = brain(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        losses.append(loss.item())
    if was_training:
        brain.train()
    return sum(losses) / len(losses)


def save_checkpoint(brain, optimizer, config, step, path):
    # Keep checkpoint tensors on CPU so it can be loaded on CPU-only machines.
    model_state = {k: v.detach().cpu() for k, v in brain.state_dict().items()}
    optimizer_state = optimizer.state_dict()
    torch.save(
        {
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer_state,
            "config": config,
            "step": step,
        },
        path,
    )


def train(args):
    seed_all(args.seed)
    files, docs = collect_corpus()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    previous_step = 0
    saved = None

    if args.resume:
        if not TOKENIZER_PATH.exists() or not LATEST_PATH.exists():
            raise FileNotFoundError(
                "No resumable checkpoint found. Run training without --resume first."
            )
        tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
        saved = torch.load(LATEST_PATH, map_location="cpu", weights_only=True)
        config = saved["config"]
        previous_step = saved["step"]
        print(f"Resuming from training step {previous_step}.")
    else:
        tokenizer = train_tokenizer(files, args.vocab_size)
        config = {
            "vocab_size": tokenizer.get_vocab_size(),
            "width": args.width,
            "heads": args.heads,
            "layers": args.layers,
            "context": args.context,
            "dropout": args.dropout,
        }

    ids = encode_documents(tokenizer, docs)
    if len(ids) < config["context"] * 2 + 4:
        raise ValueError(
            "Not enough tokens. Add more training text or reduce --context."
        )

    # Split sequentially into train/validation so the reported validation loss
    # is measured on tokens not sampled from the training segment.
    split = int(len(ids) * 0.95)
    train_data = torch.tensor(ids[:split], dtype=torch.long)
    val_data = torch.tensor(ids[split:], dtype=torch.long)
    if len(val_data) <= config["context"] + 1:
        val_data = train_data[-min(len(train_data), config["context"] * 4):]

    brain = Brain(**config)
    optimizer = torch.optim.AdamW(
        brain.parameters(), lr=args.lr, weight_decay=0.01
    )

    if saved is not None:
        brain.load_state_dict(saved["model_state_dict"])
        if "optimizer_state_dict" in saved:
            optimizer.load_state_dict(saved["optimizer_state_dict"])

    brain.to(device)
    train_data = train_data
    val_data = val_data
    print(f"Device: {device}")
    print(f"Train tokens: {len(train_data):,} | Validation tokens: {len(val_data):,}")
    print(f"Brain parameters: {sum(p.numel() for p in brain.parameters()):,}")
    print(f"Additional optimizer steps: {args.steps}")

    optimizer.zero_grad(set_to_none=True)
    brain.train()

    for local_step in range(1, args.steps + 1):
        global_step = previous_step + local_step

        # Linear warmup, then cosine decay.
        if global_step <= args.warmup_steps:
            lr_scale = global_step / max(1, args.warmup_steps)
        else:
            progress = min(
                1.0,
                (global_step - args.warmup_steps) /
                max(1, args.total_steps - args.warmup_steps),
            )
            lr_scale = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

        for group in optimizer.param_groups:
            group["lr"] = args.lr * lr_scale

        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0

        for _ in range(args.grad_accum):
            x, y = get_batch(
                train_data, args.batch_size, config["context"], device
            )
            logits = brain(x)
            loss = F.cross_entropy(
                logits.reshape(-1, config["vocab_size"]),
                y.reshape(-1),
            )
            (loss / args.grad_accum).backward()
            accumulated_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
        optimizer.step()

        if (
            local_step == 1
            or global_step % args.log_every == 0
            or local_step == args.steps
        ):
            val_loss = evaluate_loss(
                brain, val_data, config["vocab_size"], config["context"],
                args.batch_size, args.eval_batches, device,
            )
            print(
                f"step {global_step:>7} | "
                f"train loss {accumulated_loss / args.grad_accum:.4f} | "
                f"val loss {val_loss:.4f} | lr {optimizer.param_groups[0]['lr']:.2e}"
            )

        if args.save_every > 0 and global_step % args.save_every == 0:
            save_checkpoint(
                brain, optimizer, config, global_step, LATEST_PATH
            )
            print(f"Checkpoint saved at step {global_step}.")

    save_checkpoint(
        brain, optimizer, config, previous_step + args.steps, LATEST_PATH
    )
    save_checkpoint(
        brain, optimizer, config, previous_step + args.steps, CHECKPOINT_PATH
    )
    print(f"Saved model: {CHECKPOINT_PATH}")
    print(f"Saved resumable state: {LATEST_PATH}")
    if sum(map(len, docs)) < 100_000:
        print(
            "DATA WARNING: under 100,000 characters loaded. "
            "A larger, diverse corpus matters far more than endlessly repeating tiny data."
        )


@torch.no_grad()
def generate(brain, tokenizer, prompt, max_new_tokens=120, temperature=0.8, top_k=40):
    brain.eval()
    device = next(brain.parameters()).device
    ids = [tokenizer.token_to_id("[BOS]"), *tokenizer.encode(prompt).ids]
    generated = []
    eos = tokenizer.token_to_id("[EOS]")

    for _ in range(max_new_tokens):
        x = torch.tensor([ids[-brain.context:]], dtype=torch.long, device=device)
        logits = brain(x)[0, -1] / max(temperature, 1e-5)

        # Mild repetition penalty on recently generated tokens.
        for token_id in set(generated[-32:]):
            logits[token_id] /= 1.15

        if top_k > 0:
            k = min(top_k, logits.numel())
            values, _ = torch.topk(logits, k)
            logits[logits < values[-1]] = -float("inf")

        next_id = torch.multinomial(F.softmax(logits, dim=-1), 1).item()
        if next_id == eos:
            break
        ids.append(next_id)
        generated.append(next_id)

    return tokenizer.decode(generated, skip_special_tokens=True)


def chat(args):
    if not TOKENIZER_PATH.exists() or not CHECKPOINT_PATH.exists():
        print("Train first: uv run python hello.py train")
        return

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    saved = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    config = saved["config"]
    brain = Brain(**config)
    brain.load_state_dict(saved["model_state_dict"])
    brain.eval()

    print("VITALIS v3 — experimental from-scratch language model")
    print("Type /quit to exit. Not suitable for medical decisions.")
    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt.lower() in {"/quit", "quit", "exit"}:
            break
        if prompt:
            print("VITALIS:", generate(
                brain, tokenizer, prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            ))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train")
    t.add_argument("--steps", type=int, default=1000)
    t.add_argument("--batch-size", type=int, default=4)
    t.add_argument("--grad-accum", type=int, default=4)
    t.add_argument("--context", type=int, default=128)
    t.add_argument("--vocab-size", type=int, default=4096)
    t.add_argument("--width", type=int, default=192)
    t.add_argument("--heads", type=int, default=4)
    t.add_argument("--layers", type=int, default=4)
    t.add_argument("--dropout", type=float, default=0.1)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--warmup-steps", type=int, default=100)
    t.add_argument("--total-steps", type=int, default=5000)
    t.add_argument("--log-every", type=int, default=50)
    t.add_argument("--save-every", type=int, default=500)
    t.add_argument("--eval-batches", type=int, default=5)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--resume", action="store_true")
    t.set_defaults(func=train)

    c = sub.add_parser("chat")
    c.add_argument("--max-new-tokens", type=int, default=120)
    c.add_argument("--temperature", type=float, default=0.8)
    c.add_argument("--top-k", type=int, default=40)
    c.set_defaults(func=chat)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
