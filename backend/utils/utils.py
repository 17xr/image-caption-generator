import torch
import torch.nn.functional as F


def format_cap(c):
    c = str(c).strip().capitalize()
    return c if c.endswith((".", "!", "?")) else f"{c}."


def nucleus_sampling_generate(
    model, image_tensor, tokenizer, device, style_configs, max_length
):
    all_style_tokens = []
    all_styles = []
    ngram_sizes = []
    rep_penalties = []
    temps = []
    top_ks = []
    top_ps = []

    style_tokens = {
        "Concise": "[CONCISE]",
        "Narrative": "[NARRATIVE]",
        "Descriptive": "[DESCRIPTIVE]",
    }

    for style, config in style_configs.items():
        count = config.get("num", 1)

        all_style_tokens.extend([style_tokens[style]] * count)
        all_styles.extend([style] * count)
        temps.extend([config.get("temp", 1.0)] * count)
        top_ks.extend([config.get("top_k", 0)] * count)
        top_ps.extend([config.get("top_p", 1.0)] * count)
        rep_penalties.extend([config.get("rep_penalty", 1.0)] * count)
        ngram_sizes.extend([config.get("ngram", 0)] * count)

    total_batch_size = len(all_style_tokens)

    temps = torch.tensor(temps, device=device).view(-1, 1)
    rep_penalties = torch.tensor(rep_penalties, device=device).view(-1, 1)
    top_ks = torch.tensor(top_ks, device=device)
    top_ps = torch.tensor(top_ps, device=device)

    style_ids = tokenizer(
        all_style_tokens, add_special_tokens=False, return_tensors="pt", padding=True
    ).input_ids.to(device)

    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    image_features = image_tensor.expand(total_batch_size, -1, -1, -1).to(device)

    start_token = tokenizer.cls_token_id
    end_token = tokenizer.sep_token_id

    curr_seq = torch.full(
        (total_batch_size, 1), start_token, device=device, dtype=torch.long
    )
    unfinished_sequences = torch.ones(total_batch_size, device=device, dtype=torch.long)
    past_key_values = None

    for _ in range(max_length):
        with torch.no_grad():
            model_input = curr_seq if past_key_values is None else curr_seq[:, -1:]
            logits, past_key_values = model(
                image_features,
                model_input,
                style_ids=style_ids,
                past_key_values=past_key_values,
            )
            next_token_logits = logits[:, -1, :] / temps.clamp(min=1e-8)

        score = torch.gather(next_token_logits, 1, curr_seq)
        score = torch.where(score < 0, score * rep_penalties, score / rep_penalties)
        next_token_logits.scatter_(1, curr_seq, score)

        for i in range(total_batch_size):
            n = ngram_sizes[i]
            if n > 0 and curr_seq.size(1) >= n - 1:
                tokens = curr_seq[i].tolist()
                ngram_prefix = tokens[-(n - 1) :]
                for j in range(len(tokens) - n + 1):
                    if tokens[j : j + n - 1] == ngram_prefix:
                        next_token_logits[i, tokens[j + n - 1]] = -float("inf")

            k = int(top_ks[i])
            if k > 0:
                indices_to_remove = (
                    next_token_logits[i] < torch.topk(next_token_logits[i], k)[0][-1]
                )
                next_token_logits[i, indices_to_remove] = -float("inf")

            p = top_ps[i]
            if 0.0 < p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits[i], descending=True
                )
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    0, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[i, indices_to_remove] = -float("inf")

        probs = F.softmax(next_token_logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

        next_tokens = next_tokens * unfinished_sequences + end_token * (
            1 - unfinished_sequences
        )
        curr_seq = torch.cat([curr_seq, next_tokens.unsqueeze(1)], dim=1)
        unfinished_sequences = unfinished_sequences & (next_tokens != end_token).long()

        if unfinished_sequences.max() == 0:
            break

    results = {key: [] for key in style_tokens.keys()}
    for i, seq in enumerate(curr_seq):
        caption = tokenizer.decode(seq, skip_special_tokens=True).strip()
        raw_style = all_styles[i]

        results[raw_style].append(format_cap(caption))

    return results
