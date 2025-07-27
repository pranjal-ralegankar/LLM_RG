import torch
from transformers import GPT2LMHeadModel, GemmaForCausalLM

# Your provided function to create the sliding window mask
def create_sliding_window_causal_mask(sequence_length, window_size, device="cpu"):
    """
    Creates a causal mask with a sliding window attention.
    Each token can only attend to the 'window_size' previous tokens.
    """
    # Start with a tensor of all zeros
    mask = torch.zeros(sequence_length, sequence_length, device=device)

    # Create a band of ones for the sliding window
    for i in range(sequence_length):
        start_index = max(0, i - window_size + 1)
        mask[i, start_index:i+1] = 1

    # The mask should be float and have a large negative value for masked positions
    masked_mask = (1.0 - mask) * -1e9
    
    # Return in the format expected by Hugging Face: [batch_size, num_heads, seq_len, seq_len]
    # We use [1, 1, seq_len, seq_len] and let it broadcast
    return masked_mask.unsqueeze(0).unsqueeze(0)

# 1. Create the Custom Model Class for GPT2 🧠
class GPT2WithSlidingWindow(GPT2LMHeadModel):
    def __init__(self, config, window):
        super().__init__(config)
        self.config.window_size = window

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        **kwargs,
    ):
        # 1. Get sequence lengths
        new_token_len = input_ids.shape[1]
        past_len = 0
        if past_key_values is not None and len(past_key_values) > 0:
            # The length of the past is stored in the shape of the KV cache tensors
            past_len = past_key_values[0][0].shape[2]
        
        total_len = past_len + new_token_len

        # 2. Create the full sliding window mask for the entire sequence
        # This works for both the initial prompt and subsequent generation steps
        sliding_window_mask = create_sliding_window_causal_mask(
            sequence_length=total_len,
            window_size=self.config.window_size,
            device=input_ids.device
        )
        
        # 3. CRITICAL: Slice the mask
        # When using the KV cache, the model only needs the attention mask
        # for the *new* tokens (queries). We slice the full mask to get
        # the last `new_token_len` rows.
        final_attention_mask = sliding_window_mask[:, :, -new_token_len:, :]
        
        # The original forward pass can now use our correctly shaped mask
        return super().forward(
            input_ids=input_ids,
            past_key_values=past_key_values,
            # Pass our corrected and sliced mask
            attention_mask=final_attention_mask,
            **kwargs,
        )

# 2. Create the Custom Model Class for Gemmma 🧠
class GemmaWithSlidingWindow(GemmaForCausalLM):
    def __init__(self, config, window):
        super().__init__(config)
        self.config.window_size = window

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        **kwargs,
    ):
        # 1. Get sequence lengths
        new_token_len = input_ids.shape[1]
        past_len = 0
        if past_key_values is not None and len(past_key_values) > 0:
            # The length of the past is stored in the shape of the KV cache tensors
            past_len = past_key_values[0][0].shape[2]
        
        total_len = past_len + new_token_len

        # 2. Create the full sliding window mask for the entire sequence
        # This works for both the initial prompt and subsequent generation steps
        sliding_window_mask = create_sliding_window_causal_mask(
            sequence_length=total_len,
            window_size=self.config.window_size,
            device=input_ids.device
        )
        
        # 3. CRITICAL: Slice the mask
        # When using the KV cache, the model only needs the attention mask
        # for the *new* tokens (queries). We slice the full mask to get
        # the last `new_token_len` rows.
        sliding_window_mask2 = sliding_window_mask[:, :, -new_token_len:, :]

        # 4. CRITICAL: Combine masks
        # The trainer's `attention_mask` is [batch_size, seq_len] with 1s for real tokens and 0s for padding.
        # We need to expand it to match the dimensions of our sliding mask for combining.
        if attention_mask is not None:
            # Expand from [B, S] to [B, 1, S, S]
            expanded_padding_mask = self.model.get_extended_attention_mask(attention_mask, input_ids.shape)
            
            # Add the masks together. The large negative values will dominate where either mask is restrictive.
            final_attention_mask = sliding_window_mask2 + expanded_padding_mask
        else:
            final_attention_mask = sliding_window_mask2
        
        # The original forward pass can now use our correctly shaped mask
        return super().forward(
            input_ids=input_ids,
            past_key_values=past_key_values,
            # Pass our corrected and sliced mask
            attention_mask=final_attention_mask,
            **kwargs,
        )


class LoRACbCausalLM(torch.nn.Module):
    def __init__(self, base_model, window_size: int):
        super().__init__()
        self.model = base_model
        self.config = base_model.config
        self.config.window_size = window_size
        self.window_size = window_size

        # if masking is enabled, pre-build mask buffer
        if window_size > 0:
            max_L = 512 + window_size
            buf   = create_sliding_window_causal_mask(max_L, window_size, device="cuda")
            self.register_buffer("cb_mask_full", buf, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, labels=None, **kwargs):
        if labels is None:
            labels = input_ids.clone()

        # if no masking, delegate directly
        if self.window_size <= 0:
            return self.model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                labels=labels,
                **kwargs
            )

        # else apply sliding-window C_b
        new_len  = input_ids.size(1)
        past_len = past_key_values[0][0].size(2) if past_key_values else 0
        total_len = past_len + new_len

        full_mask  = self.cb_mask_full[..., :total_len, :total_len]
        final_mask = full_mask[:, :, -new_len:, :]

        return self.model.forward(
            input_ids=input_ids,
            attention_mask=final_mask,
            past_key_values=past_key_values,
            labels=labels,
            **kwargs
        )