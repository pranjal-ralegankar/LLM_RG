import torch
from transformers import GPT2LMHeadModel

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

# 1. Create the Custom Model Class 🧠
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