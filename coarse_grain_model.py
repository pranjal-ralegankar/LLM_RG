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
    def __init__(self, config):
        super().__init__(config)
        # Ensure window_size is stored in the config
        if not hasattr(self.config, "window_size"):
            # Set a default if not provided
            self.config.window_size = 128 
            print(f"Warning: 'window_size' not found in config. Defaulting to {self.config.window_size}.")

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # 2. Generate the sliding window mask
        sliding_window_mask = create_sliding_window_causal_mask(
            sequence_length=seq_len,
            window_size=self.config.window_size,
            device=device
        ) # Shape: [1, 1, seq_len, seq_len]

        # 3. Combine with the padding mask (if it exists)
        # The 'attention_mask' from the data collator is for padding.
        # It's shape is [batch_size, seq_len]
        if attention_mask is not None:
            # Reshape padding mask for broadcasting: [batch_size, 1, 1, seq_len]
            # and convert to the additive format (0 for attend, -1e9 for don't attend)
            extended_padding_mask = (1.0 - attention_mask[:, None, None, :]) * -1e9
            # Add the masks together. Broadcasting handles the dimensions.
            # The final mask will prevent attention to both padded tokens and out-of-window tokens.
            final_attention_mask = sliding_window_mask + extended_padding_mask
        else:
            final_attention_mask = sliding_window_mask

        # 4. Call the original forward method with our new mask
        return super().forward(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=final_attention_mask, # <-- Our custom mask is injected here!
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
