# PyTorch base package: Math and Tensor Stuff, Neural Network layers
import torch
# Quantized versions of PyTorch components
import brevitas.nn
# Tensor operations in einstein notation
from einops import rearrange


# Quantized version of MultiheadAttention similar to PyTorch and Brevitas with
# configuration options more suitable to our experiments and an alternative
# export path expressing attention heads via splits and branching instead of
# rearranging into the batch dimension
class QuantMultiheadAttention(torch.nn.Module):
    def __init__(
            self,
            # Size of the embedding dimension
            emb_dim,
            # Number of attention heads to distribute the embeddings to
            num_heads,
            # Enable a bias added to the input and output projections
            bias=False,
            # Amount of dropout to apply at the attention block output, i.e.,
            # after the output projection, during training
            dropout=0.0,

            # Implements attention heads in parallel by splitting instead of
            # rearranging heads into the batch dimension
            split_heads=False,

            # Input, weight and bias quantization settings of input projections,
            # shared by all three input projections
            input_projection_input_quant=None,
            input_projection_weight_quant=None,
            input_projection_bias_quant=None,

            # Quantization settings of key, query and values tensors, i.e., the
            # outputs of the input projection
            k_quant=None,
            q_quant=None,
            v_quant=None,

            # Input and output quantization of the softmax normalization of the
            # attention weights
            softmax_input_quant=None,
            softmax_output_quant=None,

            # Input, weight and bias quantization settings of output projection
            output_projection_input_quant=None,
            output_projection_weight_quant=None,
            output_projection_bias_quant=None,

            # Output quantizer of the whole attention operation following the
            # output projection
            output_quant=None,

            # Return the quantization parameters so the next layer can
            # quantize the bias
            return_quant_tensor=False
    ):
        super().__init__()

        # (Quantized) Input projections: Linear projections with optional bias
        self.q_projection = brevitas.nn.QuantLinear(
            emb_dim,
            emb_dim,
            bias=bias,
            input_quant=input_projection_input_quant,
            weight_quant=input_projection_weight_quant,
            bias_quant=input_projection_bias_quant,
            output_quant=q_quant,
            return_quant_tensor=False
        )

        self.k_projection = brevitas.nn.QuantLinear(
            emb_dim,
            emb_dim,
            bias=bias,
            input_quant=input_projection_input_quant,
            weight_quant=input_projection_weight_quant,
            bias_quant=input_projection_bias_quant,
            output_quant=k_quant,
            return_quant_tensor=False
        )

        self.v_projection = brevitas.nn.QuantLinear(
            emb_dim,
            emb_dim,
            bias=bias,
            input_quant=input_projection_input_quant,
            weight_quant=input_projection_weight_quant,
            bias_quant=input_projection_bias_quant,
            output_quant=v_quant,
            return_quant_tensor=False
        )

        # (Quantized) Output projection: Linear projection with optional bias
        self.o_projection = brevitas.nn.QuantLinear(
            emb_dim,
            emb_dim,
            bias=bias,
            input_quant=output_projection_input_quant,
            weight_quant=output_projection_weight_quant,
            bias_quant=output_projection_bias_quant,
            output_quant=output_quant,
            return_quant_tensor=return_quant_tensor and output_quant is not None
        )

        # Quantized softmax normalization of the attention weights
        self.softmax = torch.nn.Sequential(
            brevitas.nn.QuantIdentity(softmax_input_quant),
            torch.nn.Softmax(dim=-1),
            brevitas.nn.QuantIdentity(softmax_output_quant),
            # Dropout is applied to normalized attention weights
            torch.nn.Dropout(p=dropout)
        )

        self.split_heads = split_heads
        self.num_heads = num_heads
        self.embed_dim = emb_dim

    def forward(self, query, key, value, mask=None):
        # Apply input projections to query, key and value tensors
        query = self.q_projection(query)
        key = self.k_projection(key)
        value = self.v_projection(value)

        # Scale of the scaled dot-product related to the embedding dimension
        scale = self.embed_dim ** -0.5

        # No mask means all zeros mask
        if mask is None:
            mask = 0

        # Collect outputs of the attention heads as a list of tensors to be
        # concatenated - for batched heads, this is a single entry list
        heads = []
        # Number of heads and embedding dimension per head
        h, d = self.num_heads, self.embed_dim // self.num_heads

        # There are two styles of implementing the attention heads: Via the
        # batch dimension (most common) or via split/concat (for our backend)
        if self.split_heads:
            # Split query, key and value tensors along the embedding dimension
            # to have the heads as a list of tensors
            queries = torch.split(query, query.shape[-1] // h, dim=-1)
            keys = torch.split(key, key.shape[-1] // h, dim=-1)
            values = torch.split(value, value.shape[-1] // h, dim=-1)
            # Process each head individually...
            for query, key, value in zip(queries, keys, values):
                # scale * (query @ key.T) = (scale * query) @ key.T
                query = scale * query
                # Transpose embedding and sequence dimension of the key tensor
                key = key.transpose(-1, -2)
                # Scaled dot product of query and key, masked and normalized to
                # get the attention weights
                weights = self.softmax(torch.matmul(query, key) + mask)
                # Multiply attention weights to the values and rearrange the
                # heads back into the embedding dimension
                heads.append(torch.matmul(weights, value))
        else:
            # Rearrange the query, key and value tensors to have heads as part
            # of the batch dimension
            query = rearrange(query, "b s (h d) -> b h s d", h=h, d=d)
            key = rearrange(key, "b s (h d) -> b h d s", h=h, d=d)
            value = rearrange(value, "b s (h d) -> b h s d", h=h, d=d)
            # scale * (query @ key.T) = (scale * query) @ key.T
            query = scale * query
            # Scaled dot product of query and key, masked and normalized to get
            # the attention weights
            weights = self.softmax(torch.matmul(query, key) + mask)
            # Multiply attention weights to the values and rearrange the heads
            # back into the embedding dimension
            heads = [
                rearrange(torch.matmul(weights, value), "b h s d -> b s (h d)")
            ]

        # Apply output projection to the product of attention weights and values
        return self.o_projection(torch.concat(heads, dim=-1))
