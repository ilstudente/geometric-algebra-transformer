# This file requires 'torch' and 'numpy' to be installed in your Python environment.
# If you see linter errors for these imports, ensure your environment has these packages.
import torch
from torch import nn
from typing import Optional, Tuple, Union
import numpy as np

# Import GMConvCls from the GMCNN package
from GMCNN.gmcnn.gm_convolution.gmconv_classification import GMConvCls

class GMCNNLinear(nn.Module):
    """
    GM-CNN-based linear layer for multivector tensors, matching the interface of EquiLinear.

    This layer applies a GMConvCls operation to the input multivectors, using group-equivariant convolution
    instead of the standard equivariant linear map. The interface is designed to be compatible with EquiLinear
    for easy swapping in GATr architectures.

    Parameters
    ----------
    in_mv_channels : int
        Input multivector channels
    out_mv_channels : int
        Output multivector channels
    group : str
        The group type (e.g., 'cyclic', 'dihedral')
    order : int
        The order of the group
    nbr_size : int
        The size of the neighborhood
    group_matrix : np.ndarray
        The group matrix defining the group structure
    error : bool
        Whether to enable error correction in GMConvCls
    """
    def __init__(
        self,
        in_mv_channels: int,
        out_mv_channels: int,
        group: str,
        order: int,
        nbr_size: int,
        group_matrix: np.ndarray,
        error: bool = False,
    ) -> None:
        super().__init__()
        self.in_mv_channels = in_mv_channels
        self.out_mv_channels = out_mv_channels
        self.group = group
        self.order = order
        self.nbr_size = nbr_size
        self.group_matrix = group_matrix
        self.error = error

        # Instantiate the GMConvCls layer
        self.gmconv = GMConvCls(
            group=group,
            order=order,
            nbr_size=nbr_size,
            group_matrix=group_matrix,
            out_channels=out_mv_channels,
            error=error,
        )

    def forward(
        self,
        multivectors: torch.Tensor,
        scalars: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """
        Applies the GMConvCls operation to the input multivectors.

        Parameters
        ----------
        multivectors : torch.Tensor
            Input multivectors of shape (batch_size, in_channels, 1, vec_size)
        scalars : Optional[torch.Tensor]
            Optional input scalars (ignored in this implementation)

        Returns
        -------
        outputs_mv : torch.Tensor
            Output multivectors after GMConvCls
        outputs_s : None
            No scalar outputs (for compatibility with EquiLinear)
        """
        # The input is expected to be (batch_size, in_channels, 1, vec_size)
        # If input shape is (..., in_channels, 16), reshape to (batch, in_channels, 1, 16)
        if multivectors.dim() == 3 and multivectors.shape[-1] == self.order:
            # Already in (batch, in_channels, order)
            multivectors = multivectors.unsqueeze(2)  # (batch, in_channels, 1, order)
        elif multivectors.dim() == 4:
            # Assume already correct shape
            pass
        else:
            raise ValueError(f"Unexpected input shape for multivectors: {multivectors.shape}")

        outputs_mv = self.gmconv(multivectors)
        return outputs_mv, None 