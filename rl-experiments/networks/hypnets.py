import torch.nn as nn
import hypll.nn as hnn
from hypll.tensors import TangentTensor

from networks.nets import DeepSet, CategoricalMLP


def manifold_map(x, manifold):
    """
    Maps a tensor in Euclidean space onto a Riemannian Manifold
    """
    tangents = TangentTensor(x, man_dim=-1, manifold=manifold)
    return manifold.expmap(tangents)

class HyperbolicMLP(nn.Module):
    """
    MLP with 2 Euclidean layers and 2 hyperbolic layers
    in_features: input dimension
    manifold: Desired manifold (e.g. Poincare Disc)
    euc_width: Width of the Euclidean hidden layer
    hyp_width: Width of the hyperbolic hidden layer
    out_features: Output dimension
    """
    def __init__(self, in_features, out_features, manifold, euc_width=64, hyp_width=64):
        super(HyperbolicMLP, self).__init__()
        self.manifold = manifold

        # Euclidean layers
        self.euc_layer1 = nn.Linear(in_features, euc_width)
        self.euc_layer2 = nn.Linear(euc_width, hyp_width)
        self.euc_relu = nn.ReLU()

        # Hyperbolic layers
        self.hyp_layer1 = hnn.HLinear(in_features=hyp_width, out_features=hyp_width, manifold=manifold)
        self.hyp_layer2 = hnn.HLinear(in_features=hyp_width, out_features=out_features, manifold=manifold)
        self.hyp_relu = hnn.HReLU(manifold=manifold)

    def forward(self, x):
        """
        Forward pass.
        """
        # Pass through Euclidean layers
        x = self.euc_relu(self.euc_layer1(x))
        x = self.euc_layer2(x)

        # Map to hyperbolic space
        x = manifold_map(x, self.manifold)

        # Pass through Hyperbolic layers
        x = self.hyp_relu(self.hyp_layer1(x))
        x = self.hyp_layer2(x)
        return x
    

class HyperbolicCategoricalMLP(nn.Module):
    def __init__(self, cat_features, embedding_dims, euc_hidden_dims, hyp_hidden_dims, output_dim, manifold):
        super(HyperbolicCategoricalMLP, self).__init__()
        
        self.manifold = manifold
        
        # Use the CategoricalMLP for Euclidean part
        self.euc_mlp = CategoricalMLP(cat_features, embedding_dims, euc_hidden_dims, hyp_hidden_dims[0])
        
        # Hyperbolic layers
        hyp_layers = []
        for i in range(1, len(hyp_hidden_dims)):
            hyp_layers.append(hnn.HLinear(hyp_hidden_dims[i-1], hyp_hidden_dims[i], manifold=manifold))
            hyp_layers.append(hnn.HReLU(manifold=manifold))
        
        hyp_layers.append(hnn.HLinear(hyp_hidden_dims[-1], output_dim, manifold=manifold))
        
        self.hyp_mlp = nn.Sequential(*hyp_layers)
    
    def forward(self, x):
        # Pass through Euclidean layers
        euc_output = self.euc_mlp(x)
        
        # Map to hyperbolic space
        hyp_input = manifold_map(euc_output, self.manifold)
        
        # Pass through Hyperbolic layers
        output = self.hyp_mlp(hyp_input)
        
        return output


class HyperbolicDeepSet(nn.Module):
    """
    DeepSet architecture for set-valued inputs with customizable phi function. 
    Final layers transform input into hyperbolic space.
    See: https://arxiv.org/abs/1703.06114
    """
    def __init__(self, input_dim, hidden_dim, output_dim, manifold, phi=None):
        super(HyperbolicDeepSet, self).__init__()

        self.deepset = DeepSet(input_dim, hidden_dim, hidden_dim, phi=phi)
        self.manifold = manifold

        # Hyperbolic layers
        self.hyp_layer1 = hnn.HLinear(
            in_features=hidden_dim, out_features=hidden_dim, manifold=manifold
        )
        self.hyp_relu = hnn.HReLU(manifold=manifold)
        self.hyp_layer2 = hnn.HLinear(
            in_features=hidden_dim, out_features=output_dim, manifold=manifold
        )

    def forward(self, x, mask):
        # Process input through DeepSet
        x = self.deepset(x, mask)

        # Map directly to hyperbolic space
        x = manifold_map(x, self.manifold)

        # Pass through Hyperbolic layers
        x = self.hyp_layer1(x)
        x = self.hyp_relu(x)
        x = self.hyp_layer2(x)

        return x
