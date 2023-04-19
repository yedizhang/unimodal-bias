import torch
import torch.nn as nn
import torch.nn.functional as F


class shallow_net(nn.Module):
    """
    A Linear Neural Net with no hidden layer
    """
    def __init__(self, in_dim, out_dim, gamma=1e-12):
        """
        Args:
            int in_dim: Input dimension
            int hid_dim: Hidden dimension
        """
        super().__init__()
        self.in_out = nn.Linear(in_dim, out_dim, bias=False)
        self._init_weights(gamma)

    def forward(self, x):
        """
        Args:
            torch.Tensor x: Input tensor
        Returns:
            torch.Tensor out: Output/Prediction
        """
        out = self.in_out(x)
        return out

    def _init_weights(self, gamma):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=gamma)


class early_fusion(nn.Module):
    """
    A Linear Neural Net with one hidden layer
    """
    def __init__(self, in_dim, hid_dim, out_dim, gamma=1e-12):
        """
        Args:
            int in_dim: Input dimension
            int out_dim: Ouput dimension
            int hid_dim: Hidden dimension
        """
        super().__init__()
        bias = False
        self.in_hid = nn.Linear(in_dim, hid_dim, bias=bias)
        self.hid_out = nn.Linear(hid_dim, out_dim, bias=bias)
        self._init_weights(gamma)

    def forward(self, x):
        """
        Args:
            torch.Tensor x: Input tensor
        Returns:
            torch.Tensor hid: Hidden layer activity
            torch.Tensor out: Output/Prediction
        """
        hid = self.in_hid(x)
        # hid = F.relu(hid)
        out = self.hid_out(hid)
        return out
    
    def _init_weights(self, gamma):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=gamma)
                if m.bias is not None:
                    nn.init.normal_(m.bias, mean=0, std=gamma)


class late_fusion(nn.Module):
    """
    A Linear Neural Net with one hidden layer
    """
    def __init__(self, in_dim, hid_dim, out_dim, gamma=1e-12):
        """
        Args:
            int in_dim: Input dimension
            int out_dim: Ouput dimension
            int hid_dim: Hidden dimension for one modality
        """
        super().__init__()
        bias = False
        self.inA_hid = nn.Linear(in_dim//2, hid_dim, bias=bias)
        self.inB_hid = nn.Linear(in_dim//2, hid_dim, bias=bias)
        self.hid_out = nn.Linear(hid_dim*2, out_dim, bias=bias)

        self._init_weights(gamma)

    def forward(self, x1, x2):
        """
        Args:
            torch.Tensor x: Input tensor
        Returns:
            torch.Tensor hid: Hidden layer activity
            torch.Tensor out: Output/Prediction
        """
        hidA = self.inA_hid(x1)
        hidB = self.inB_hid(x2)
        hid_fuse = torch.cat((hidA, hidB), -1)
        # hid_fuse = F.relu(hid_fuse)
        out = self.hid_out(hid_fuse)
        return out

    def _init_weights(self, gamma):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=gamma)
                if m.bias is not None:
                    nn.init.normal_(m.bias, mean=0, std=gamma)


class deep_fusion(nn.Module):
    """
    A Linear Neural Net with multiple hidden layers
    """
    def __init__(self, in_dim, hid_dim, out_dim, depth, fuse_depth, gamma):
        super(deep_fusion, self).__init__()
        self.depth = depth
        self.fuse_depth = fuse_depth

        self.layers = nn.ModuleDict()
        for i in range(1, fuse_depth):  # iterate 0, 1, ..., fuse_depth-1
            self.layers['encodeA_'+str(i)] = torch.nn.Linear(in_dim//2, hid_dim, bias=False)
            self.layers['encodeB_'+str(i)] = torch.nn.Linear(in_dim//2, hid_dim, bias=False)
        self.layers['fuse'] = torch.nn.Linear(hid_dim*2, hid_dim, bias=False)
        for i in range(fuse_depth, depth):  # iterate fuse_depth, ..., depth-1
            self.layers['decode_'+str(i)] = torch.nn.Linear(hid_dim, hid_dim, bias=False)
        # self.layers['hid_out'] = nn.Linear(hid_dim*2, out_dim, bias=False)

        self._init_weights(gamma)

    def forward(self, x1, x2):
        for i in range(1, self.fuse_depth):
            x1 = self.layers['encodeA_'+str(i)](x1)
            x2 = self.layers['encodeB_'+str(i)](x2)
        x = torch.cat((x1, x2), -1)
        x = self.layers['fuse'](x)
        for i in range(self.fuse_depth, self.depth):
            x = self.layers['decode_'+str(i)](x)
        return x

    def _init_weights(self, gamma):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=gamma)