import torch
import numpy as np
import torch.nn as nn


def low_rank_approximate(mat_org: torch.tensor, rank=32):
    """ Learning a low-rank decomposition for the given matrix.

    Args:
        mat_org (torch.tensor): the given matrix.
        rank (int, optional): defined rank value. Defaults to 16.
    """
    device = mat_org.device

    if not device == 'cpu':
        mat_org = mat_org.cpu()
    u, s, vh = np.linalg.svd(mat_org.detach().numpy(), full_matrices=True)

    s_val = np.sqrt(np.diag(s[:rank])) # half singular value
    mat_q = torch.tensor(u[:, :rank] @ s_val)
    mat_r = torch.tensor(s_val @ vh[:rank, :])
    error = nn.functional.mse_loss(mat_q @ mat_r, mat_org)

    mat_q = mat_q.to(device)
    mat_r = mat_r.to(device)

    output = {'mat_q': mat_q,
              'mat_r': mat_r.t(),
              'error': error}
    return output


class ModuleLowRank(object):
    """ Replace the original Linear matrix with two low-rank linear matrices.

    Args:
        compress_ratio (int): the pre-defined rank ratio value.
        name_omit (list of str): the omitted name list for low-rank approximation.
        is_approximate (bool, optional): perform low-rank approximation. Defaults to True.
    """

    def __init__(self,
                 compress_ratio=3,
                 name_omit=list(),
                 is_approximate=True):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.name_omit = name_omit
        self.is_approximate = is_approximate

    def _apply(self, name: str, module: nn.Linear):
        """ Apply nn.Sequential for replacement of the Linear module.

        Args:
            name (str): module name
            module (nn.Linear): the given Linear module
        """
        shape = (module.in_features, module.out_features)
        weight, bias = module.weight, module.bias

        rank = (shape[0] * shape[1]) // (self.compress_ratio * (shape[0] + shape[1]))
        rank = int(rank)

        # Add two new Linear modules
        module_l = nn.Linear(shape[0], rank, bias=False,)
        module_r = nn.Linear(rank, shape[1], bias=not bias is None,)
        module_l = module_l.to(weight.device) # for old pytorch version
        module_r = module_r.to(weight.device) # for old pytorch version

        if self.is_approximate:
            lr_out = low_rank_approximate(weight.t(), rank)
            weight_l, weight_r = lr_out['mat_q'], lr_out['mat_r']

            module_l.weight.data.copy_(weight_l.t())
            module_r.weight.data.copy_(weight_r)
            if not bias is None:
                module_r.bias.data.copy_(bias)
        else:
            weight_l, weight_r = None, None

        return {'weight_l': weight_l,
                'weight_r': weight_r,
                'module_rep': nn.Sequential(module_l, module_r)}

    def __call__(self, module: nn.Module):
        copied_modules = {name: module_sub for name, module_sub in module.named_modules()}
        for name, module_sub in copied_modules.items():
            if isinstance(module_sub, nn.Linear): 
                if any(n in name for n in self.name_omit):
                    continue
                if module_sub.out_features < 10:
                    continue # for some head matrix, such as image-text match head

                base, localname = module, name
                while '.' in localname:
                    prefix, localname = localname.split('.', 1)
                    base = base.__getattr__(prefix)
                output = self._apply(name, module_sub)
                print("applying low rank on", name)

                setattr(base, localname, output['module_rep'])

        return module
