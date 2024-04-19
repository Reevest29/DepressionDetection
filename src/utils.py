import torch

def to_device(iterable,device):
    import pdb; pdb.set_trace()
    return [x.to(device) for x in iterable]