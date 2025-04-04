from torch import nn
import torch

class Expert(nn.Module):
    def __init__(self, modules, device="cuda"):
        super(Expert, self).__init__()
        modules = [module() for module in modules]
        self.layers = nn.Sequential(*modules).to(device)

    def forward(self, x):
        return self.layers(x)


class ExpertsModel(nn.Module):
    def __init__(self, modules, end, gate, device="cuda"):
        super(ExpertsModel, self).__init__()
        self.num_experts = 2**(len(modules)-1)
        modules = [layer for sublist in modules for layer in (sublist if isinstance(sublist, list) else [sublist])]
        self.device = device
        self.gate = gate(self.num_experts).to(device)
        self.end = end # This is used for the final output layer if needed
        self.experts = [Expert(modules, device) for _ in range(self.num_experts)]
    
    def parameters(self, recurse: bool = True):
        # Collect parameters from the root node, all child nodes, the gate, and the end module
        params = []
        for node in self.experts:
            params.extend(node.parameters(recurse=recurse))
        
        # Eliminate duplicate parameters by using a set
        unique_params = set(params)
        params = list(unique_params)

        params.extend(self.gate.parameters(recurse=recurse))
        return iter(params)


    def forward(self, x):
        gate_output = torch.max(self.gate(x), dim=1).indices

        # Use the gate output to decide which node to start in
        selected_nodes = [self.experts[i] for i in gate_output]
        outputs = torch.stack([node(x[i].unsqueeze(0)) for i, node in enumerate(selected_nodes)]).squeeze(1)

        if self.end:
            outputs = self.end(outputs)
        return outputs
    
        
            
    
