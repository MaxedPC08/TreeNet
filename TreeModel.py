from torch import nn
import torch
class TreeNode(nn.Module):
    def __init__(self, modules, root=False):
        super(TreeNode, self).__init__()
        modules = [module() for module in modules]
        self.module = nn.Sequential(*modules)
        self.parent = None
        self.root = root

    def set_parent(self, parent_node):
        self.parent = parent_node

    def forward(self, x):
        if self.root:
            return self.module(x)
        return self.parent(self.module(x))

class TreeModel(nn.Module):
    def __init__(self, modules, end, gate, device="cuda"):
        super(TreeModel, self).__init__()
        modules = list(reversed(modules))
        self.root = TreeNode(modules[0], True)
        self.nodes = []
        self.total_nodes = [self.root]
        self.device = device
        self.setupNode(self.root, modules[1:])
        self.gate = gate(len(self.nodes))
        self.end = end  # This is used for the final output layer if needed

    def parameters(self, recurse: bool = True):
        # Collect parameters from the root node, all child nodes, the gate, and the end module
        params = []
        for node in self.total_nodes:
            params.extend(node.parameters(recurse=recurse))
        
        # Eliminate duplicate parameters by using a set
        unique_params = set(params)
        params = list(unique_params)

        params.extend(self.gate.parameters(recurse=recurse))
        return iter(params)
    
    def setupNode(self, parent_node, modules):
        if len(modules) == 0:
            return
        new_node_1 = TreeNode(modules[0])
        new_node_1.set_parent(parent_node)

        new_node_2 = TreeNode(modules[0])
        new_node_2.set_parent(parent_node)
        
        new_node_1.to(self.device)
        new_node_2.to(self.device)

        self.total_nodes.append(new_node_1)
        self.total_nodes.append(new_node_2)

        if len(modules) == 1:
            self.nodes.append(new_node_1)
            self.nodes.append(new_node_2)
            return
        self.setupNode(new_node_1, modules[1:])
        self.setupNode(new_node_2, modules[1:])


    def forward(self, x):
        gate_output = torch.max(self.gate(x), dim=1).indices

        # Use the gate output to decide which node to start in
        selected_nodes = [self.nodes[i] for i in gate_output]
        x_split = torch.split(x, 1, dim=0)

        # Parallelize the computation using torch.jit.fork
        futures = [torch.jit.fork(node, x_split[i]) for i, node in enumerate(selected_nodes)]
        outputs = torch.cat([torch.jit.wait(future) for future in futures], dim=0)

        if self.end:
            outputs = self.end(outputs)
        return outputs  
