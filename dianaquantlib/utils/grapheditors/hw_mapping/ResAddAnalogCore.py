from dianaquantlib.utils.Requantizers.muladd import MulAdd
from quantlib.editing.editing.editors.base.rewriter.rewriter import Rewriter

from typing import List
from dianaquantlib.utils.grapheditors import DianaAps
from quantlib.editing.editing.editors.base.composededitor import ComposedEditor
from quantlib.editing.editing.editors.base.rewriter.applier import Applier
from quantlib.editing.editing.editors.base.rewriter.finder import Finder
from quantlib.editing.editing.editors.base.rewriter.rewriter import Rewriter
import torch.fx as fx
from quantlib.editing.graphs.fx import quantlib_symbolic_trace
from quantlib.editing.graphs.fx.fxnodes import FXOpcodeClasses
from quantlib.editing.graphs.nn.epstunnel import EpsTunnel
from torch import nn
import torch 
from dianaquantlib.core.Operations import AnalogAccumulator

# analog core patterns 

class ResidualAddsAnalogCoreFinder(Finder): # only looks for residual add pattern: 2 inputs , one of which is quantized is passed through eps_tunnel , while the other is a BN. Other cases without the BN are handles with the eps construct simplifier and epstunnel after add
    def __init__(self) -> None:
        super().__init__()
    
    def find(self, g: fx.GraphModule) -> List[DianaAps]:
        aps : List[DianaAps] = []  
        for n in g .graph.nodes: 
            if ( n.op in FXOpcodeClasses.CALL_FUNCTION.value or n.op in FXOpcodeClasses.CALL_METHOD.value ) and "add" in str(n):
                predecessors = [p for p in n.all_input_nodes]
                assert isinstance(g.get_submodule(predecessors[0].target) , EpsTunnel) and isinstance(g.get_submodule(predecessors[1].target) , EpsTunnel)
                if len(predecessors) <= 2: 
                    corrects = 0 
                    try: 
                        # pattern is eps tunnel - act - bn 
                        #searching for batchnorms  need to traverse back 2 steps

                        bn_candidate_0 = [p for p in [p for p in predecessors[0].all_input_nodes][0].all_input_nodes][0]
                        bn_candidate_1 = [p for p in [p for p in predecessors[1].all_input_nodes][0].all_input_nodes][0]
                         
                        if len([u for u in bn_candidate_0.users])==1 and isinstance(g.get_submodule(bn_candidate_0.target)   , nn.BatchNorm2d) : 
                            aps.append(DianaAps('bn_0' , n)) 
                        elif len([u for u in bn_candidate_1.users])==1 and isinstance(g.get_submodule(bn_candidate_1 .target) , nn.BatchNorm2d) :  
                            aps.append(DianaAps('bn_1' , n)) 


                    except: 
                        continue               
        return aps 
    
    def check_aps_commutativity(self, aps: List[DianaAps]) -> bool:
        return len(aps) == len(set(ap.node for ap in aps))  # each `fx.Node` should appear at most once 

class ResidualAddsAnalogCoreApplier(Applier): 
    def __init__(self):
        super().__init__()
        self.bn_bitwidth = torch.Tensor([2**8]) 
    def _apply(self, g: fx.GraphModule, ap: DianaAps, id_: str) -> fx.GraphModule:
        node = ap.node
        predecessors = [p for p in node.all_input_nodes]
        assert (len(predecessors) <=2) 

        users = [u for u in node.users] 
        
        if ap.type == 'bn_0':
            eps_tunnel_node = predecessors[0]
            node_act = [p for p in eps_tunnel_node.all_input_nodes][0] 
            activation_eps = g.get_submodule(eps_tunnel_node.target).eps_out.clone().detach()  
            node_bn = [p for p in node_act.all_input_nodes][0]
            node_tunnel =  predecessors[1]

            # replace add input and delete eps tunnel and activation 
            

        elif ap.type == 'bn_1' : 
            eps_tunnel_node = predecessors[1]
            node_act = [p for p in eps_tunnel_node.all_input_nodes][0] 
            activation_eps = g.get_submodule(eps_tunnel_node.target).eps_out.clone().detach()  
            node_bn = [p for p in node_act.all_input_nodes][0]
            node_tunnel =  predecessors[0]
        
        node.replace_input_with(eps_tunnel_node , node_bn) 
        g.delete_submodule(node_act.target)
        g.delete_submodule(eps_tunnel_node.target)
        g.graph.erase_node(eps_tunnel_node)
        g.graph.erase_node(node_act)

        module_tunnel : EpsTunnel = g.get_submodule(node_tunnel.target)
        module_bn = g.get_submodule(node_bn.target)
        # absorb the scale from the adc 
        nodes_bn_predecessors = [p for p in node_bn.all_input_nodes] 
        nodes_bn_users = [u for u in node_bn.users]  
        assert(len(nodes_bn_users) == 1)

        eps_in = torch.Tensor([1]) 
        assert len(nodes_bn_predecessors) == 1
        try: 
            eps_module = g.get_submodule(nodes_bn_predecessors[0].target)  
            if isinstance(eps_module , AnalogAccumulator): 
        
                acc_predecessors = [p for p in nodes_bn_predecessors[0].all_input_nodes]# predecessors of analog accum 
                assert len(acc_predecessors) ==1 
                e_mod = g .get_submodule(acc_predecessors[0].target) 
                assert isinstance(e_mod, EpsTunnel) 
                eps_module = e_mod 
            if isinstance(eps_module, EpsTunnel):
                eps_in = eps_module._eps_out.clone().detach() 
                eps_module.set_eps_out(torch.ones_like(eps_module.eps_out)) 
            
        except: 
            pass 
    
        #compute mul add with the scale also absorb the eps tunnel of activation that followed BN to match the other eps_tunnel 
        match_factor =  activation_eps/module_tunnel.eps_out  # factor to be absorbed into batchnorm 
        absorbed_factor = match_factor / activation_eps
        tunnel_eps_out = module_tunnel.eps_out.clone().detach() 
        assert(node_tunnel is not None and node_bn is not None)
        
        shape = node_bn.meta['tensor_meta'].shape
        mi      = module_bn.running_mean 
        sigma   = torch.sqrt(module_bn.running_var + module_bn.eps) 
        gamma   = module_bn.weight 
        beta    = module_bn.bias
        broadcast_shape = tuple(1 if i != 1 else mi.numel() for i, _ in enumerate(range(0, len(shape))))
        mi    = mi.reshape(broadcast_shape)
        sigma = sigma.reshape(broadcast_shape)
        gamma = gamma.reshape(broadcast_shape)
        beta  = beta.reshape(broadcast_shape)

        # Mul gamma / sigma 
        # Add beta    = module_bn.bias
        mul = eps_in *gamma /sigma  * absorbed_factor
        add =(-mi * gamma + beta * sigma) / (sigma) * absorbed_factor
        #print( "Max mul: " , torch.max(torch.abs(mul)))
        #print( "Max add: " , torch.max(torch.abs(add)))
        factored_power_of_2 = torch.Tensor([15])
        while True  : 
            max_multiplier = torch.max(torch.maximum(torch.abs(mul ), torch.abs(add )))
            
            if  torch.exp2(factored_power_of_2) * max_multiplier > self.bn_bitwidth/2 :
                factored_power_of_2 -= 1
                
                
               
            else : 
                #print("Maximum Value: " ,  torch.exp2(factored_power_of_2) * max_multiplier)
                #print("Factored power of 2: "  ,factored_power_of_2) 
              #  print(max_multiplier)
                break
                 
            
    
        add =torch.clamp(torch.round( torch.exp2(factored_power_of_2) *  add ),  min = -self.bn_bitwidth /2, max = self.bn_bitwidth/2 - 1)
        mul = torch.clamp(torch.round(torch.exp2(factored_power_of_2) * mul ) ,  min = -self.bn_bitwidth /2, max = self.bn_bitwidth/2 - 1)
        #print(add)

        # add new Mul add module and delete batch norm 
        mul_add_target = id_ 
        muladd_module = MulAdd(mul , add ) 
        g.add_submodule(mul_add_target, muladd_module) 
        with g.graph.inserting_after(node_bn ): 
            mul_add_node = g.graph.call_module(mul_add_target , (nodes_bn_predecessors[0], )) 
        node.replace_input_with(node_bn , mul_add_node)      
        g.delete_submodule(node_bn.target) 
        g.graph.erase_node(node_bn) 

        # Shift other input of addition by scale and insert eps tunnel after the addition 
        scale_factor = torch.exp2(factored_power_of_2)
        #module_tunnel.set_eps_out(torch.clamp(torch.exp2(torch.round(torch.log2(module_tunnel._eps_out *scale_factor) )), min=torch.Tensor([1])))
        module_tunnel.set_eps_out(torch.clamp(scale_factor, min=torch.Tensor([1])))
        #print("Module_tunnel scale: " , module_tunnel._eps_out) 
        
        
        for u in users: 
            try: 
                candidate = g.get_submodule(u.target)
                if not isinstance(candidate, EpsTunnel): 
                    eps_out_target = id_ +  f'[{str(self._counter)}]'
                    new_module = EpsTunnel(torch.Tensor([1]))
                 
                    new_module.set_eps_out(tunnel_eps_out/scale_factor)
                    g.add_submodule(eps_out_target , new_module)
                    with g.graph.inserting_after(node): 
                        out_eps_node = g.graph.call_module(eps_out_target, (node, )) 
                    u.replace_input_with(node , out_eps_node)
                else: 
                    print("TUNNEL ALREADY EXISTS ")
                    candidate.set_eps_in(torch.ones_like(candidate.eps_in) ) 
                    candidate.set_eps_out(torch.ones_like(candidate.eps_out) * tunnel_eps_out/scale_factor) 
            except: 
                pass 
        return g 

class ResidualAddsAnalogCoreRewriter(Rewriter) : 
    def __init__(self):
        super(ResidualAddsAnalogCoreRewriter, self).__init__(name='ResidualAnalogAdditions', symbolic_trace_fn=quantlib_symbolic_trace,finder= ResidualAddsAnalogCoreFinder(), applier=ResidualAddsAnalogCoreApplier())
