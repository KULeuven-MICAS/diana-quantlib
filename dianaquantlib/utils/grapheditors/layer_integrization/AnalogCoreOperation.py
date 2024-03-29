# find  pattern analog conv epstunnel identity epstunnel accumulator epstunnel 
# replace that with  DIANAConv2D and epstunnel (absorb the scaling into the dianaconv weights or the epstunnel after)  copy the weight values and biases 
from typing import List
import torch 
from torch import nn 
from dianaquantlib.core.Operations import AnalogAccumulator, AnalogConv2d, AnalogGaussianNoise, DIANAIdentity
from quantlib.algorithms.qmodules.qmodules.qmodules import _QModule
from quantlib.editing.editing.editors.base.composededitor import ComposedEditor
from quantlib.editing.editing.editors.base.editor import Editor
from quantlib.editing.editing.editors.base.rewriter.applier import Applier
from quantlib.editing.editing.editors.nnmodules.applicationpoint import NodesMap
from quantlib.editing.editing.editors.nnmodules.applier import NNModuleApplier
from quantlib.editing.editing.editors.nnmodules.finder.nnsequential import PathGraphMatcher
from quantlib.editing.editing.editors.nnmodules.pattern.base.pattern import NNModulePattern
from quantlib.editing.editing.editors.nnmodules.pattern.nnsequential.factory.candidates import Candidates, NNModuleDescription
from quantlib.editing.editing.editors.nnmodules.pattern.nnsequential.factory.factory import generate_named_patterns
from quantlib.editing.editing.editors.nnmodules.pattern.nnsequential.factory.roles import Roles
from quantlib.editing.editing.editors.nnmodules.rewriter.factory import get_rewriter_class
from quantlib.editing.editing.fake2true.integerisation.linearopintegeriser.finder import LinearOpIntegeriserMatcher
from quantlib.editing.graphs.nn.epstunnel import EpsTunnel
import torch.fx as fx

from quantlib.editing.graphs.nn.requant import Requantisation
_BN_KWARGS = {'num_features': 1}

analog_roles  = Roles([


    ('eps_in',  Candidates([
        ('Eps', NNModuleDescription(class_=EpsTunnel, kwargs= {'eps': torch.Tensor([1.0])})),
    ])),
    ('conv', Candidates([
        ('QConv2d', NNModuleDescription(class_=AnalogConv2d, kwargs={'qrangespec':'ternary' , 'qgranularityspec':'per-array' , 'qhparamsinitstrategyspec' :'meanstd' , 'in_channels': 1, 'out_channels': 1, 'kernel_size': 1}))
    ])),
    
    ('eps_conv_out', Candidates([
        ('Eps', NNModuleDescription(class_=EpsTunnel, kwargs= {'eps': torch.Tensor([1.0])})),
    ])),
     ('identity', Candidates([
        ('Requant', NNModuleDescription(class_=DIANAIdentity, kwargs={'qrangespec':'ternary' , 'qgranularityspec':'per-array' , 'qhparamsinitstrategyspec' :'meanstd' })),
    ])),
     ('eps_identity_out', Candidates([
        ('Eps', NNModuleDescription(class_=EpsTunnel, kwargs= {'eps': torch.Tensor([1.0])})),
    ])),
     ('noise', Candidates([
        ('Noise', NNModuleDescription(class_=AnalogGaussianNoise, kwargs= {'signed':True , 'bitwidth': 6})),
    ])),# noise node 
     ('eps_noise_out', Candidates([
         ('',     None),  
        ('Eps', NNModuleDescription(class_=EpsTunnel, kwargs= {'eps': torch.Tensor([1.0])})),
    ])),
    ('accumulator', Candidates([
        ('Accumulator', NNModuleDescription(class_=AnalogAccumulator, kwargs={})),
    ]))
])
# The layer integrization values cannot an accurate representation of what the actual result is, because the preservation of order of operations is not the same when removing the analog accumulator (not taking into account partial sums) so the accuracy of integrized model isin't relevant
class AnalogConvIntegrizerApplier(NNModuleApplier) : 
    def __init__(self, pattern: NNModulePattern):
        super().__init__(pattern)
    @classmethod 
    def create_torch_module(cls, qconv : AnalogConv2d, ) :  
        assert isinstance(qconv, AnalogConv2d)  
        new_module = nn.Conv2d(in_channels=qconv.in_channels,
                                out_channels=qconv.out_channels,
                                kernel_size=qconv.kernel_size,
                                stride=qconv.stride,
                                padding=qconv.padding,
                                dilation=qconv.dilation,
                                groups=qconv.groups,
                                bias=False)
        new_module.weight.data = torch.round(qconv.qweight.data.clone().detach()  / qconv.scale.data.clone().detach()) # fake quantized / scale = true quantized
        #new_module.weight.data = qconv.qweight.data.clone().detach()  # fake quantized / scale = true quantized
        
        new_module.register_buffer("is_analog" , torch.Tensor([True])) 
        
        return new_module
        

    def _apply(self, g: fx.GraphModule, ap: NodesMap, id_: str) -> fx.GraphModule:
        # get handles on matched `fx.Node`s
        name_to_match_node = self.pattern.name_to_match_node(nodes_map=ap)
        node_eps_in  = name_to_match_node['eps_in']
        node_conv  = name_to_match_node['conv']
        node_conv_out = name_to_match_node['eps_conv_out']
        node_identity_out = name_to_match_node["eps_identity_out"]
        node_noise= name_to_match_node['noise']
        node_noise_out= name_to_match_node['eps_noise_out'] if 'eps_noise_out' in name_to_match_node.keys() else None # 
        node_accumulator  = name_to_match_node['accumulator']


        # get handles on matched `nn.Module`s
        name_to_match_module = self.pattern.name_to_match_module(nodes_map=ap, data_gm=g)
        module_eps_in  = name_to_match_module['eps_in']
        module_conv  = name_to_match_module['conv']
        module_conv_out = name_to_match_module["eps_conv_out"]
        
         # create the integerised linear operation
        new_target = id_
        new_module = AnalogConvIntegrizerApplier.create_torch_module(module_conv)
  
        # add the requantised linear operation to the graph...
        
        g.add_submodule(new_target, new_module)
        with g.graph.inserting_after(node_eps_in):
            new_node = g.graph.call_module(new_target, args=(node_eps_in,))
        node_conv_out.replace_input_with(node_conv, new_node)
        module_conv_out.set_eps_in(torch.ones_like(module_conv_out.eps_in))
        module_eps_in.set_eps_out(torch.ones_like(module_eps_in.eps_out))

        node_accumulator.replace_all_uses_with(node_noise_out)
        node_noise.replace_all_uses_with(node_identity_out)
        
       
            
       
        
        # noise prop 
         # ...and delete the old operation
        g.delete_submodule(node_conv.target)
        g.graph.erase_node(node_conv)
        g.delete_submodule(node_accumulator.target)
        g.graph.erase_node(node_accumulator)

        
        
       
        return g

class AnalogConvIntegrizer(ComposedEditor):   
    def __init__(self):
        # generate rewriters for qconv2d and qlinear 
        admissible_screenplays = list(analog_roles.all_screenplays)

    # programmatically generate all the patterns, then for each pattern generate the corresponding `Rewriter`
        editors : List[Editor]
        editors =[]
        for name, pattern in generate_named_patterns(analog_roles, admissible_screenplays):
            class_name = name + 'DianaAnalogIntegeriser'
            class_ = get_rewriter_class(class_name, pattern, PathGraphMatcher, AnalogConvIntegrizerApplier)
            editors.append(class_())
        super().__init__(editors)
    pass 
 