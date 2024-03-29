from typing import List, Tuple
from dianaquantlib.utils.grapheditors import DianaAps
from quantlib.editing.editing.editors.base.composededitor import ComposedEditor
from quantlib.editing.editing.editors.base.rewriter.applier import Applier
from quantlib.editing.editing.editors.base.rewriter.finder import Finder
from quantlib.editing.editing.editors.base.rewriter.rewriter import Rewriter
import torch.fx as fx
import torch.nn as nn
from functools import partial
from quantlib.editing.graphs.fx import quantlib_symbolic_trace, FXOpcodeClasses
from quantlib.editing.editing.float2fake.quantisation.modulewiseconverter.modulewisedescription.modulewisedescription import ModuleWiseDescription, ModuleWiseDescriptionSpecType

from dianaquantlib.core.Operations import AnalogAccumulator, AnalogConv2d, AnalogOutIdentity, DIANAIdentity, DIANALinear, IdentityType , DIANAConv2d


MODULES_WITH_QUANTIZERS = [  AnalogConv2d , DIANAConv2d , DIANALinear]


class DianaOpQuantFinder(Finder):

    def __init__(self):
        super().__init__()

    def find(self, g: fx.GraphModule) -> List[DianaAps]:
        def getmodule_name(module) :

            if type(module) == DIANAConv2d:
                return 'conv'
            elif type(module) == DIANALinear:
                return 'linear'
            elif type(module) == AnalogConv2d:
                return 'aconv'
            return ''

        aps: List[DianaAps] = []
        # get names of submodules that match the search
        name_to_module = {}
        for name,mod in g.named_modules():
            if type(mod) in MODULES_WITH_QUANTIZERS:
                name_to_module[name] = mod

        for n in g.graph.nodes:
            if n.target in name_to_module.keys():
                aps.append(DianaAps(getmodule_name(name_to_module[n.target]), n))
                # Comment line above and uncomment lines below in case input nodes should not be quantized
                #prev = [p for p in n.all_input_nodes][0]
                #if prev.op not in FXOpcodeClasses.PLACEHOLDER.value:
                #    aps.append(DianaAps(getmodule_name(name_to_module[n.target]), n))

        return aps
    def check_aps_commutativity(self, aps: List[DianaAps]) -> bool:
        return len(aps) == len(set(ap.node for ap in aps))  # each `fx.Node` should appear at most once


class DianaOpQuantApplier(Applier):

    def __init__(self, modulewisedescriptionspec: ModuleWiseDescriptionSpecType,
                 analogcoredescriptionspec: Tuple[str, ...]):
        super().__init__()

        # Fetch the quantization description for Indentity functions
        self.identity_qdescription = None
        modulewisedescription = ModuleWiseDescription(modulewisedescriptionspec)
        for _, (n2mfilter, qdescription) in modulewisedescription.items():
            if len(n2mfilter({'x': nn.Identity()})) > 0:
                self.identity_qdescription = qdescription

        assert self.identity_qdescription is not None
        self.analogcoredescriptionspec = analogcoredescriptionspec

    def _apply(self, g: fx.GraphModule, ap: DianaAps, id_: str) -> fx.GraphModule:

        type_out = IdentityType.default
        if ap.type == 'aconv':
            type_out = IdentityType.AIMC_OUT

        identity_cls = partial(DIANAIdentity, self.identity_qdescription.qrange,
                                              self.identity_qdescription.qgranularity,
                                              self.identity_qdescription.qhparamsinitstrategy)
        qpre = identity_cls()
        qpost = None
        users = [u for u in ap.node.users]
        if type_out == IdentityType.AIMC_OUT:
            qpost = AnalogOutIdentity(self.analogcoredescriptionspec[0],
                                      self.analogcoredescriptionspec[2])
        # if the current node points to the last (non analog) linear layer, insert DIANAIdentity
        elif len(users) == 1 and users[0].op in FXOpcodeClasses.OUTPUT.value:
            qpost = identity_cls()

        pre_target = id_

        g.add_submodule(pre_target ,qpre)

        input_node = None
            # get input x of qpre

        if len(ap.node.all_input_nodes) ==1:
            input_node = ap.node.all_input_nodes[0]

        else:
            raise ValueError
        with g.graph.inserting_before(ap.node):
            pre_node = g.graph.call_module(pre_target, args=(input_node,))
        ap.node.replace_input_with(input_node , pre_node)
        # Now put quantizer after

         # add the quantiser to the graph (interposing it between the two linear nodes)
        # We want that after the rewriting each user of `node_pre` reads the
        # output of `new_node` instead; however, in the intermediate state,
        # `new_node` will itself be a user of `node_pre`. Therefore, we need
        # to determine who these users are before `new_node` becomes one of
        # them.
        if qpost is not None:
            post_target = id_ + f'[{str(self._counter)}]'
            g.add_submodule(post_target, qpost)
            downstream_nodes = list(ap.node.users)
            with g.graph.inserting_after(ap.node):
                post_node = g.graph.call_module(post_target , args=(ap.node,))
            for u in downstream_nodes:
                u.replace_input_with(ap.node, post_node)
        return g


class DianaF2FInterposer(ComposedEditor): #insert quantidentities between

    def __init__(self, modulewisedescriptionspec: ModuleWiseDescriptionSpecType,
                 analogcoredescriptionspec: Tuple[str, ...]):

        rewriter = Rewriter(name='DianaInterposer',
                            symbolic_trace_fn=quantlib_symbolic_trace,
                            finder=DianaOpQuantFinder(),
                            applier=DianaOpQuantApplier(modulewisedescriptionspec, analogcoredescriptionspec))

        super(DianaF2FInterposer, self).__init__([rewriter ])

#endregion

