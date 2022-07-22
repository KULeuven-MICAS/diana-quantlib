
from abc import abstractmethod
import enum
from numpy import require

import torch 
from torch import Tensor, nn
from typing import Union , Tuple


from quantlib.algorithms.qalgorithms.qatalgorithms.pact.qactivations import PACTReLU
from quantlib.algorithms.qbase.qhparams.qhparams import create_qhparams
from quantlib.algorithms.qbase.qrange.qrange import resolve_qrangespec



from quantlib.algorithms.qmodules.qmodules import  QIdentity

from quantlib.algorithms.qmodules.qmodules.qlinears import QConv2d, QLinear 

import math

from quantlib.algorithms.qbase import QRangeSpecType, QGranularitySpecType, QHParamsInitStrategySpecType
from DianaModules.utils._FakeQuantizer import _FakeDQuantiser, _FakeAQuantiser
from quantlib.algorithms.qmodules.qmodules.qmodules import _QModule

class DianaBaseOperation:   
    @abstractmethod
    def map_scales(self, new_bitwidth=8, signed = True , HW_Behaviour=False): 
        pass
    
    def redefine_qhparams(self : _QModule, qrangespec:               QRangeSpecType):  
        assert(issubclass(type(self), _QModule))
        self._qrange = resolve_qrangespec(qrangespec)
        zero, n_levels, step, scale = create_qhparams(self._qrange)
        self.zero =  torch.tile(zero,     self._observer.broadcasting_shape)
        self.n_levels=  torch.tile(n_levels,     self._observer.broadcasting_shape)
        self.step=  torch.tile(step,    self._observer.broadcasting_shape)
        self.scale =  torch.tile(scale,   self._observer.broadcasting_shape)
        self._init_qhparams() # reinitialize scale and zero 
    def get_bitwidth(self): 
        pass


class IdentityType(enum.Enum):  
    default = enum.auto() # 8 Bits 
    AIMC_IN= enum.auto() # 7 bits
    AIMC_OUT= enum.auto() # 6 bits 
    DIGITAL_IN = enum.auto() # 8 bits (digital core)
    DIGITAL_OUT = enum.auto() # 8 bits 

class DIANALinear(QLinear , DianaBaseOperation): 
    def __init__(self, qrangespec : QRangeSpecType,
                 qgranularityspec : QGranularitySpecType,
                 qhparamsinitstrategyspec : QHParamsInitStrategySpecType,in_features : int, out_features: int, bias=False): 
                 super().__init__(qrangespec,
                 qgranularityspec,
                 qhparamsinitstrategyspec ,in_features=in_features, out_features=out_features, bias=bias) 
    def _register_qop(self): #used for autograd functions with non-standard backward gradients 
        self._qop = _FakeDQuantiser.apply 
    
    def _call_qop(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        bw_clip_lo = 2**round(math.log2((abs(self.clip_lo)/ (self.scale * self.step)).item())) #quantized clip lo and high
        bw_clip_hi =2**round(math.log2((abs(self.clip_hi)/ (self.scale * self.step)).item())) -1 
        if self.clip_lo < 0: 
            bw_clip_lo = -bw_clip_lo + 1
        return self._qop(x, bw_clip_lo,bw_clip_hi, self.step, self.scale)
    
    def map_scales(self, new_bitwidth=8, signed=True, HW_Behaviour=False):
        if HW_Behaviour: 
            self.redefine_qhparams({'bitwidth' : 7, 'signed': True}) 
        else : 
            self.redefine_qhparams({'bitwidth' : new_bitwidth, 'signed': signed})              
# Activations
class DIANAIdentity(QIdentity , DianaBaseOperation): # general purpose identity for harmoniser adds ( Quant operation )
    def __init__(self,
                 qrangespec:               QRangeSpecType,
                 qgranularityspec:         QGranularitySpecType,
                 qhparamsinitstrategyspec: QHParamsInitStrategySpecType ,
                 QuantizerType : IdentityType = IdentityType.default):

        super().__init__(qrangespec,
                 qgranularityspec,
                 qhparamsinitstrategyspec) 
        self._type =  QuantizerType
    def _register_qop(self): #used for autograd functions with non-standard backward gradients 
        self._qop = _FakeDQuantiser.apply 

    def _call_qop(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        bw_clip_lo = 2**round(math.log2((abs(self.clip_lo)/ (self.scale * self.step)).item())) #quantized clip lo and high
        bw_clip_hi =2**round(math.log2((abs(self.clip_hi)/ (self.scale * self.step)).item())) -1 
        if self.clip_lo < 0: 
            bw_clip_lo = -bw_clip_lo +1
        return self._qop(x, bw_clip_lo,bw_clip_hi, self.step, self.scale)
    def map_scales(self, new_bitwidth=8, signed=True, HW_Behaviour=False):
        if HW_Behaviour: 
            if self._type == IdentityType.default: 
                self.redefine_qhparams({'bitwidth' : 8, 'signed': True}) 
            elif self._type == IdentityType.AIMC_IN: 
                self.redefine_qhparams({'bitwidth' : 7, 'signed': True}) 
            elif self._type == IdentityType.AIMC_OUT: 
                self.redefine_qhparams({'bitwidth' : 6, 'signed': True}) 
            else:
                self.redefine_qhparams({'bitwidth' : 8, 'signed': True}) 
        else : 
            self.redefine_qhparams({'bitwidth' : new_bitwidth, 'signed': signed}) 
    def get_bitwidth(self):
        if self._type == IdentityType.default: 
            return 8 
        elif self._type == IdentityType.AIMC_IN: 
            return 7 
        elif self._type == IdentityType.AIMC_OUT: 
            return 6 
        else:
            return 8 
class DIANABiasIdentity(QIdentity , DianaBaseOperation) : 
    def __init__(self, qrangespec: QRangeSpecType, qgranularityspec: QGranularitySpecType, qhparamsinitstrategyspec: QHParamsInitStrategySpecType):

         super().__init__(qrangespec, qgranularityspec, qhparamsinitstrategyspec)
    def map_scales(self, new_bitwidth=8, signed=True, HW_Behaviour=False):
         return super().map_scales(new_bitwidth, signed, HW_Behaviour)

class DIANAReLU( PACTReLU  , DianaBaseOperation): 
    def stop_observing(self):
        super().stop_observing() 
        self.bw_clip_hi = 2**round(math.log2((abs(self.clip_hi)/ (self.scale * self.step)).item())) - 1
        # edit the scale

    def map_scales(self, new_bitwidth=8, signed=True, HW_Behaviour=False):

        if HW_Behaviour: 
            self.redefine_qhparams({'bitwidth' : 7, 'signed': True})

            #  clip here and freeze 
            self.freeze() 
            

            
        else : 
            self.redefine_qhparams({'bitwidth' : new_bitwidth, 'signed': signed}) 
# How I have it right now it will be a convolution in the digital core if it's not followed by a batch norm otherwise it's an analog core
class DIANAConv2d(QConv2d , DianaBaseOperation):
    
    def __init__(self,
                 qrangespec:               QRangeSpecType,
                 qgranularityspec:         QGranularitySpecType,
                 qhparamsinitstrategyspec: QHParamsInitStrategySpecType,
                 in_channels:              int,
                 out_channels:             int,
                 kernel_size:              Tuple[int, ...],
                 stride:                   Tuple[int, ...] = 1,
                 padding:                  int = 0,
                 dilation:                 Tuple[int, ...] = 1,
                 groups:                   int = 1 , 
                 bias : bool = False 
                 ):
                    self.is_analog = not bias # In linearopbn cannonocalisation the bias is set to none if bn follows conv 
                    self.bias_exists = False
                    super().__init__(qrangespec,
                         qgranularityspec,
                         qhparamsinitstrategyspec,
                         in_channels,
                         out_channels,
                         kernel_size,
                         stride,
                         padding,
                         dilation,
                         groups,
                         bias=False) 
                    if bias == True: 
                        self.qbiasdidentity = DIANAIdentity({'bitwidth': 8 , 'signed': True}, 'per-array', 'minmax')
                        self.register_parameter(name='qbias', param = torch.nn.parameter.Parameter( torch.rand(out_channels ) -0.5)) 
                        self.qbias.requires_grad = True
                        self.bias_exists = True
                        self.bias_shape_set = False
          
    def _register_qop(self): #used for autograd functions with non-standard backward gradients 
        if self.is_analog:
             self._qop = _FakeAQuantiser.apply
        else: 
            self._qop = _FakeDQuantiser.apply 
 
    def _call_qop(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self._qop(x, self.clip_lo, self.clip_hi, self.step, self.scale) 
    def forward(self , x: torch.Tensor): 
        out = 0 
        
        conv_out = super().forward(x) 
        if self.bias_exists: 
            out = self.qbiasdidentity(self.qbias.unsqueeze(1).unsqueeze(1).expand(conv_out.shape[1:]).unsqueeze(0).expand(conv_out.shape) ) # broadcasting  
        return conv_out  +out 

    def map_scales(self, new_bitwidth=8, signed=True, HW_Behaviour=False):
        if HW_Behaviour: 
            if (self.is_analog): 
                self.redefine_qhparams('ternary') 
            else: 
                self.redefine_qhparams({'bitwidth' : 8 , 'signed': True})   

        else : 
            self.redefine_qhparams({'bitwidth' : new_bitwidth, 'signed': signed})   

    @classmethod
    def from_fp_module(cls,
                       fpm:                      nn.Conv2d,
                       qrangespec:               QRangeSpecType,
                       qgranularityspec:         QGranularitySpecType,
                       qhparamsinitstrategyspec: QHParamsInitStrategySpecType) -> QConv2d:
        """Special constructor to build ``QConv2d``s from FP ``Conv2d``s."""
        shape = None
        if fpm.bias is not None: 
            shape = fpm.bias.shape

        qconv2d = cls(qrangespec,
                      qgranularityspec,
                      qhparamsinitstrategyspec,
                      in_channels=fpm.in_channels,
                      out_channels=fpm.out_channels,
                      kernel_size=fpm.kernel_size,
                      stride=fpm.stride,
                      padding=fpm.padding,
                      dilation=fpm.dilation,
                      groups=fpm.groups,
                      bias=(fpm.bias is not None))

        # copy parameters over
        qconv2d.weight.data.copy_(fpm.weight.data)
        if fpm.bias is not None:
            qconv2d.qbias.data.copy_(fpm.bias.data)

        return qconv2d


## Classes for emulation . Deprecated . Might leave those if someone wants to create custom models witht the quantized building blocks  
#class DQScaleBias(DianaModule): # input is quantized
    #def __init__(self, qrangespec:               QRangeSpecType,
                 #qgranularityspec:         QGranularitySpecType,
                 #qhparamsinitstrategyspec: QHParamsInitStrategySpecType,
                 #in_channels:             torch.Tensor): #channels * batch_size  
        #super().__init__() 
        #self.qinput = DBatchNormIdentity(qrangespec, qgranularityspec, qhparamsinitstrategyspec)
        #self.qscale = DIANAIdentity(qrangespec,
                 #qgranularityspec,
                 #qhparamsinitstrategyspec)
        #self.register_parameter(name='biases', param = torch.nn.parameter.Parameter( (torch.rand(in_channels) -0.5)*2))
        #self.register_parameter(name='weights', param = torch.nn.parameter.Parameter( (torch.rand(in_channels)-0.5 )* 2))
        #self.qidentity = DIANAIdentity(qrangespec,
                 #qgranularityspec,
                 #qhparamsinitstrategyspec) 

    #@property 
    #def qweight(self): 
        #return self.qscale(self.weights)
    #@property 
    #def qbias(self): 
        #return self.qidentity(self.biases)
    #def get_weight_scale(self): 
        #return self.qscale.scale
    #def get_bias_scale(self): 
        #return self.qidentity.scale
    #@classmethod
    #def from_fp_module(cls, bn_layer : nn.BatchNorm2d, qrangespec: QRangeSpecType, qgranularityspec: QGranularitySpecType, qhparamsinitstrategyspec: QHParamsInitStrategySpecType):
        #return DQScaleBias(qrangespec, qgranularityspec, qhparamsinitstrategyspec, bn_layer.num_features)
    #def forward(self, input):
        ### Broadcasting to get weights/biases in the correct format 
        #broadcasted_biases = self.qidentity(self.biases).unsqueeze(1).unsqueeze(1)
        #broadcasted_weights = self.qscale(self.weights).unsqueeze(1).unsqueeze(1)
       ## if (len(input.size())) > 1: 
            ##broadcasted_weights = broadcasted_weights.unsqueeze(1)
            ##broadcasted_biases= broadcasted_biases.unsqueeze(1)
            ##if(len(input.size())) > 2: 
                ##broadcasted_weights = broadcasted_weights.unsqueeze(1)
                ##broadcasted_biases= broadcasted_biases.unsqueeze(1)
        
        #broadcasted_weights = broadcasted_weights.expand(input.size())
        #broadcasted_biases = broadcasted_biases.expand(input.size())

        #return self.qinput(input) * broadcasted_weights+ broadcasted_biases
## pooling layer use regular pool and then have it go through Qidentity 
#class DQPool2d(DianaModule): # input quantized 

    #def __init__(self, qrangespec:               QRangeSpecType,
                 #qgranularityspec:         QGranularitySpecType,
                 #qhparamsinitstrategyspec: QHParamsInitStrategySpecType ,kernel_size= (1, 1), stride=None, padding=0, adaptive=False, output_size = None) : 
        #super().__init__() 
        #self.qinput = DIANAIdentity(qrangespec,qgranularityspec, qhparamsinitstrategyspec)
        
        #if not adaptive or output_size is None: 
            #self.avgpool = nn.AvgPool2d(kernel_size=kernel_size,stride = stride, padding=padding) # TODO: Implement the custom avgpool 
        #else: 
            #self.avgpool = nn.AdaptiveAvgPool2d(output_size=output_size) 
    #@classmethod
    #def from_fp_module(cls, pool_layer : Union[nn.AvgPool2d, nn.AdaptiveAvgPool2d], qrangespec: QRangeSpecType, qgranularityspec: QGranularitySpecType, qhparamsinitstrategyspec: QHParamsInitStrategySpecType):
        #adaptive = False 
        #output_size = None
        #if type(pool_layer) == nn.AdaptiveAvgPool2d:  
            #adaptive = True 
            #output_size = pool_layer.output_size 
     
        #return DQPool2d(qrangespec, qgranularityspec, qhparamsinitstrategyspec, pool_layer.kernel_size, stride =pool_layer.stride , padding = pool_layer.padding, adaptive=adaptive, output_size=output_size)
    #def forward(self, input) : 
        #out = self.avgpool(self.qinput(input))
        #return out 

#class DQFC(DianaModule): # output quantized and input as well #TODO do I Quantized bias ? 
    #def __init__(self,qrangespec:               QRangeSpecType,
                 #qgranularityspec:         QGranularitySpecType,
                 #qhparamsinitstrategyspec: QHParamsInitStrategySpecType , in_features: int , out_features: int): 
        #super().__init__()
        #self.qinput = DIANAIdentity(qrangespec, qgranularityspec, qhparamsinitstrategyspec)
        #self.qlinear = DIANALinear(qrangespec=qrangespec,qgranularityspec=qgranularityspec,qhparamsinitstrategyspec=qhparamsinitstrategyspec ,in_features=in_features, out_features=out_features)
        #self.register_parameter(name='biases', param = torch.nn.parameter.Parameter( torch.rand(out_features) -0.5)) # think of another way to initialise the biases 
        #self.qidentity = DIANAIdentity(qrangespec,
                 #qgranularityspec,
                 #qhparamsinitstrategyspec) 
        #self.qout = DIANAIdentity(qrangespec,
                 #qgranularityspec,
                 #qhparamsinitstrategyspec)
    #@classmethod
    #def from_fp_module(cls, linear_layer : nn.Linear, qrangespec: QRangeSpecType, qgranularityspec: QGranularitySpecType, qhparamsinitstrategyspec: QHParamsInitStrategySpecType):

        #return DQFC(qrangespec=qrangespec, qgranularityspec=qgranularityspec,qhparamsinitstrategyspec=qhparamsinitstrategyspec, in_features=linear_layer.in_features, out_features=linear_layer.out_features)

    #def forward(self, input : torch.Tensor): 
        #broadcasted_input = input.flatten(start_dim=1) 
        #broadcasted_biases = self.qidentity(self.biases).expand(input.size(0),)# broadcasted for batches 
        #return self.qout(self.qlinear(broadcasted_input)  + broadcasted_biases) # make sure input sizes match 

#class DQConv2d_d(DianaModule): # default bias false , implemented withoutput quantizer. 
    #def __init__(self, qrangespec:               QRangeSpecType,
                 #qgranularityspec:         QGranularitySpecType,
                 #qhparamsinitstrategyspec: QHParamsInitStrategySpecType ,
                 #in_channels:              int,
                 #out_channels:             int,
                 #kernel_size:              Tuple[int, ...],
                 #stride:                   Tuple[int, ...] = 1,
                 #padding:                  int = 0,
                 #dilation:                 Tuple[int, ...] = 1,
                 #groups:                   int = 1,
                 #bias : bool = False ):
        #super().__init__() 
        #self.qin = DIANAIdentity(qrangespec=qrangespec, qgranularityspec=qgranularityspec, qhparamsinitstrategyspec=qhparamsinitstrategyspec) 
        #self.qconv = DIANAConv2d(qrangespec=qrangespec,qgranularityspec=qgranularityspec,qhparamsinitstrategyspec=qhparamsinitstrategyspec,kernel_size=kernel_size,stride=stride, padding=padding,in_channels=in_channels ,out_channels=out_channels, dilation=dilation, groups=groups, bias = bias) 
        #self.qout = DIANAIdentity(qrangespec=qrangespec, qgranularityspec=qgranularityspec, qhparamsinitstrategyspec=qhparamsinitstrategyspec)
        #self.bias_enabled = bias 
        #if self.bias_enabled: 
            #self.register_parameter(name='biases', param = torch.nn.parameter.Parameter( torch.rand(out_channels) -0.5)) # think of another way to initialise the biases 
            #self.qbiasin = DIANAIdentity(qrangespec=qrangespec, qgranularityspec=qgranularityspec, qhparamsinitstrategyspec=qhparamsinitstrategyspec) 
 
    #def forward(self, input): 
        #broadcasted_bias = 0 
        #if self.bias_enabled: # still need to initialize the bias correctly. Right now it's not initialized correctly 
            #broadcasted_bias = self.qbiasin(self.biases)
        #output = self.qout(self.qconv(self.qin(input)) + broadcasted_bias)
        #return output

##In training scales are already matched no custom res_add is needed, but when doing inference scales need to be accounted for 



