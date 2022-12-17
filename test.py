filename = "zoo/cifar10/resnet20/FQ-epoch=01-val_acc=0.9869.ckpt" 
from DianaModules.core.Operations import AnalogConv2d
from DianaModules.utils.BaseModules import DianaModule
from DianaModules.models.cifar10.LargeResnet import resnet20
import pytorch_lightning as pl
import torch
from DianaModules.utils.compression.QuantStepper import QuantDownStepper 
from DianaModules.utils.serialization.Loader import ModulesLoader  
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as ds
cfg_path = "serialized_models/resnet20.yaml" 
fp_path  = "zoo/cifar10/resnet20/FP_weights.pth"
#define dataset 
train_dataset =  ds.CIFAR10('./data/cifar10/train', train =True ,download=False, transform=torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(32, 4),torchvision.transforms.ToTensor() ,torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
test_dataset =  ds.CIFAR10('./data/cifar10/validation', train =False,download=False, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))] ) )

#define dataloader 
train_dataloader = DataLoader(train_dataset , pin_memory=True , num_workers=28, shuffle=False, batch_size=256)
val_dataloader = DataLoader(test_dataset, pin_memory=True , num_workers=28 , batch_size=256)
#Module laoder 
loader = ModulesLoader()
descriptions = loader.load(cfg_path)
#quantize model 
trainer = pl.Trainer(accelerator="gpu", strategy = "dp" ,devices=-1)   
#instantiate model and load pre-trained fp 
module = resnet20()  
module.load_state_dict(DianaModule.remove_dict_prefix(torch.load(fp_path, map_location="cpu")["state_dict"])) 
module.eval() 


# edit the from_trainedfp_model function to change the intitial quantization parameters 
model = DianaModule(DianaModule.from_trainedfp_model(module , modules_descriptors=descriptions))
model.attach_train_dataloader(train_dataloader, torch.Tensor([0.03125])) 
model.attach_quantization_dataloader(train_dataloader) 
model.set_quantized(activations=False) 
model.gmodule.load_state_dict(DianaModule.remove_prefixes(torch.load(filename, map_location="cpu")["state_dict"]))  
stepper = QuantDownStepper(model, 6 , initial_quant={"bitwidth": 8, "signed" :True}, target_quant="ternary") 
# View weights information before and weights information after (Before and after ternary)
# Before 
for i in range(6+1) : 
    for _ , mod in model.named_modules(): 
        if (isinstance(mod, AnalogConv2d)) :  
            #info about fp weights
            mean = torch.mean(mod.weight)
            var  = torch.var(mod.weight)
            max  = torch.max(mod.weight)
            min  = torch.min(mod.weight)
            print(f"At {8-i} Bits, floating point information: \n mean: {mean} \n var: {var} \n max: {max} \n min: {min}")
            #info about true quantized weights 
            mean = torch.mean(mod.qweight/mod.scale)
            var  = torch.var (mod.qweight/mod.scale)
            max  = torch.max (mod.qweight/mod.scale)
            min  = torch.min (mod.qweight/mod.scale)
            print(f"At {8-i} Bits, true-quantized information: \n mean: {mean} \n var: {var} \n max: {max} \n min: {min}")
            break 
    print(f"Testing {8-i} bits acc")
    trainer.validate(model ,val_dataloader)
    model.set_quantized(False)  # Re determine the thresholds
    stepper.step() 
