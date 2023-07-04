import torch
from models.model_torchscript import STBVMM


ckpt_path = 'ckpt/ckpt_e49.pth.tar'

# Create model
model = STBVMM(img_size=384, patch_size=1, in_chans=3,
                embed_dim=192, depths=[6, 6, 6, 6, 6, 6], num_heads=[6, 6, 6, 6, 6, 6],
                window_size=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                norm_layer=torch.nn.LayerNorm, ape=False, patch_norm=True,
                use_checkpoint=False, img_range=1., resi_connection='1conv',
                manipulator_num_resblk=1)

# Create a dummy input
exampleA = torch.randn((1,3,192,192))
exampleB = torch.randn((1,3,192,192))
mag_factor = torch.tensor(0.2)

# Trace the model
traced_script_module = torch.jit.trace(model, [exampleA, exampleB, mag_factor])

# Save the model
traced_script_module.save("STB-VMM.pt")
