import os
from glob import glob
from collections import OrderedDict
from pytorch_lightning.utilities.cloud_io import load as pl_load


def get_state_dict_from_checkpoint(log_dir, fold_num, model_name):
    ckpt_path = glob(os.path.join(log_dir, f'checkpoints/*fold{fold_num}*.ckpt'))[0]
    state_dict = pl_load(ckpt_path, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    did_distillation = bool(sum(['_teacher_model' in k for k in state_dict.keys()]))
    state_dict = OrderedDict((k.replace('model.model.', 'model2.').replace('model.', '').replace('model2.', 'model.')
                              if 'model.' in k else k, v) for k, v in
                             state_dict.items() if '_teacher_model' not in k)
    num_classes = 5
    if did_distillation and (model_name != 'deit_base_distilled_patch16_384'):
        num_classes = 10
    return state_dict, num_classes

def freeze_top_layers(model, model_name):
    if 'efficientnet' in model_name:
        targets = ['blocks.6', 'conv_head', 'bn2', 'classifier']
    elif 'resnext50_32x4d' in model_name:
        targets = ['layer4', 'fc']
    else:
        raise ValueError(f"{model_name} can't be freezed")

    for name, param in model.named_parameters():
        if param.requires_grad:
            is_freeze = not bool(sum([name.startswith(target) for target in targets]))
            if is_freeze:
                param.requires_grad = False
    return model