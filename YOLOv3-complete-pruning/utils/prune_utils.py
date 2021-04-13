import torch
from terminaltables import AsciiTable
from copy import deepcopy
import numpy as np
import torch.nn.functional as F

 
def get_sr_flag(epoch, sr):
    # return epoch >= 5 and sr
    return sr
#对yolov3tiny剪枝的剪枝层次索引获取
def parse_module_defs3(module_defs):

    CBL_idx = []
    Conv_idx = []
    for i, module_def in enumerate(module_defs):
        if module_def['type'] == 'convolutional':
            if module_def['batch_normalize'] == '1':
                CBL_idx.append(i)
            else:
                Conv_idx.append(i)

    ignore_idx = set()
#只有一个upsample层
    ignore_idx.add(18)
    

    prune_idx = [idx for idx in CBL_idx if idx not in ignore_idx]

    return CBL_idx, Conv_idx, prune_idx
#对shortcut进行剪枝时的索引获取    
def parse_module_defs2(module_defs):

    CBL_idx = []
    Conv_idx = []
    shortcut_idx=dict()
    shortcut_all=set()
    for i, module_def in enumerate(module_defs):
        if module_def['type'] == 'convolutional':
            if module_def['batch_normalize'] == '1':
                CBL_idx.append(i)
            else:
                Conv_idx.append(i)

    ignore_idx = set()
    for i, module_def in enumerate(module_defs):
        if module_def['type'] == 'shortcut':
            #identity_idx就是shortcut另一个输入层的id,还有一个就是i-1
            identity_idx = (i + int(module_def['from']))
            #有几个shortcut的输入是一个卷积层
            if module_defs[identity_idx]['type'] == 'convolutional':
                
                #ignore_idx.add(identity_idx)
                #shortcut_idx中存储的是shortcut层输入的卷积层索引
                #也就是short_idx中存储的是：shortcut_id -1 :: input_conv_id
                #shortcut_id -1 也就是short_cut上一层卷积的id
                shortcut_idx[i-1]=identity_idx
                shortcut_all.add(identity_idx)
            elif module_defs[identity_idx]['type'] == 'shortcut':
                
                #ignore_idx.add(identity_idx - 1)
                #当另一个shortcut1作为输入时，索引变成shortcut1的上一个卷积层id
                shortcut_idx[i-1]=identity_idx-1
                shortcut_all.add(identity_idx-1)
            #将shortcut的上一层的id添加到short_all中
            shortcut_all.add(i-1)
    #上采样层前的卷积层不裁剪
    ignore_idx.add(84)
    ignore_idx.add(96)

    prune_idx = [idx for idx in CBL_idx if idx not in ignore_idx]

    return CBL_idx, Conv_idx, prune_idx,shortcut_idx,shortcut_all
#获取正常剪枝和规整剪枝的层次索引，这里不参与卷积的是upsample前的卷积层和shortcut前的卷积层，以及yolo层前的卷积层（不带bn的卷积）
def parse_module_defs(module_defs):
    #cbl_idx索引的是带BN的卷积层
    CBL_idx = []
    #cinv_idx索引的是不带bn的卷积层
    Conv_idx = []
    for i, module_def in enumerate(module_defs):
        if module_def['type'] == 'convolutional':
            if module_def['batch_normalize'] == '1':
                CBL_idx.append(i)
            else:
                Conv_idx.append(i)
    ignore_idx = set()
    #shortcut层前面的CBL模块不参与剪枝
    for i, module_def in enumerate(module_defs):
        if module_def['type'] == 'shortcut':
            ignore_idx.add(i-1)
            identity_idx = (i + int(module_def['from']))
            if module_defs[identity_idx]['type'] == 'convolutional':
                ignore_idx.add(identity_idx)
            elif module_defs[identity_idx]['type'] == 'shortcut':
                ignore_idx.add(identity_idx - 1)
    #上采样层前的卷积层不裁剪
    ignore_idx.add(84)
    ignore_idx.add(96)

    prune_idx = [idx for idx in CBL_idx if idx not in ignore_idx]

    return CBL_idx, Conv_idx, prune_idx

#抽取参与剪枝的CBL模块中bn层的scales的绝对值
def gather_bn_weights(module_list, prune_idx):

    size_list = [module_list[idx][1].weight.data.shape[0] for idx in prune_idx]

    bn_weights = torch.zeros(sum(size_list))
    index = 0
    for idx, size in zip(prune_idx, size_list):
        bn_weights[index:(index + size)] = module_list[idx][1].weight.data.abs().clone()
        index += size

    return bn_weights


def write_cfg(cfg_file, module_defs):

    with open(cfg_file, 'w') as f:
        for module_def in module_defs:
            f.write(f"[{module_def['type']}]\n")
            for key, value in module_def.items():
                if key != 'type':
                    f.write(f"{key}={value}\n")
            f.write("\n")
    return cfg_file

#调整Bn层scales的梯度
class BNOptimizer():

    @staticmethod
    def updateBN(sr_flag, module_list, s, prune_idx):
        if sr_flag:
            for idx in prune_idx:
                # Squential(Conv, BN, Lrelu)
                bn_module = module_list[idx][1]
                #在梯度中添加稀疏化惩罚的梯度，也就是l1的梯度
                bn_module.weight.grad.data.add_(s * torch.sign(bn_module.weight.data))  # L1


def obtain_quantiles(bn_weights, num_quantile=5):

    sorted_bn_weights, i = torch.sort(bn_weights)
    total = sorted_bn_weights.shape[0]
    quantiles = sorted_bn_weights.tolist()[-1::-total//num_quantile][::-1]
    print("\nBN weights quantile:")
    quantile_table = [
        [f'{i}/{num_quantile}' for i in range(1, num_quantile+1)],
        ["%.3f" % quantile for quantile in quantiles]
    ]
    print(AsciiTable(quantile_table).table)

    return quantiles


def get_input_mask(module_defs, idx, CBLidx2mask):

    if idx == 0:
        return np.ones(3)

    if module_defs[idx - 1]['type'] == 'convolutional':
        return CBLidx2mask[idx - 1]
    elif module_defs[idx - 1]['type'] == 'shortcut':
        return CBLidx2mask[idx - 2]
    elif module_defs[idx - 1]['type'] == 'route':
        route_in_idxs = []
        for layer_i in module_defs[idx - 1]['layers'].split(","):
            if int(layer_i) < 0:
                route_in_idxs.append(idx - 1 + int(layer_i))
            else:
                route_in_idxs.append(int(layer_i))
        if len(route_in_idxs) == 1:
            return CBLidx2mask[route_in_idxs[0]]
        elif len(route_in_idxs) == 2:
            return np.concatenate([CBLidx2mask[in_idx - 1] for in_idx in route_in_idxs])
        else:
            print("Something wrong with route module!")
            raise Exception

#用剪枝后的权重填充剪枝后的模型结构
def init_weights_from_loose_model(compact_model, loose_model, CBL_idx, Conv_idx, CBLidx2mask):

    for idx in CBL_idx:
        compact_CBL = compact_model.module_list[idx]
        loose_CBL = loose_model.module_list[idx]
        out_channel_idx = np.argwhere(CBLidx2mask[idx])[:, 0].tolist()

        compact_bn, loose_bn         = compact_CBL[1], loose_CBL[1]
        compact_bn.weight.data       = loose_bn.weight.data[out_channel_idx].clone()
        compact_bn.bias.data         = loose_bn.bias.data[out_channel_idx].clone()
        compact_bn.running_mean.data = loose_bn.running_mean.data[out_channel_idx].clone()
        compact_bn.running_var.data  = loose_bn.running_var.data[out_channel_idx].clone()

        input_mask = get_input_mask(loose_model.module_defs, idx, CBLidx2mask)
        in_channel_idx = np.argwhere(input_mask)[:, 0].tolist()
        compact_conv, loose_conv = compact_CBL[0], loose_CBL[0]
        tmp = loose_conv.weight.data[:, in_channel_idx, :, :].clone()
        compact_conv.weight.data = tmp[out_channel_idx, :, :, :].clone()

    for idx in Conv_idx:
        compact_conv = compact_model.module_list[idx][0]
        loose_conv = loose_model.module_list[idx][0]

        input_mask = get_input_mask(loose_model.module_defs, idx, CBLidx2mask)
        in_channel_idx = np.argwhere(input_mask)[:, 0].tolist()
        compact_conv.weight.data = loose_conv.weight.data[:, in_channel_idx, :, :].clone()
        compact_conv.bias.data   = loose_conv.bias.data.clone()

#调整剪枝的模型，处理剪枝之后，也就是BN层的scales被置0之后的偏置
def prune_model_keep_size(model, prune_idx, CBL_idx, CBLidx2mask):

    pruned_model = deepcopy(model)
    #循环遍历每一个剪枝层
    for idx in prune_idx:
        #获得对应层的mask矩阵
        mask = torch.from_numpy(CBLidx2mask[idx]).cuda()
        #定位相应的Bn层位置
        bn_module = pruned_model.module_list[idx][1]
        #将相应scales置0
        bn_module.weight.data.mul_(mask)
        #计算剪掉的那些层的偏置的激活
        activation = F.leaky_relu((1 - mask) * bn_module.bias.data, 0.1)

        # 两个上采样层前的卷积层
        #定位下一个卷积层的idx
        next_idx_list = [idx + 1]
        #这里是upsample前的两个卷积层，idx是79和91的时候出现分支，还要加上分支的输出层次
        if idx == 79:
            next_idx_list.append(84)
        elif idx == 91:
            next_idx_list.append(96)
        #循环遍历全部的下一层
        for next_idx in next_idx_list:
            #定位相应的层次的位置
            next_conv = pruned_model.module_list[next_idx][0]
            #将相应的卷积层权重求和，这里比如原来的卷积层权重维度是[kernel_numbel,input_channel,size,size]-->[kernel_numbel,input_channel]
            #将上面的到的被剪除的层次的偏置的激活值跟这里的权重矩阵做矩阵乘法，结果调整成一维
            #这里的激活矩阵调整之后的维度是[input_channel,1],所以最终的offset的维度就是[kernel_numel,1]
            #offset代表的是上一层被剪除的通道中的偏置输入当前层进行的计算
            conv_sum = next_conv.weight.data.sum(dim=(2, 3))
            offset = conv_sum.matmul(activation.reshape(-1, 1)).reshape(-1)
            #如果是CBL模块
            if next_idx in CBL_idx:
                #将上面的offset添加到其中的BN层的均值中，虽然是减去这个只值，实际上是把这个结果加进去了
                next_bn = pruned_model.module_list[next_idx][1]
                next_bn.running_mean.data.sub_(offset)
            else:
                #这里需要注意的是，对于convolutionnal，如果有BN，则该层卷积层不使用bias，如果无BN，则使用bias
                #不是CBL模块，直接加到偏置上
                next_conv.bias.data.add_(offset)
        #然后将相应的BN层或者说卷积的偏置乘上mask，同样剪除偏置
        bn_module.bias.data.mul_(mask)

    return pruned_model


def obtain_bn_mask(bn_module, thre):

    thre = thre.cuda()
    #比较权重的绝对值和thre的大小，返回1或者0
    mask = bn_module.weight.data.abs().ge(thre).float()

    return mask
