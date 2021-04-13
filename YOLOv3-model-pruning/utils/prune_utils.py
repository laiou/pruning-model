import torch
from terminaltables import AsciiTable
from copy import deepcopy
import numpy as np
import torch.nn.functional as F


def get_sr_flag(epoch, sr):
    # return epoch >= 5 and sr
    return sr

#获取相应参与剪枝层次的索引
def parse_module_defs(module_defs):

    CBL_idx = []
    Conv_idx = []
    #这里提取conv_bn的层次索引添加到CBL_idx
    #单独的conv索引添加到conv_idx
    for i, module_def in enumerate(module_defs):
        if module_def['type'] == 'convolutional':
            if module_def['batch_normalize'] == '1':
                CBL_idx.append(i)
            else:
                Conv_idx.append(i)

    ignore_idx = set()
    #这里将shortcut层的输入层次的索引添加到ignore_idx
    for i, module_def in enumerate(module_defs):
        if module_def['type'] == 'shortcut':
            ignore_idx.add(i-1)
            identity_idx = (i + int(module_def['from']))
            if module_defs[identity_idx]['type'] == 'convolutional':
                ignore_idx.add(identity_idx)
            elif module_defs[identity_idx]['type'] == 'shortcut':
                ignore_idx.add(identity_idx - 1)
    #同时将84和96层也添加到ignore_idx中，不参与剪枝
    #这两个层次实际上就是upsample前面的两个卷积层
    ignore_idx.add(84)
    ignore_idx.add(96)
    #得到prune_idx的索引，既不是shortut层的输入的CBL层也不是upsample层前面两个CBL层的索引
    prune_idx = [idx for idx in CBL_idx if idx not in ignore_idx]

    return CBL_idx, Conv_idx, prune_idx

#获取具体的BN层权重，只提取参与剪枝层次的BN层的scales，最终抽取到的是对应scales的绝对值
def gather_bn_weights(module_list, prune_idx):
    #size_list抽取相应的bn权重的维度，一个通道对应一个Bn中的scale
    size_list = [module_list[idx][1].weight.data.shape[0] for idx in prune_idx]

    bn_weights = torch.zeros(sum(size_list))
    index = 0
    for idx, size in zip(prune_idx, size_list):
        #获取具体的scales的绝对值返回
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


class BNOptimizer():

    @staticmethod
    def updateBN(sr_flag, module_list, s, prune_idx):
        if sr_flag:
            for idx in prune_idx:
                # Squential(Conv, BN, Lrelu)
                #定位到对应参与剪枝层次中的BN模块
                bn_module = module_list[idx][1]
                #选择BN模块中的缩放因子进行稀疏性正则化，也就是在原来的缩放因子数值上再加上+s或者-s，取决于scales的正负
                #假设原来的BN缩放因子是scale,现在就变成了scale+s或者scales-s
                #这里跟论文实现有关，整体loss = train_loss +s*求和L1(scales)，所以在更新scales的时候还要考虑上这部分的梯度
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

#根据上一层剪枝的结果，获取当前输入的mask矩阵
def get_input_mask(module_defs, idx, CBLidx2mask):
    #idx ==0，就是输入层，3通道输入数据都保留
    if idx == 0:
        return np.ones(3)
    #这里没有看upsample层是因为upsample层后面没有接卷积层（yolov3中）
    #上一层是卷积。。则当前层输入的mask就是上一层剪枝的mask
    if module_defs[idx - 1]['type'] == 'convolutional':
        return CBLidx2mask[idx - 1]
        #如果上一层是shortcut，则当前层输入的mask就是shortcut上一层的剪枝mask
    elif module_defs[idx - 1]['type'] == 'shortcut':
        return CBLidx2mask[idx - 2]
        #如果上一层是route层
    elif module_defs[idx - 1]['type'] == 'route':
        route_in_idxs = []
        #遍历参与route的每一个层次
        for layer_i in module_defs[idx - 1]['layers'].split(","):
            if int(layer_i) < 0:
                #抽取对应层次的idx
                route_in_idxs.append(idx - 1 + int(layer_i))
            else:
                route_in_idxs.append(int(layer_i))
        #如果只有一层参与route
        if len(route_in_idxs) == 1:
            #那么就返回那一层的剪枝mask矩阵作为当前层的输入mask
            return CBLidx2mask[route_in_idxs[0]]
            #如果是两个层次参与mask(yolov3中只有这两种情况)
        elif len(route_in_idxs) == 2:
            #拼接两个层次的剪枝mask矩阵，这里跟具体的网络结构有关，两个层次的route层都不是直接的卷积层输出，中间要么隔了upsample层，要么隔了shortcut层
            #所以是in_dex-1
            return np.concatenate([CBLidx2mask[in_idx - 1] for in_idx in route_in_idxs])
        else:
            print("Something wrong with route module!")
            raise Exception

#用调整过的原来模型的权重初始化剪枝模型
def init_weights_from_loose_model(compact_model, loose_model, CBL_idx, Conv_idx, CBLidx2mask):
    #循环遍历每一个CBL模块，注意不是所有的CBL都参与了剪枝
    for idx in CBL_idx:
        #定位新模型和剪枝模型中对应层次的位置
        compact_CBL = compact_model.module_list[idx]
        loose_CBL = loose_model.module_list[idx]
        #获取对应输出通道的索引，利用np.argwhere抽取权重mask矩阵中非零元素的索引
        out_channel_idx = np.argwhere(CBLidx2mask[idx])[:, 0].tolist()
        #定位BN层
        compact_bn, loose_bn         = compact_CBL[1], loose_CBL[1]
        #对BN层相应保存的通道上的参数进行复制
        compact_bn.weight.data       = loose_bn.weight.data[out_channel_idx].clone()
        compact_bn.bias.data         = loose_bn.bias.data[out_channel_idx].clone()
        compact_bn.running_mean.data = loose_bn.running_mean.data[out_channel_idx].clone()
        compact_bn.running_var.data  = loose_bn.running_var.data[out_channel_idx].clone()
        #接下来就是对conv的参数进行赋值了
        #获取相应的输入的mask矩阵，根据上一层的剪枝结果获得当前层的输入的mask矩阵
        #具体实现参考utils/prune_utils.py
        input_mask = get_input_mask(loose_model.module_defs, idx, CBLidx2mask)
        #提取相应的保留的输入通道的索引
        in_channel_idx = np.argwhere(input_mask)[:, 0].tolist()
        #定位卷积层的位置
        compact_conv, loose_conv = compact_CBL[0], loose_CBL[0]
        #这里两步骤分别是根据上一层剪枝结果，抽取相应的weights数据，然后在这个基础上根据当前层剪枝结果获得最终权值
        tmp = loose_conv.weight.data[:, in_channel_idx, :, :].clone()
        compact_conv.weight.data = tmp[out_channel_idx, :, :, :].clone()
    #然后在循环处理正常的卷积层，逻辑是一样的
    for idx in Conv_idx:
        compact_conv = compact_model.module_list[idx][0]
        loose_conv = loose_model.module_list[idx][0]

        input_mask = get_input_mask(loose_model.module_defs, idx, CBLidx2mask)
        in_channel_idx = np.argwhere(input_mask)[:, 0].tolist()
        compact_conv.weight.data = loose_conv.weight.data[:, in_channel_idx, :, :].clone()
        compact_conv.bias.data   = loose_conv.bias.data.clone()

#对模型进行剪枝，这里的模型实际上是yolov3
def prune_model_keep_size(model, prune_idx, CBL_idx, CBLidx2mask):

    pruned_model = deepcopy(model)
    #循环遍历每一个参与剪枝的层次
    for idx in prune_idx:
        #获取对应的权重mask矩阵
        mask = torch.from_numpy(CBLidx2mask[idx]).cuda()
        bn_module = pruned_model.module_list[idx][1]
        #将相应的BN层权重和mask矩阵做乘法，将相应权重置0
        bn_module.weight.data.mul_(mask)
        #这里将scale置0的通道或者说是将剪除掉的通道上的偏置通过激活函数。。正常是y = wx+b，但是被剪除的通道上w是0，只剩下b了
        #用1-mask来形成一个新的mask来得到被剪除通道上的偏置
        activation = F.leaky_relu((1 - mask) * bn_module.bias.data, 0.1)

        # 两个上采样层前的卷积层
        #next_id_list存储当前参与剪枝层次的下一个层次，或者说以当前层输出作为输入的层次，这里只看下一个卷积层，yolov3中的层次类别不多。。
        next_idx_list = [idx + 1]
        #因为是yolov3模型，所以84层和96层分别是两个upsample层前面的卷积层，实际上yolov3的卷积层，不带BN的只有yolo层前面的那一个卷积层
        #当idx是79和91的时候，刚好是产生多尺度检测的分支节点，所以输出除了原本的idx+1之外。。还要加上后面的分支的输出节点
        if idx == 79:
            next_idx_list.append(84)
        elif idx == 91:
            next_idx_list.append(96)
        #循环遍历相应的输出层
        for next_idx in next_idx_list:
            #定位相应下一个层次的卷积操作
            next_conv = pruned_model.module_list[next_idx][0]
            #将相应的卷积层权重求和，这里比如原来的卷积层权重维度是[kernel_numbel,input_channel,size,size]-->[kernel_numbel,input_channel]
            conv_sum = next_conv.weight.data.sum(dim=(2, 3))
            #将上面的到的被剪除的层次的偏置的激活值跟这里的权重矩阵做矩阵乘法，结果调整成一维
            #这里的激活矩阵调整之后的维度是[input_channel,1],所以最终的offset的维度就是[kernel_numel,1]
            #offset代表的是上一层被剪除的通道中的偏置输入当前层进行的计算
            offset = conv_sum.matmul(activation.reshape(-1, 1)).reshape(-1)
            #如果下一层也是一个CBL模块
            if next_idx in CBL_idx:
                #定位其中的BN层
                next_bn = pruned_model.module_list[next_idx][1]
                #将其中BN层的中的mean减去上面的offset
                #这里的逻辑是为了将前一层剪除的通道上偏置的值的计算结果直接添加到下一层对应的位置上，BN层的计算是x-mean,然后除以方差等等。。
                #所以这里用mean减去offset,最终实际上还是将上一层剪除通道上偏置的计算值加到了对应的结果里面，可以把这里的均值当成某个通道上的偏置理解也行
                next_bn.running_mean.data.sub_(offset)
            else:
                #如果下一层卷积层不是CBL模块，其实就是yolo层前面的卷积层了
                #在他的偏置上加上上面的offset,实际上将上一层剪除的通道中的偏置的计算累加到这一层的偏置里面了
                next_conv.bias.data.add_(offset)
        #最后将当前剪枝层的偏置数据乘上mask矩阵，将相应的偏置置0
        bn_module.bias.data.mul_(mask)

    return pruned_model

#获得BN剪枝的mask矩阵
def obtain_bn_mask(bn_module, thre):

    thre = thre.cuda()
    #比较BN层权重的绝对值和阈值thre的大小，如果|scales|>=thre,返回1，否则返回0
    mask = bn_module.weight.data.abs().ge(thre).float()

    return mask
