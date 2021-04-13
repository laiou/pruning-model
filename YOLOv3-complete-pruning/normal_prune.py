from models import *
from utils.utils import *
import torch
import numpy as np
from copy import deepcopy
from test import test
from terminaltables import AsciiTable
import time
from utils.utils import *
from utils.prune_utils import *
import os


class opt():
    model_def = "cfg/yolov3-hand.cfg"
    data_config = "cfg/oxfordhand.data"
    model = 'weights/last.pt'

#指定GPU
#torch.cuda.set_device(2)
percent = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Darknet(opt.model_def).to(device)

if opt.model:
    if opt.model.endswith(".pt"):
        model.load_state_dict(torch.load(opt.model, map_location=device)['model'])
    else:
        _ = load_darknet_weights(model, opt.model)
        

data_config = parse_data_cfg(opt.data_config)

valid_path = data_config["valid"]
class_names = load_classes(data_config["names"])


eval_model = lambda model:test(model=model,cfg=opt.model_def, data=opt.data_config)


obtain_num_parameters = lambda model:sum([param.nelement() for param in model.parameters()])

#这个不应该注释掉，等会要恢复
with torch.no_grad():
    origin_model_metric = eval_model(model)
origin_nparameters = obtain_num_parameters(model)
#抽取相应的参与剪枝的CBL模块的id，不包含Bn的卷积层id，以及全部的CBL索引
CBL_idx, Conv_idx, prune_idx= parse_module_defs(model.module_defs)


#抽取参与剪枝的CBL中BN层scales的绝对值
bn_weights = gather_bn_weights(model.module_list, prune_idx)

#torch.sort返回二维列表，第一维是排序后的值列表，第二维是排序后的值列表对应的索引
#对上面得到的绝对值进行排序，注意是升序排序
sorted_bn = torch.sort(bn_weights)[0]


#避免剪掉所有channel的最高阈值(每个BN层的gamma的最大值的最小值即为阈值上限)
#计算剪枝阈值的最小阈值，避免剪除某一层的全部通道
highest_thre = []
for idx in prune_idx:
    #.item()可以得到张量里的元素值
    #highest_thre收集的是参与剪枝的cbl中bn层scales绝对值中的最大值
    highest_thre.append(model.module_list[idx][1].weight.data.abs().max().item())
#找到所有绝对值中最小的那个
highest_thre = min(highest_thre)

# 找到highest_thre对应的下标对应的百分比
#计算相应的最小的绝对值的索引在整个BN_weights中的比例位置，作为剪枝比例的最小阈值
percent_limit = (sorted_bn==highest_thre).nonzero().item()/len(bn_weights)

print(f'Threshold should be less than {highest_thre:.4f}.')
print(f'The corresponding prune ratio is {percent_limit:.3f}.')


# 该函数有很重要的意义：
# ①先用深拷贝将原始模型拷贝下来，得到model_copy
# ②将model_copy中，BN层中低于阈值的α参数赋值为0
# ③在BN层中，输出y=α*x+β，由于α参数的值被赋值为0，因此输入仅加了一个偏置β
# ④很神奇的是，network slimming中是将α参数和β参数都置0，该处只将α参数置0，但效果却很好：其实在另外一篇论文中，已经提到，可以先将β参数的效果移到
# 下一层卷积层，再去剪掉本层的α参数

# 该函数用最简单的方法，让我们看到了，如何快速看到剪枝后的效果


#这里将遍历参与剪枝的cbl层次，将其中相应的bn层的scales的绝对值小于剪枝阈值的置0，测试剪枝后模型的效果
def prune_and_eval(model, sorted_bn, percent=.0):
    #深拷贝一个模型的副本
    model_copy = deepcopy(model)
    #根据剪枝比例计算相应的剪枝阈值的索引
    thre_index = int(len(sorted_bn) * percent)
    #获得α参数的阈值，小于该值的α参数对应的通道，全部裁剪掉
    #抽取剪枝的阈值
    thre = sorted_bn[thre_index]

    print(f'Channels with Gamma value less than {thre:.4f} are pruned!')

    remain_num = 0
    #遍历全部参与剪枝的层次
    for idx in prune_idx:
        #定位其中的BN层的位置
        bn_module = model_copy.module_list[idx][1]
        #计算当前BN层的mask矩阵，具体实现参考utils/prune_utils.py
        #mask是一个只包括0和1的矩阵
        mask = obtain_bn_mask(bn_module, thre)

        remain_num += int(mask.sum())
        #将BN层权重和mask做乘法，将小于阈值的权值变成0
        bn_module.weight.data.mul_(mask)
    with torch.no_grad():
        mAP = eval_model(model_copy)[1].mean()

    print(f'Number of channels has been reduced from {len(sorted_bn)} to {remain_num}')
    print(f'Prune ratio: {1-remain_num/len(sorted_bn):.3f}')
    print(f'mAP of the pruned model is {mAP:.4f}')

    return thre

#测试剪枝后模型的效果
threshold = prune_and_eval(model, sorted_bn, percent)



#****************************************************************
#虽然上面已经能看到剪枝后的效果，但是没有生成剪枝后的模型结构，因此下面的代码是为了生成新的模型结构并拷贝旧模型参数到新模型


#%%
#获取卷积核mask矩阵，最终返回的num_filters中存储的是每一个CBL模块中保留的通道数，filters_mask是每一个CBL模块的mask矩阵
def obtain_filters_mask(model, thre, CBL_idx, prune_idx):

    pruned = 0
    total = 0
    num_filters = []
    filters_mask = []
    #CBL_idx存储的是所有带BN的卷积层（YOLO层的前一层卷积层是不带BN的）
    #遍历全部的CBL_idx
    for idx in CBL_idx:
        bn_module = model.module_list[idx][1]
        #遍历参与剪枝的CBL模块
        if idx in prune_idx:
            #计算相应的Bn层的mask
            mask = obtain_bn_mask(bn_module, thre).cpu().numpy()
            remain = int(mask.sum())
            #统计一共剪除的通道数
            pruned = pruned + mask.shape[0] - remain

            if remain == 0:
                print("Channels would be all pruned!")
                raise Exception

            print(f'layer index: {idx:>3d} \t total channel: {mask.shape[0]:>4d} \t '
                  f'remaining channel: {remain:>4d}')
        else:
            #对于不参与剪枝的CBL模块，mask就是全1的矩阵
            mask = np.ones(bn_module.weight.data.shape)
            remain = mask.shape[0]

        total += mask.shape[0]
        num_filters.append(remain)
        filters_mask.append(mask.copy())

    #因此，这里求出的prune_ratio,需要裁剪的α参数/cbl_idx中所有的α参数
    prune_ratio = pruned / total
    print(f'Prune channels: {pruned}\tPrune ratio: {prune_ratio:.3f}')

    return num_filters, filters_mask
#获取卷积核的mask矩阵，通过Bn层剪除某些通道之后，对相应的层次的卷积核是有影响的，所以最终生成模型的时候需要删除相应的卷积核
num_filters, filters_mask = obtain_filters_mask(model, threshold, CBL_idx, prune_idx)


#CBLidx2mask存储CBL_idx中，每一层BN层对应的mask
#这里将每一个CBL模块的idx和相应的mask矩阵绑定，变成一个字典，这里面的mask实际上还是Bn层的mask
CBLidx2mask = {idx: mask for idx, mask in zip(CBL_idx, filters_mask)}
#对模型进行剪枝，这里只是处理剪枝之后相应层次的遗留偏置，将对应的偏置数据处理到下一层中，这个时候模型结构实际上还是原来的模型结构
#具体参考utils/prune_utils.py
pruned_model = prune_model_keep_size(model, prune_idx, CBL_idx, CBLidx2mask)




with torch.no_grad():
    mAP = eval_model(pruned_model)[1].mean()
print('after prune_model_keep_size map is {}'.format(mAP))


#获得原始模型的module_defs，并修改该defs中的卷积核数量
#复制原来的模型结构
compact_module_defs = deepcopy(model.module_defs)
#调整剪枝后的卷积核数量
for idx, num in zip(CBL_idx, num_filters):
    assert compact_module_defs[idx]['type'] == 'convolutional'
    compact_module_defs[idx]['filters'] = str(num)


#利用剪枝的模型结构和超参数创建新模型
compact_model = Darknet([model.hyperparams.copy()] + compact_module_defs).to(device)
#获取剪枝后模型的参数
compact_nparameters = obtain_num_parameters(compact_model)
#用剪枝后的权重填充剪枝后的模型结构，具体实现参考utils/prune_utils.py
init_weights_from_loose_model(compact_model, pruned_model, CBL_idx, Conv_idx, CBLidx2mask)


random_input = torch.rand((16, 3, 416, 416)).to(device)

def obtain_avg_forward_time(input, model, repeat=200):

    model.eval()
    start = time.time()
    with torch.no_grad():
        for i in range(repeat):
            output = model(input)
    avg_infer_time = (time.time() - start) / repeat

    return avg_infer_time, output

pruned_forward_time, pruned_output = obtain_avg_forward_time(random_input, pruned_model)
compact_forward_time, compact_output = obtain_avg_forward_time(random_input, compact_model)



# 在测试集上测试剪枝后的模型, 并统计模型的参数数量
with torch.no_grad():
    compact_model_metric = eval_model(compact_model)


# 比较剪枝前后参数数量的变化、指标性能的变化
metric_table = [
    ["Metric", "Before", "After"],
    ["mAP", f'{origin_model_metric[1].mean():.6f}', f'{compact_model_metric[1].mean():.6f}'],
    ["Parameters", f"{origin_nparameters}", f"{compact_nparameters}"],
    ["Inference", f'{pruned_forward_time:.4f}', f'{compact_forward_time:.4f}']
]
print(AsciiTable(metric_table).table)



# 生成剪枝后的cfg文件并保存模型
pruned_cfg_name = opt.model_def.replace('/', f'/prune_{percent}_')

#由于原始的compact_module_defs将anchor从字符串变为了数组，因此这里将anchors重新变为字符串

for item in compact_module_defs:
    if item['type']=='yolo':
        item['anchors']='10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326'

pruned_cfg_file = write_cfg(pruned_cfg_name, [model.hyperparams.copy()] + compact_module_defs)
print(f'Config file has been saved: {pruned_cfg_file}')

#compact_model_name = opt.model.replace('/', f'/prune_{percent}_')
compact_model_name = 'weights/yolov3_hand_normal_pruning_'+str(percent)+'percent.weights'

save_weights(compact_model, path=compact_model_name)
print(f'Compact model has been saved: {compact_model_name}')



