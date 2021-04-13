from models import *
from utils.utils import *
import torch
import numpy as np
from copy import deepcopy
from test import evaluate
from terminaltables import AsciiTable
import time
from utils.prune_utils import *

class opt():
    model_def = "config/yolov3-hand.cfg"
    data_config = "config/oxfordhand.data"
    model = 'checkpoints/yolov3_ckpt.pth'


#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Darknet(opt.model_def).to(device)
model.load_state_dict(torch.load(opt.model))

data_config = parse_data_config(opt.data_config)
valid_path = data_config["valid"]
class_names = load_classes(data_config["names"])

eval_model = lambda model:evaluate(model, path=valid_path, iou_thres=0.5, conf_thres=0.01,
    nms_thres=0.5, img_size=model.img_size, batch_size=8)
obtain_num_parameters = lambda model:sum([param.nelement() for param in model.parameters()])

origin_model_metric = eval_model(model)
origin_nparameters = obtain_num_parameters(model)
#前面都是加载模型和验证等部分
#剪枝从这里开始
#还是先获取相应的层次索引，实现参考utils/prune_utils.py
CBL_idx, Conv_idx, prune_idx= parse_module_defs(model.module_defs)
#获取具体的Bn层权重，具体实现参考utils/prune_utils.py
#最终返回的是对应权重的绝对值，因为剪枝的是尺度因子接近0的通道。索引返回scales的绝对值即可
bn_weights = gather_bn_weights(model.module_list, prune_idx)
#对Bn中scales的绝对值进行排序，升序排序
sorted_bn = torch.sort(bn_weights)[0]

# 避免剪掉所有channel的最高阈值(每个BN层的gamma的最大值或最小值的绝对值即为阈值上限)
#抽取每一层剪枝的阈值，防止剪掉某一层的全部通道
highest_thre = []
for idx in prune_idx:
    #取的是某一层scales绝对值最大的值
    highest_thre.append(model.module_list[idx][1].weight.data.abs().max().item())
#获取其中最小值，也就是最终的阈值实际上应该是所有层次中scales的绝对值的最小值
highest_thre = min(highest_thre)

# 找到highest_thre对应的下标对应的百分比，因为是升序排序，所以实际剪枝比例不能不这个大
percent_limit = (sorted_bn==highest_thre).nonzero().item()/len(bn_weights)

print(f'Threshold should be less than {highest_thre:.4f}.')
print(f'The corresponding prune ratio is {percent_limit:.3f}.')

#%%
def prune_and_eval(model, sorted_bn, percent=.0):
    model_copy = deepcopy(model)
    #这里实际上就是根据指定的比例来计算具体的阈值
    thre_index = int(len(sorted_bn) * percent)
    #获得具体的BN剪枝阈值，绝对值小于这个值得BN层scales会被置0
    thre = sorted_bn[thre_index]

    print(f'Channels with Gamma value less than {thre:.4f} are pruned!')

    remain_num = 0
    #循环处理剪枝的层次
    for idx in prune_idx:

        bn_module = model_copy.module_list[idx][1]
        #通过mask进行剪枝，具体实现参考utils/prune_utils.py
        #根据阈值获取当前BN层剪枝的mask矩阵,得到一个值不是0就是1的矩阵
        mask = obtain_bn_mask(bn_module, thre)
        #统计mask中1的个数，也就是这一层留下的通道数
        remain_num += int(mask.sum())
        #Bn层scales跟mask相乘，将scales绝对值小于阈值的置0
        bn_module.weight.data.mul_(mask)

    mAP = eval_model(model_copy)[2].mean()

    print(f'Number of channels has been reduced from {len(sorted_bn)} to {remain_num}')
    print(f'Prune ratio: {1-remain_num/len(sorted_bn):.3f}')
    print(f'mAP of the pruned model is {mAP:.4f}')

    return thre
#指定剪枝阈值
percent = 0.85
#利用阈值和mask完成剪枝并验证map
threshold = prune_and_eval(model, sorted_bn, percent)
#%%
#获取对应的权重的msak
def obtain_filters_mask(model, thre, CBL_idx, prune_idx):

    pruned = 0
    total = 0
    num_filters = []
    filters_mask = []
    #循环处理每一个CBL层，虽然不是所有的CBL层都参与剪枝，但是对某一层剪枝之后，会影响前后层次的通道，
    #每一个CBL层都要处理
    for idx in CBL_idx:
        bn_module = model.module_list[idx][1]
        #判断当前CBL中的BN层是否参与剪枝
        if idx in prune_idx:
            #获取对应的BN层剪枝mask
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
            #如果这个BN不剪枝，mask矩阵全1
            mask = np.ones(bn_module.weight.data.shape)
            remain = mask.shape[0]
        #total统计全部的通道数
        total += mask.shape[0]
        #得到每一层输出的保留通道数
        num_filters.append(remain)
        #同时获取每一层BN对应的mask矩阵
        filters_mask.append(mask.copy())
    #计算整个网络的剪枝率
    prune_ratio = pruned / total
    print(f'Prune channels: {pruned}\tPrune ratio: {prune_ratio:.3f}')

    return num_filters, filters_mask
#对BN层相应scales置0之后，等于去掉了相应输入和输出的某一个通道，为减少模型体积，这时候就可以删除对应的相关权重了
#num_filters是一个列表，记录个CBL层最终输出的通道数，filters_mask也是一个列表，存储每一个CBL层中的mask矩阵
num_filters, filters_mask = obtain_filters_mask(model, threshold, CBL_idx, prune_idx)

#%%
#将CBL的索引和存储每一个CBL的mask的filters_mask对应起来，组成一个字典
CBLidx2mask = {idx: mask for idx, mask in zip(CBL_idx, filters_mask)}
#对模型进行剪枝，具体实现参考utils/prune_utils.py
pruned_model = prune_model_keep_size(model, prune_idx, CBL_idx, CBLidx2mask)

eval_model(pruned_model)


#%%
#复制一份剪枝前的模型结构，也就是剪枝之前模型的cfg的解析结果
compact_module_defs = deepcopy(model.module_defs)
for idx, num in zip(CBL_idx, num_filters):
    assert compact_module_defs[idx]['type'] == 'convolutional'
    #调整相应层次的的卷积核个数，或者说输出通道数，因为剪枝会删除部分通道，num_filters实际上就是剪枝之后每层的输出通道
    #也就是这一层的卷积核个数
    compact_module_defs[idx]['filters'] = str(num)

#%%
#根据同样的超参数和剪枝后的新cfg结构创建一个新模型
compact_model = Darknet([model.hyperparams.copy()] + compact_module_defs).to(device)
#统计当期模型参数量
compact_nparameters = obtain_num_parameters(compact_model)
#初始化这个新模型。用见之后的权重数据或者说前面调整过的参数
#具体实现参考utils/prune_utils.py，这一步结束，就得到了一个完整的剪枝之后的模型了，接下来只用保存和生成相应的cfg就可以了
init_weights_from_loose_model(compact_model, pruned_model, CBL_idx, Conv_idx, CBLidx2mask)

#%%
random_input = torch.rand((1, 3, model.img_size, model.img_size)).to(device)

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

diff = (pruned_output-compact_output).abs().gt(0.001).sum().item()
if diff > 0:
    print('Something wrong with the pruned model!')

#%%
# 在测试集上测试剪枝后的模型, 并统计模型的参数数量
compact_model_metric = eval_model(compact_model)

#%%
# 比较剪枝前后参数数量的变化、指标性能的变化
metric_table = [
    ["Metric", "Before", "After"],
    ["mAP", f'{origin_model_metric[2].mean():.6f}', f'{compact_model_metric[2].mean():.6f}'],
    ["Parameters", f"{origin_nparameters}", f"{compact_nparameters}"],
    ["Inference", f'{pruned_forward_time:.4f}', f'{compact_forward_time:.4f}']
]
print(AsciiTable(metric_table).table)

#%%
# 生成剪枝后的cfg文件并保存模型
pruned_cfg_name = opt.model_def.replace('/', f'/prune_{percent}_')
pruned_cfg_file = write_cfg(pruned_cfg_name, [model.hyperparams.copy()] + compact_module_defs)
print(f'Config file has been saved: {pruned_cfg_file}')

compact_model_name = opt.model.replace('/', f'/prune_{percent}_')
torch.save(compact_model.state_dict(), compact_model_name)
print(f'Compact model has been saved: {compact_model_name}')
