from copy import deepcopy
from csnet.losses.loss_basic import SoftIoULoss, MultiSoftIoULoss, DiceLoss, BceLoss, L1Loss
from csnet.losses.loss_sufficiency import SufficiencyLoss
from csnet.losses.loss_edge import EdgeLoss
from csnet.losses.loss_sls_iou_sdm import SLSIoULoss, SDMLoss



#loss_cfg={
#             'num_preds': 1,
#             'loss_1': {
#                 'type': 'SoftIoULoss',
#                 'weight': 1,
#                 'smooth': 1,
#                 'reduction': 'mean'
#             },
#             'loss_2': {
#                 'type': 'SufficiencyLoss',
#                 'weight': 0.01,
#                 'temperature': 1,
#                 'reduction': 'mean'
#             }
def build_loss(loss_cfg):
    loss_cfg = deepcopy(loss_cfg)
    num_preds = loss_cfg.pop('num_preds')
    loss_fn = {}
    loss_weight = {}
    train_loss = {}#存储损失权重
    train_iter_loss = {}
    test_loss = {}
    use_sufficiency_loss = False
    use_edge_loss = False
    for loss in loss_cfg:
        #loss_fn{'SoftIoULoss' : , 'SufficiencyLoss' :  }
        loss_fn[loss_cfg[loss]['type']] = choose_loss(loss_cfg[loss])
        #根据配置文件创建损失函数并设置权重
        loss_weight[loss_cfg[loss]['type']] = loss_cfg[loss]['weight']
        if loss_cfg[loss]['type'] == 'SufficiencyLoss':
            use_sufficiency_loss = True
            train_loss[loss_cfg[loss]['type']] = 0.
            train_iter_loss[loss_cfg[loss]['type']] = 0.
            test_loss[loss_cfg[loss]['type']] = 0.
        elif loss_cfg[loss]['type'] == 'EdgeLoss':
            use_edge_loss = True
            train_loss[loss_cfg[loss]['type']] = 0.
            train_iter_loss[loss_cfg[loss]['type']] = 0.
            test_loss[loss_cfg[loss]['type']] = 0.
        else:
            #为每个预测分配权重
            if not isinstance(loss_weight[loss_cfg[loss]['type']], (tuple, list)):
                loss_weight[loss_cfg[loss]['type']] = [loss_weight[loss_cfg[loss]['type']]] * num_preds
            assert len(loss_weight[loss_cfg[loss]['type']]) == num_preds
            for idx in range(num_preds):
                train_loss[loss_cfg[loss]['type'] + '_' + str(idx)] = 0.
                train_iter_loss[loss_cfg[loss]['type'] + '_' + str(idx)] = 0.
                test_loss[loss_cfg[loss]['type'] + '_' + str(idx)] = 0.
    assert len(loss_fn) > 0
    return loss_fn, loss_weight, train_loss, train_iter_loss, test_loss, use_sufficiency_loss, use_edge_loss


def choose_loss(sub_loss_cfg):
    sub_loss_cfg = deepcopy(sub_loss_cfg)
    loss_type = sub_loss_cfg.pop('type')
    sub_loss_cfg.pop('weight')
    if loss_type == 'SoftIoULoss':
        criterion = SoftIoULoss(**sub_loss_cfg)
    elif loss_type == 'MultiSoftIoULoss':
        criterion = MultiSoftIoULoss(**sub_loss_cfg)
    elif loss_type == 'DiceLoss':
        criterion = DiceLoss(**sub_loss_cfg)
    elif loss_type == 'BceLoss':
        criterion = BceLoss(**sub_loss_cfg)
    elif loss_type == 'L1Loss':
        criterion = L1Loss(**sub_loss_cfg)
    elif loss_type == 'SufficiencyLoss':
        criterion = SufficiencyLoss(**sub_loss_cfg)
    elif loss_type == 'EdgeLoss':
        criterion = EdgeLoss(**sub_loss_cfg)
    elif loss_type == 'SLSIoULoss':
        criterion = SLSIoULoss(**sub_loss_cfg)
    elif loss_type == 'SDMLoss':
        criterion = SDMLoss(**sub_loss_cfg)
    else:
        raise NotImplementedError(
            f"Invalid loss type '{loss_type}'. Only SoftIoULoss, DiceLoss, BceLoss, L1Loss, SufficiencyLoss, EdgeLoss, SLSIoULoss, and SDMLoss are supported.")
    return criterion
