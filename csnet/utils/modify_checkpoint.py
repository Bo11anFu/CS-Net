import torch
import copy


for i in range(1, 49):
    # 加载原始权重
    checkpoint_path = f'/media/Data1/gr/CSNet/results/MISTNet_IBLoss_MIST_0912_no_select_test/20250424_025353/checkpoints_modified/model_epoch_{str(i)}.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # # 查看所有保存的键
    # print("Checkpoint contains these keys:", checkpoint.keys())

    # 获取原始模型参数
    model_state_dict = checkpoint['model']

    # 查看旧参数字典
    print('修改前的参数名：')
    for key in list(model_state_dict.keys()):
        print(key)

    # 创建新参数字典
    new_model_state_dict = copy.deepcopy(model_state_dict)

    # 批量修改参数名
    for old_key in list(model_state_dict.keys()):
        new_key = old_key
        # ResNet
        new_key = new_key.replace('Encoder', 'encoder')
        new_key = new_key.replace('encoder_3', 'encoder_4')
        new_key = new_key.replace('encoder_2', 'encoder_3')
        new_key = new_key.replace('encoder_1', 'encoder_2')
        new_key = new_key.replace('encoder_0', 'encoder_1')
        # ImplicitMotionCompensation
        new_key = new_key.replace('SSCP', 'IMC')
        new_key = new_key.replace('SSCA_list', 'SNCBs')
        # MSAM
        new_key = new_key.replace('IFF', 'MFA')
        new_key = new_key.replace('inter_frame_fusion_list', 'aggregators')
        # ProgressiveDistillationDecoder
        new_key = new_key.replace('Decoder', 'decoder')
        new_key = new_key.replace('decoder_2', 'decoder_3')
        new_key = new_key.replace('decoder_1', 'decoder_2')
        new_key = new_key.replace('decoder_0', 'decoder_1')
        new_key = new_key.replace('bottleneck', 'MFB')
        new_key = new_key.replace('activation', 'modulator')
        # 更新字典
        if new_key != old_key:
            new_model_state_dict[new_key] = new_model_state_dict.pop(old_key)

    # 验证新参数名
    print('\n修改后的参数名：')
    for key in list(new_model_state_dict.keys()):
        print(key)

    # 保存修改后的权重
    new_checkpoint = {'model': new_model_state_dict}
    new_checkpoint_path = checkpoint_path.replace(".pth", "_modified.pth")
    torch.save(new_checkpoint, new_checkpoint_path)
    print(f"修改后的权重（仅参数）已保存至：{new_checkpoint_path}")

    # # 查看优化器状态
    # optimizer_state_dict = checkpoint['optimizer']
    # print("\nOptimizer state:")
    # print(optimizer_state_dict.keys())  # 显示优化器状态的键
    #
    # # 查看学习率调度器状态
    # lr_scheduler_state_dict = checkpoint['lr_scheduler']
    # print("\nLR Scheduler state:")
    # print(lr_scheduler_state_dict.keys())  # 显示调度器状态的键
    #
    # # 查看其他信息
    # print("\nEpoch:", checkpoint['epoch'])
    # print("\nConfig:", checkpoint['config'])
