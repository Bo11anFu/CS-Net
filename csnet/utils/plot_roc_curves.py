import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

# 全局样式配置
rc('font', family='Times New Roman')


def plot_roc_curves(fpr_list, tpr_list, method_names, colors, output_filename):
    # 样式参数配置
    MAIN_FONT_SIZE = 22.05
    AXIS_LINEWIDTH = 1.83
    LINE_WIDTH = 2.5  # 曲线线宽
    FIG_SIZE = (12, 8)
    DASHED_LINE_STYLE = '--'
    DIAGONAL_COLOR = '#666666'

    # 初始化画布
    plt.figure(figsize=FIG_SIZE)
    ax = plt.gca()

    # 绘制对角线
    ax.plot([0, 1], [0, 1], DASHED_LINE_STYLE,
            color=DIAGONAL_COLOR, linewidth=LINE_WIDTH * 0.8)

    # 绘制各方法ROC曲线
    for fpr, tpr, name, color in zip(fpr_list, tpr_list, method_names, colors):
        ax.plot(fpr, tpr, color=color, label=name, linewidth=LINE_WIDTH)

    # 坐标轴设置
    ax.set_xlabel('False positive rate', fontsize=MAIN_FONT_SIZE)
    ax.set_ylabel('True positive rate', fontsize=MAIN_FONT_SIZE)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])

    # 刻度参数设置
    ax.tick_params(axis='both', which='both', direction='in',
                   labelsize=MAIN_FONT_SIZE, width=AXIS_LINEWIDTH)

    # 坐标轴线宽设置
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_LINEWIDTH)

    # 图例设置
    legend = ax.legend(loc='lower right', fontsize=MAIN_FONT_SIZE)
    legend.get_frame().set_linewidth(AXIS_LINEWIDTH)

    # 保存图像
    plt.savefig(output_filename, transparent=True, bbox_inches='tight')
    plt.close()


# 示例用法
if __name__ == '__main__':
    # 示例数据（需要替换为真实数据）
    method_names = ['DSAT', 'SIATD', 'NUDT-MIRSDT', 'TSIRMT', 'MIST (ours)']
    colors = ['#768DD1', '#91CC75', '#FAC858', '#FDAA86', '#EE6666']


    # 生成示例数据（请替换为真实数据）
    def generate_sample_curve():
        x = np.linspace(0, 1, 100)
        y = np.sqrt(x)  # 示例曲线
        return x, y


    fpr_list = [generate_sample_curve()[0] for _ in method_names]
    tpr_list = [generate_sample_curve()[1] for _ in method_names]

    plot_roc_curves(fpr_list, tpr_list, method_names, colors, 'roc_curves.svg')