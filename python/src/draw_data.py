import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csvs_to_read = [['resources/neural_network/evaluation_models_dumps/a_e20_n100_i100_lr1e-06_bs16_os1_s0_dump.csv',
                 'resources/neural_network/evaluation_models_dumps/a_e20_n1000_i100_lr1e-06_bs16_os1_s0_dump.csv',
                 'resources/neural_network/evaluation_models_dumps/a_e20_n5000_i100_lr1e-06_bs16_os1_s0_dump.csv',
                 'resources/neural_network/evaluation_models_dumps/a_e20_n10000_i100_lr1e-06_bs16_os1_s0_dump.csv'],

                ['resources/neural_network/evaluation_models_dumps/a_e20_n10000_i100_lr1e-06_bs16_os1_s0_dump.csv',
                 'resources/neural_network/evaluation_models_dumps/a_e20_n10000_i100_lr2e-06_bs16_os1_s0_dump.csv',
                 'resources/neural_network/evaluation_models_dumps/a_e20_n10000_i100_lr5e-06_bs16_os1_s0_dump.csv',
                 'resources/neural_network/evaluation_models_dumps/a_e20_n10000_i100_lr1e-05_bs16_os1_s0_dump.csv'],

                ['resources/neural_network/evaluation_models_dumps/train_e20_10000_1e6_teacc801.csv',
                 'resources/neural_network/evaluation_models_dumps/a_e20_n10000_i100_lr1e-06_bs8_os1_s0_dump.csv',
                 'resources/neural_network/evaluation_models_dumps/a_e20_n10000_i100_lr1e-06_bs16_os1_s0_dump.csv',
                 'resources/neural_network/evaluation_models_dumps/a_e20_n10000_i100_lr1e-06_bs32_os1_s0_dump.csv'],

                ['resources/neural_network/evaluation_models_dumps/a_e20_n10000_i100_lr1e-06_bs16_os0_s0_dump.csv',
                 'resources/neural_network/evaluation_models_dumps/a_e20_n10000_i100_lr1e-06_bs16_os0_s1_dump.csv',
                 'resources/neural_network/evaluation_models_dumps/a_e20_n10000_i100_lr1e-06_bs16_os1_s0_dump.csv',
                 'resources/neural_network/evaluation_models_dumps/a_e20_n10000_i100_lr1e-06_bs16_os1_s1_dump.csv'],
                ]

titles = ["Model's Performance on Varying Number of Input Sequences n",
          "Model's Performance on Varying Number of Learning Rates lr",
          "Model's Performance on Varying Number of Batch Sizes bs",
          "Model's Performance on Varying Oversampling and Shuffling Settings"]

labels = [['n=100', 'n=1000', 'n=5000', 'n=10000'],
          ['lr=1e-6', 'lr=2e-6', 'lr=5e-6', 'lr=1e-5'],
          ['bs=2', 'bs=8', 'bs=16', 'bs=32'],
          ['sh=0, os=0', 'sh=0, os=1', 'sh=1, os=0', 'sh=1, os=1']]

for i in range(4):

    data1 = pd.read_csv(csvs_to_read[i][0])
    data2 = pd.read_csv(csvs_to_read[i][1])
    data3 = pd.read_csv(csvs_to_read[i][2])
    data4 = pd.read_csv(csvs_to_read[i][3])

    train_accuracies = [data1['train_accuracy'], data2['train_accuracy'], data3['train_accuracy'], data4['train_accuracy']]
    validation_accuracies = [data1['validation_accuracy'], data2['validation_accuracy'], data3['validation_accuracy'], data4['validation_accuracy']]
    train_losses = [data1['train_loss'], data2['train_loss'], data3['train_loss'], data4['train_loss']]
    validation_losses = [data1['validation_loss'], data2['validation_loss'], data3['validation_loss'], data4['validation_loss']]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 9))
    x = range(20)
    fig.suptitle(titles[i])

    # Subplot 1
    ax1.grid(linestyle='--', linewidth=0.3)
    ax1.set_xticks(x[::2])
    ax1.set_yticks(np.arange(0, 1.1, step=0.1))
    ax1.set_ylim(0, 1.0)

    ax1.plot(x, train_accuracies[0], label=labels[i][0])
    ax1.plot(x, train_accuracies[1], label=labels[i][1])
    ax1.plot(x, train_accuracies[2], label=labels[i][2])
    ax1.plot(x, train_accuracies[3], label=labels[i][3])
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Train Accuracy')
    # ax1.set_title('Train Accuracy')
    ax1.legend()

    # Subplot 2
    ax2.grid(linestyle='--', linewidth=0.3)
    ax2.set_xticks(x[::2])
    ax2.set_yticks(np.arange(0, 1.1, step=0.1))
    ax2.set_ylim(0, 1.0)

    ax2.plot(x, validation_accuracies[0], label=labels[i][0])
    ax2.plot(x, validation_accuracies[1], label=labels[i][1])
    ax2.plot(x, validation_accuracies[2], label=labels[i][2])
    ax2.plot(x, validation_accuracies[3], label=labels[i][3])
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Validation Accuracy')
    # ax2.set_title('Validation Accuracy')
    ax2.legend()

    # Subplot 3
    ax3.grid(linestyle='--', linewidth=0.3)
    ax3.set_xticks(x[::2])
    if i != 2:
        ax3.set_yticks(np.arange(0.00, 0.18, step=0.02))
        ax3.set_ylim(0, 0.18)
    else:
        ax3.set_yticks(np.arange(0.00, 1.3, step=0.2))
        ax3.set_ylim(0, 1.3)

    ax3.plot(x, train_losses[0], label=labels[i][0])
    ax3.plot(x, train_losses[1], label=labels[i][1])
    ax3.plot(x, train_losses[2], label=labels[i][2])
    ax3.plot(x, train_losses[3], label=labels[i][3])
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Train Loss')
    # ax3.set_title('Train Loss')
    ax3.legend()

    # Subplot 3
    ax4.grid(linestyle='--', linewidth=0.3)
    ax4.set_xticks(x[::2])
    if i != 2:
        ax4.set_yticks(np.arange(0.00, 0.18, step=0.02))
        ax4.set_ylim(0, 0.18)
    else:
        ax4.set_yticks(np.arange(0.00, 1.3, step=0.2))
        ax4.set_ylim(0, 1.3)

    ax4.plot(x, validation_losses[0], label=labels[i][0])
    ax4.plot(x, validation_losses[1], label=labels[i][1])
    ax4.plot(x, validation_losses[2], label=labels[i][2])
    ax4.plot(x, validation_losses[3], label=labels[i][3])
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('Validation Loss')
    # ax4.set_title('Validation Loss')
    ax4.legend()

    plt.tight_layout()
    plt.show()
