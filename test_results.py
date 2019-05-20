import numpy as np
import matplotlib.pyplot as plt
import pickle


if __name__ == '__main__':
    # Plotting success rate
    """with open('./results/all_fictive.pkl', 'rb') as f:
        ddpg_all_fictive = pickle.load(f)
    with open('./results/1_fictive.pkl', 'rb') as f:
        ddpg_1_fictive = pickle.load(f)
    with open('./results/ddpg_gcp.pkl', 'rb') as f:
        ddpg_gcp = pickle.load(f)
    with open('./results/ddpg_only.pkl', 'rb') as f:
        ddpg_only = pickle.load(f)
    with open('./results/maml_gcp.pkl', 'rb') as f:
        maml_gcp = pickle.load(f)
    with open('./results/maml_no_gcp.pkl', 'rb') as f:
        maml_no_gcp = pickle.load(f)

    x = np.arange(0, len(maml_gcp), 1)
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(x, maml_gcp, '-r')
    ax.plot(x, maml_no_gcp, '-k')
    ax.plot(x, ddpg_1_fictive, '-g')
    ax.plot(x, ddpg_all_fictive, '-b')
    ax.plot(x, ddpg_gcp, '-y')
    ax.plot(x, ddpg_only, '-m')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height*0.9])
    plt.title('Training success rate over 500 epoch , 2D Hypercube')
    plt.ylabel('Success rate')
    plt.xlabel('Epochs')
    plt.grid()
    plt.legend(['maml_gcp', 'maml_no_gcp', 'ddpg_1_fictive', 'ddpg_all_fictive', 'ddpg_gcp', 'ddpg_only'],
               fancybox=True, shadow=True, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)
    plt.show()"""
    # Plot steps needed to converge
    """with open('./results/all_fictive_time.pkl', 'rb') as f:
        ddpg_all_fictive = pickle.load(f)
    with open('./results/1_fictive_time.pkl', 'rb') as f:
        ddpg_1_fictive = pickle.load(f)
    with open('./results/ddpg_gcp_time.pkl', 'rb') as f:
        ddpg_gcp = pickle.load(f)
    with open('./results/ddpg_only_time.pkl', 'rb') as f:
        ddpg_only = pickle.load(f)
    with open('./results/maml_gcp_time.pkl', 'rb') as f:
        maml_gcp = pickle.load(f)
    with open('./results/maml_no_gcp_time.pkl', 'rb') as f:
        maml_no_gcp = pickle.load(f)
    means_maml_gcp = np.array([np.mean(a) for a in maml_gcp])
    means_maml_no_gcp = np.array([np.mean(a) for a in maml_no_gcp])
    means_ddpg_only = np.array([np.mean(a) for a in ddpg_only])
    means_ddpg_gcp = np.array([np.mean(a) for a in ddpg_gcp])
    means_ddpg_1_fictive = np.array([np.mean(a) for a in ddpg_1_fictive])
    means_ddpg_all_fictive = np.array([np.mean(a) for a in ddpg_all_fictive])

    std_maml_gcp = np.array([np.std(a) for a in maml_gcp])
    std_maml_no_gcp = np.array([np.std(a) for a in maml_no_gcp])
    std_ddpg_only = np.array([np.std(a) for a in ddpg_only])
    std_ddpg_gcp = np.array([np.std(a) for a in ddpg_gcp])
    std_ddpg_1_fictive = np.array([np.std(a) for a in ddpg_1_fictive])
    std_ddpg_all_fictive = np.array([np.std(a) for a in ddpg_all_fictive])
    x = np.arange(0, len(means_maml_gcp), 1)
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(x, means_maml_gcp, '-r')
    ax.fill_between(x, means_maml_gcp + std_maml_gcp, means_maml_gcp - std_maml_gcp, color='red', alpha=0.2)
    ax.plot(x, means_maml_no_gcp, '-k')
    ax.fill_between(x, means_maml_no_gcp + std_maml_no_gcp, means_maml_no_gcp - std_maml_no_gcp, color='black', alpha=0.2)
    ax.plot(x, means_ddpg_1_fictive, '-g')
    ax.fill_between(x, means_ddpg_1_fictive + std_ddpg_1_fictive, means_ddpg_1_fictive - std_ddpg_1_fictive, color='green', alpha=0.2)
    ax.plot(x, means_ddpg_all_fictive, '-b')
    ax.fill_between(x, means_ddpg_all_fictive + std_ddpg_all_fictive, means_ddpg_all_fictive - std_ddpg_all_fictive, color='blue', alpha=0.2)
    ax.plot(x, means_ddpg_gcp, '-y')
    ax.fill_between(x, means_ddpg_gcp + std_ddpg_gcp, means_ddpg_gcp - std_ddpg_gcp, color='yellow', alpha=0.2)
    ax.plot(x, means_ddpg_only, '-m')
    ax.fill_between(x, means_ddpg_only + std_ddpg_only, means_ddpg_only - std_ddpg_only, color='magenta', alpha=0.2)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height*0.9])
    plt.title('Number of steps needed to reach goal in training , 2D Hypercube')
    plt.ylabel('Success rate')
    plt.xlabel('Epochs')
    plt.ylim(0, 250)
    plt.grid()
    # plt.legend(['maml_gcp', 'maml_no_gcp', 'ddpg_1_fictive', 'ddpg_all_fictive', 'ddpg_gcp', 'ddpg_only'],
    #           fancybox=True, shadow=True, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)
    plt.legend(['maml+gcp'])
    plt.show()"""
    # Plots for maml gradients
    with open('./results/0g_maml_gcp_time_D.pkl', 'rb') as f:
        maml_gcp_0g = pickle.load(f)
    with open('./results/1g_maml_gcp_time_D.pkl', 'rb') as f:
        maml_gcp_1g = pickle.load(f)
    with open('./results/2g_maml_gcp_time_D.pkl', 'rb') as f:
        maml_gcp_2g = pickle.load(f)
    with open('./results/3g_maml_gcp_time_D.pkl', 'rb') as f:
        maml_gcp_3g = pickle.load(f)

    means_maml_gcp_0g = np.mean(maml_gcp_0g)
    means_maml_gcp_1g = np.mean(maml_gcp_1g)
    means_maml_gcp_2g = np.mean(maml_gcp_2g)
    means_maml_gcp_3g = np.mean(maml_gcp_3g)

    std_maml_gcp_0g = np.std(maml_gcp_0g)
    std_maml_gcp_1g = np.std(maml_gcp_1g)
    std_maml_gcp_2g = np.std(maml_gcp_2g)
    std_maml_gcp_3g = np.std(maml_gcp_3g)
    stds = np.array([std_maml_gcp_0g, std_maml_gcp_1g, std_maml_gcp_2g, std_maml_gcp_3g])
    means = np.array([means_maml_gcp_0g, means_maml_gcp_1g, means_maml_gcp_2g, means_maml_gcp_3g])
    x = np.arange(0, 4, 1)
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(x, means, '-m')
    ax.fill_between(x, means + stds, means - stds, color='magenta', alpha=0.2)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    plt.title('Average steps needed to reach goal in testing , 2D Hypercube')
    plt.ylabel('Average steps needed to reach goal')
    plt.xlabel('Gradient steps')
    plt.ylim(0, 250)
    plt.grid()
    # plt.legend(['maml_gcp', 'maml_no_gcp', 'ddpg_1_fictive', 'ddpg_all_fictive', 'ddpg_gcp', 'ddpg_only'],
    #           fancybox=True, shadow=True, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)
    plt.legend(['maml+gcp'])
    plt.show()

