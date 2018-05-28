from scipy import io
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns;
from numpy import array
sns.set()
import os

# plt.rcParams["font.family"] = "Arial"

matplotlib.style.use("seaborn-whitegrid")
# plt.figure(dpi=1200)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 12


def plot_cm(s, st, t):
    root_path = "/home/chenchao/JDMC"
    cmp_list = ["Blues", "PuBuGn", "BuPu", "GnBu", "Greens", "Greys", "OrRd", "Oranges", "PuBu", "PuBuGn"]
    cm = []
    data = io.loadmat(os.path.join(root_path, s))
    predict = data.get("predict").T
    real = data.get("real").T
    cm_s = confusion_matrix(y_true=real, y_pred=predict)
    cm.append(cm_s)
    data = io.loadmat(os.path.join(root_path, st))
    predict = data.get("predict").T
    real = data.get("real").T
    cm_st = confusion_matrix(y_true=real, y_pred=predict)
    cm.append(cm_st)
    data = io.loadmat(os.path.join(root_path, t))
    predict = data.get("predict").T
    real = data.get("real").T
    cm_t = confusion_matrix(y_true=real, y_pred=predict)
    cm.append(cm_t)
    cm_str = [r"$ELM_{s}$"+"(Source Only)\n", r"$JDMC$"+"(Source+Target)\n", r"$ELM_{t}$"+"(Target Only)\n"]
    categories = ["backpack", "bike", "calculator", "headphones", "keyboard", "laptop", "monitor", "mouse", "mug",
                  "projector"]
    note = ["(a)", "(b)", "(c)"]
    f, axarr = plt.subplots(1, 3, figsize=(10, 10))
    for i in range(3):
        sns.heatmap(cm[i], ax=axarr[i], cmap=cmp_list[2], cbar=False, xticklabels=categories, yticklabels=categories)
        axarr[i].set_title(cm_str[i])
        axarr[i].set_yticklabels(axarr[i].get_yticklabels(), rotation=0)
        axarr[i].set_xticklabels(axarr[i].get_xticklabels(), rotation=90)
        axarr[i].set_xlabel("Predicted label\n" + note[i])
        axarr[i].set_ylabel("Real label")
        axarr[i].set(adjustable='box-forced', aspect=(1.0 / axarr[i].get_data_ratio() * 1.0))
        # axarr[i].annotate('A')
    # ax = sns.heatmap(cm,cmap="cool")
    f.tight_layout()
    plt.savefig("cm.eps", format="eps", bbox_inches="tight")

    plt.show()
    # f.savefig("cm.eps",format="eps",dpi=1200)


def plot_Text():
    # matplotlib.style.use("seaborn-whitegrid")
    # # plt.figure(dpi=1200)
    # plt.rcParams['font.size'] = 10
    # plt.rcParams['axes.labelsize'] = 12
    # plt.rcParams['axes.labelweight'] = 'bold'
    # plt.rcParams['axes.titlesize'] = 12
    # plt.rcParams['xtick.labelsize'] = 10
    # plt.rcParams['ytick.labelsize'] = 10
    # plt.rcParams['legend.fontsize'] = 10
    # plt.rcParams['figure.titlesize'] = 12

    f, axarr = plt.subplots(1, 4, figsize=(15, 15))
    x = [5, 10, 15, 20]

    # svms=[29.7,28.8,29.5,28.9]
    # svm_e=[1.5,1.3,1.5,1.3]
    #
    # elms=[42.3,39.9,40.0,40.9]
    # elms_e=[1.3,1.6,1.2,1.5]


    svmt = [56.1, 67.0, 70.5, 74.5]
    svmt_e = [1.8, 1.0, 1.0, 0.6]

    # English
    elmt = [60.3, 67.0, 68.8, 72.2]
    elmt_e = [1.2, 1.0, 0.8, 0.5]

    gfk = [56.8, 64.2, 68.8, 71.7]
    gfk_e = [1.2, 0.7, 0.6, 0.5]

    mmdt = [69.2, 71.4, 73.3, 75.2]
    mmdt_e = [0.9, 0.6, 0.7, 0.6]

    cdls = [61.0, 70.2, 73.8, 76.5]
    cdls_e = [1.9, 0.7, 0.6, 0.5]

    ptelm = [69.8, 73.0, 75.7, 77.6]
    ptelm_e = [0.6, 0.4, 0.5, 0.4]

    axarr[0].errorbar(x, svmt, svmt_e, label=r"$SVM_{t}$", linestyle="--",marker="<")
    axarr[0].errorbar(x, elmt, elmt_e, label=r"$ELM_{t}$", linestyle="--",marker="s")
    axarr[0].errorbar(x, gfk, gfk_e, label=r"$GFK$", color='c',marker="d")
    axarr[0].errorbar(x, mmdt, mmdt_e, label=r"$MMDT$",color="black",marker="^")
    axarr[0].errorbar(x, cdls, cdls_e, label=r"$CDLS$", color='b',marker=">")
    axarr[0].errorbar(x, ptelm, ptelm_e, label=r"$JDMC$", color='r', linewidth=2.0,marker="o")
    axarr[0].set(adjustable='box-forced', aspect=(1.0 / axarr[0].get_data_ratio() * 1.0))
    axarr[0].legend(loc="lower right", numpoints=1, fancybox=True)
    axarr[0].set_xlabel("# labeled target domain data per class\n(a)")
    axarr[0].set_ylabel("Accuracy(%)")

    # French
    gfk = [60.5, 66.9, 70.2, 72.3]
    gfk_e = [0.6, 0.6, 0.6, 0.6]

    mmdt = [68.9, 72.8, 73.6, 74.7]
    mmdt_e = [0.8, 0.4, 0.5, 0.4]

    cdls = [64.9, 70.5, 73.7, 75.6]
    cdls_e = [1.5, 0.8, 0.6, 0.6]

    ptelm = [67.8, 73.4, 75.3, 77.1]
    ptelm_e = [0.5, 0.5, 0.5, 0.4]

    axarr[1].errorbar(x, svmt, svmt_e, label=r"$SVM_{t}$", linestyle="--",marker="<")
    axarr[1].errorbar(x, elmt, elmt_e, label=r"$ELM_{t}$", linestyle="--",marker="s")
    axarr[1].errorbar(x, gfk, gfk_e, label=r"$GFK$", color='c',marker="d")
    axarr[1].errorbar(x, mmdt, mmdt_e, label=r"$MMDT$", color="black",marker="^")
    axarr[1].errorbar(x, cdls, cdls_e, label=r"$CDLS$", color='b',marker=">")
    axarr[1].errorbar(x, ptelm, ptelm_e, label=r"$JDMC$", color='r', linewidth=2.0,marker="o")
    axarr[1].set(adjustable='box-forced', aspect=(1.0 / axarr[1].get_data_ratio() * 1.0))
    axarr[1].legend(loc="lower right", numpoints=1, fancybox=True)
    axarr[1].set_xlabel("# labeled target domain data per class\n(b)")
    axarr[1].set_ylabel("Accuracy(%)")

    # German
    gfk = [57.6, 65.2, 68.4, 70.8]
    gfk_e = [1.0, 0.8, 0.6, 0.5]

    mmdt = [68.6, 72.1, 73.8, 75.7]
    mmdt_e = [0.7, 0.6, 0.4, 0.5]

    cdls = [62.1, 70.8, 73.7, 75.9]
    cdls_e = [2.2, 0.8, 0.6, 0.5]

    ptelm = [69.3, 73.5, 75.8, 76.9]
    ptelm_e = [0.8, 0.5, 0.5, 0.3]

    axarr[2].errorbar(x, svmt, svmt_e, label=r"$SVM_{t}$", linestyle="--",marker="<")
    axarr[2].errorbar(x, elmt, elmt_e, label=r"$ELM_{t}$", linestyle="--",marker="s")
    axarr[2].errorbar(x, gfk, gfk_e, label=r"$GFK$", color='c',marker="d")
    axarr[2].errorbar(x, mmdt, mmdt_e, label=r"$MMDT$", color="black",marker="^")
    axarr[2].errorbar(x, cdls, cdls_e, label=r"$CDLS$", color='b',marker=">")
    axarr[2].errorbar(x, ptelm, ptelm_e, label=r"$JDMC$", color='r', linewidth=2.0,marker="o")
    axarr[2].set(adjustable='box-forced', aspect=(1.0 / axarr[2].get_data_ratio() * 1.0))
    axarr[2].legend(loc="lower right", numpoints=1, fancybox=True)
    axarr[2].set_xlabel("# labeled target domain data per class\n(c)")
    axarr[2].set_ylabel("Accuracy(%)")

    # Italian
    gfk = [62.9, 65.7, 69.5, 71.6]
    gfk_e = [0.8, 0.7, 0.6, 0.6]

    mmdt = [70.5, 72.5, 74.5, 76.2]
    mmdt_e = [0.8, 0.6, 0.5, 0.5]

    cdls = [62.3, 71.0, 74.1, 75.9]
    cdls_e = [2.2, 0.9, 0.6, 0.5]

    ptelm = [68.5, 73.8, 75.6, 77.5]
    ptelm_e = [0.7, 0.5, 0.3, 0.4]

    axarr[3].errorbar(x, svmt, svmt_e, label=r"$SVM_{t}$", linestyle="--",marker="<")
    axarr[3].errorbar(x, elmt, elmt_e, label=r"$ELM_{t}$", linestyle="--",marker="s")
    axarr[3].errorbar(x, gfk, gfk_e, label=r"$GFK$", color='c',marker="d")
    axarr[3].errorbar(x, mmdt, mmdt_e, label=r"$MMDT$", color="black",marker="^")
    axarr[3].errorbar(x, cdls, cdls_e, label=r"$CDLS$", color='b',marker=">")
    axarr[3].errorbar(x, ptelm, ptelm_e, label=r"$JDMC$", color='r', linewidth=2.0,marker="o")
    axarr[3].set(adjustable='box-forced', aspect=(1.0 / axarr[3].get_data_ratio() * 1.0))
    axarr[3].legend(loc="lower right", numpoints=1, fancybox=True)
    axarr[3].set_xlabel("# labeled target domain data per class\n(d)")
    axarr[3].set_ylabel("Accuracy(%)")

    # f.subplots_adjust(top=0.983,bottom=0.017,left=0.032,right=0.992,hspace=0.2,wspace=0.147)
    f.tight_layout()
    plt.savefig("text.eps", format="eps", bbox_inches="tight")
    plt.show()



def plot_sensitivity():
    f, axarr = plt.subplots(2, 2, figsize=(6, 6))
    lambda1=[0.001,0.005,0.01,0.05,0.1,0.5,1,10,20]
    lambda1_acc = [74.4, 75.7,76.8, 76.9, 78.6, 74.9, 74.6,67.8,62.1]
    lambda1_std =2* array([1.1, 1.2, 1.2, 1.3, 1.1, 1.1, 1.2,1.0,1.3])

    lambda2 = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 10, 100]
    lambda2_acc = [75.1, 76.2, 76.9, 77.6, 75.9, 75.6, 75.2, 71.8, 64.4]
    lambda2_std =2* array([1.2, 1.3, 0.9, 1.1, 1.2, 1.3, 1.4, 0.9, 1.4])

    lambda3 = [1, 5,10,15, 20, 25, 30, 35, 40]
    lambda3_acc = [33.1, 54.7, 68.6, 73.8, 74.6, 75.6, 75.4, 75.4,74.2]
    lambda3_std =2* array([1.4, 1.4, 1.1, 1.0, 0.9, 0.8, 1.2,1,1.3])

    lambda4 = [1,10,50,100, 200, 500, 1000, 10000]
    lambda4_acc =[40.2,73.6, 75.9, 75.2, 75.7, 74.0, 73.4, 71.6]
    lambda4_std =2* array([1.4, 1.5, 1.3, 1.1, 1.0, 1.2, 1.5, 1.3])

    axarr[0][0].errorbar(lambda1, lambda1_acc, lambda1_std,fmt='g-s')
    axarr[0][1].errorbar(lambda2, lambda2_acc, lambda2_std,fmt='g-o')
    axarr[1][0].errorbar(lambda3, lambda3_acc, lambda3_std,fmt='g-o')
    axarr[1][1].errorbar(lambda4, lambda4_acc, lambda4_std, label=r"$\it{IT}\rightarrow\it{SP}$", fmt='g-o')
    # axarr[0][0].set_xticks(x1)
    # axarr[0][0].set_xticklabels(lambda1)
    axarr[0][0].set_xscale('log')
    axarr[0][1].set_xscale('log')
    axarr[1][1].set_xscale('log')
    axarr[0][0].set_xticks([0.001,0.01,0.1,1,10,100])
    axarr[0][1].set_xticks([0.001,0.01,0.1,1,10,100])
    axarr[1][0].set_xticks([1, 10, 20, 30, 40])
    axarr[1][1].set_xticks([1,10,100,1000,10000])


    lambda1=[0.001,0.005,0.01,0.05,0.1,0.5,1,10,20]
    lambda1_acc = [47.7,49.8,52.0,51.5, 55.2, 54.9, 57.1,53.5,48.1]
    lambda1_std =2* array([1.0,1.2, 0.9, 0.9, 1.0, 1.2, 1.1,1.2,1.3])
    axarr[0][0].errorbar(lambda1, lambda1_acc, lambda1_std,fmt='r-o')
    axarr[0][0].set(aspect=(1.0 / axarr[0][0].get_data_ratio() * 1.0))
    axarr[0][0].set_xlabel("(a) "+r"$\lambda_{1}$")
    axarr[0][0].set_ylabel("Accuracy(%)")

    lambda2 = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 10, 100]
    lambda2_acc = [48.4, 50.1, 52.5, 52.6, 53.2, 54.1, 56.6, 55.7, 53.1]
    lambda2_std =2* array([1.4, 1.0, 1.4, 1.2, 0.8, 0.9, 1.3, 1.2, 1.4])
    axarr[0][1].errorbar(lambda2, lambda2_acc, lambda2_std,fmt='r-o')


    axarr[0][1].set(aspect=(1.0 / axarr[0][1].get_data_ratio() * 1.0))
    axarr[0][1].set_xlabel("(b) "+r"$\lambda_{2}$")
    axarr[0][1].set_ylabel("Accuracy(%)")

    lambda3 = [1, 5, 10, 15, 20, 25, 30, 35, 40]
    lambda3_acc = [32.6, 50.1, 56.7, 55.3, 55.1, 53.8, 53.1, 50.7,47.2]
    lambda3_std =2* array([1.4, 1.2, 1.3, 1.3,1.4, 1.4, 1.2,1.1, 1.4])
    axarr[1][0].errorbar(lambda3, lambda3_acc, lambda3_std,fmt='r-o')

    axarr[1][0].set(aspect=(1.0 / axarr[1][0].get_data_ratio() * 1.0))
    axarr[1][0].set_xlabel("(c) " + r"$\lambda_{3}$")
    axarr[1][0].set_ylabel("Accuracy(%)")

    lambda4 = [1,10,50,100, 200, 500, 1000, 10000]
    lambda4_acc = [55.7,55.7,57.2,58.1,57.1,55.1,56.1,55.2]
    lambda4_std =2* array([1.2,1.3,1.3,1.2,1.3,1.3,1.5,1.4])
    axarr[1][1].errorbar(lambda4, lambda4_acc, lambda4_std, label=r"$\it{amazon}\rightarrow\it{webcam}$", fmt='r-o')
    # axarr[1][1].set_xticks([100,300,500,700,900])
    # axarr[1][1].set_yticks([60,65.0,70.0,75.0])
    # axarr[1][1].set_xticks(node)
    axarr[1][1].set(aspect=(1.0 / axarr[1][1].get_data_ratio() * 1.0))
    # axarr[1][1].legend(loc="lower right",numpoints=1,fancybox=True)
    axarr[1][1].set_xlabel("(d) "+r"$\lambda_{4}$")
    axarr[1][1].set_ylabel("Accuracy(%)")

    plt.figlegend(loc="lower center", fancybox=True, numpoints=1,  bbox_to_anchor=[0.58,-0.02])
    f.tight_layout()
    plt.savefig("sensitivity.eps", format="eps", bbox_inches="tight")
    plt.show()


def plot_convergency():
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(6,3))
    iter=[1,2,3,4,5,6,7,8,9,10]
    Office_Accuracy=[52.3,55.5,56.2,56.7,56.4,56.5,56.3,56.6,55.9,56.4]
    Text_Accuracy=[72.3,74.6,75.0,75.6,75.2,75.5,75.8,76.1,75.9,75.8]
    l1,l2=ax1.plot(iter,Office_Accuracy,'r-o',iter,Text_Accuracy,'g-s')
    ax1.set_ylim(ymin=45)
    fig.legend((l2,l1),(r"$\it{IT}\rightarrow\it{SP}$",r"$\it{amazon}\rightarrow\it{webcam}$"),loc='lower center',bbox_to_anchor=[0.56,-0.02])
    ax1.set_xlabel('#iterations')
    ax1.set_ylabel('Accuracy(%)')
    ax1.set_xticks([2,4,6,8])

    Office_difference=[0.048,0.013,0.007,0.007,0.006,0.006,0.006,0.007,0.006,0.006]
    Text_difference=[0.057,0.039,0.016,0.013,0.013,0.012,0.011,0.012,0.013,0.011]
    l3,l4=ax2.plot(iter,Office_difference,'r-o',iter,Text_difference,'g-s')
    fig.legend((l4,l3),(r"$\it{IT}\rightarrow\it{SP}$",r"$\it{amazon}\rightarrow\it{webcam}$"),loc='lower center',bbox_to_anchor=[0.56,-0.02])
    ax2.set_xlabel('#iterations')
    ax2.set_ylabel(r'$\bf{\Vert\beta^{t+1}-\beta^t\Vert_F^2}$')
    ax2.set_xticks([2,4,6,8])
    fig.tight_layout()
    plt.savefig("convergency.eps", format="eps", bbox_inches="tight")
    plt.show()

if __name__ == '__main__':
    # plot_cm("elms.mat","JDMC.mat","elmt.mat")
    # plot_Text()
    # plot_sensitivity()
    plot_convergency()


























