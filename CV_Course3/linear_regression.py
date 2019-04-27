import numpy as np
import matplotlib.pyplot as plt
import time

"""
更python的way，我的理解是直接用numpy数组进行运算，而不采用for循环操作每一个点
所以对原始代码的改动也都是这方面的
另外加了一个计算RMSE和R^2的函数来查看拟合精度
最后，可能将所有代码写成一个类更便于操作吧，还没这么做。
"""


def inference(w, b, x_list):
    """
    定义hypothesis:
    y = w * x + b
    """
    pred_y_list = w * x_list + b
    return pred_y_list


def eval_cost(x_list, gt_y_list, w, b):
    """
    定义cost function (quadratic cost here)
    Cost = 1/2m * sum((pred_y - y_t) ** 2)
    """
    # avg_cost = 0
    # for i in range(len(x_list)):
    #     avg_cost += 0.5 * (inference(w, b, x_list[i]) - gt_y_list[i]) ** 2
    cost_list = 0.5 * (inference(w, b, x_list) - gt_y_list) ** 2
    avg_cost = cost_list.mean()
    return avg_cost


def cal_gradient(pred_y_list, gt_y_list, x_list):
    """
    计算梯度
    dJ/dw  --> dw (偏导)
    dJ/db  --> db
    """
    # diff = pred_y - gt_y
    # dw = diff * x
    # db = diff
    diff_list = pred_y_list - gt_y_list
    dw_list = diff_list * x_list
    db_list = diff_list
    avg_dw = dw_list.mean()
    avg_db = db_list.mean()
    return avg_dw, avg_db


def gradient_descent(x_list, gt_y_list, w, b, lr):
    """
    梯度下降:
    for parameter θ： θ = θ + △θ
    where △θ = -η·▽C
    ▽C is the so called GRADIENT
    根据梯度下降，w，b的更新方式为：
    w = w - lr * avg_dw
    b = b - lr * avg_db
    dw 为cost function对w的偏导，即w方向的梯度, 这是通过一个样本求得的
    而实际有很多样本，所以梯度为所有样本梯度的算术平均 [数值计算]
    """
    # # avg_dw, avg_db = 0, 0
    # tot_dw, tot_db = 0, 0
    # for i in range(len(x_list)):
    #     pred_y = inference(w, b, x_list[i])
    #     dw, db = cal_gradient(pred_y, gt_y_list[i], x_list[i])
    #     tot_dw += dw
    #     tot_db += db
    pred_y_list = np.array([inference(w, b, x_list[i]) for i in range(len(x_list))])
    avg_dw, avg_db = cal_gradient(pred_y_list, gt_y_list, x_list)

    # avg_dw = tot_dw / len(x_list)
    # avg_db = tot_db / len(x_list)
    w -= lr * avg_dw
    b -= lr * avg_db
    return w, b


def train(x_list, gt_y_list, batch_size, lr, max_iter):
    """
    在数据集上开始训练：
    训练过程即不断进行梯度下降的过程，
    这儿的训练结束条件为达到最大迭代次数max_iter
    """
    w, b = 0, 0
    x_list = x_list
    gt_y_list = gt_y_list
    # for i in range(max_iter):
    #     w, b = gradient_descent(x_list, gt_y_list, w, b, lr)
    #     if not i % 50:  # print current w, b and loss every 50 iterations
    #         print("w: {:.4f} | b: {:.4f}".format(w, b))
    #         print("current loss: {}".format(eval_cost(x_list, gt_y_list, w, b)))
    for i in range(max_iter):
        batch_idxs = np.random.choice(len(x_list), batch_size, replace=False)
        batch_x = [x_list[j] for j in batch_idxs]
        batch_y = [gt_y_list[k] for k in batch_idxs]
        w, b = gradient_descent(batch_x, batch_y, w, b, lr)
        if not i % 100:
            print("w: {:.4f} | b: {:.4f}".format(w, b))
            print("current loss: {}".format(eval_cost(x_list, gt_y_list, w, b)))
    return w, b


def generate_sample_data():
    """
    模拟产生数据集用于训练:
    先产生w, b
    再产生x_list
    再根据w, b和x_list产生y_list
    """
    # w = random.randint(0, 10) + random.random()
    # b = random.randint(0, 5) + random.random()
    w = np.random.randint(0, 10) + np.random.random()
    b = np.random.randint(0, 5) + np.random.random()

    num_samples = 100

    # x_list = []
    # y_list = []
    # for i in range(num_samples):
    #     x = random.randint(0, 10) * random.random()
    #     y = w * x + b + random.random() * random.randint(-1, 1)
    #     x_list.append(x)
    #     y_list.append(y)
    x_list = np.random.uniform(0, 100, num_samples)
    y_list = w * x_list + b + np.random.uniform(-30, 30, num_samples)
    y_list_line = w * x_list + b

    return x_list, y_list, w, b, y_list_line


def cal_accuracy(pred_y_list, y_truth_list):
    mse = np.sum((pred_y_list - y_truth_list) ** 2)
    rmse = np.sqrt(mse.mean())  # Root Mean Squared Error
    ssr = np.sum((pred_y_list - y_truth_list) ** 2)
    sst = np.sum((y_truth_list - y_truth_list.mean()) ** 2)
    r2_score = 1 - (ssr/sst)  # Coefficient of Determination
    return rmse, r2_score


def run():
    x_list, y_list, w, b, y_list_line = generate_sample_data()
    fig = plt.figure(figsize=(12, 8))
    ax1 = plt.subplot(1, 1, 1)
    ax1.scatter(x_list, y_list)

    lr = 0.000001
    max_iter = 10000
    batch_size = 50
    cw, cb = train(y_list, x_list, batch_size, lr, max_iter)

    pred_y_list = inference(cw, cb, x_list)
    y_truth_list = y_list
    rmse, r2_score = cal_accuracy(pred_y_list, y_truth_list)
    print("true_w = {:.4f} | true_b = {:.4f}".format(w, b))
    print("rmse = {:.4f} | r2_score = {:.4f}".format(rmse, r2_score))

    ax1.plot(x_list, y_list_line, color='r')
    ax1.plot(x_list, cw * x_list + cb)
    plt.savefig("test.jpg", format='jpg', dpi=300)
    plt.show()


if __name__ == '__main__':
    time_start = time.clock()
    run()
    time_end = time.clock()
    print("time_used: {}".format(time_end - time_start))
