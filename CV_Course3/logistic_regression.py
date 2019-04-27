import numpy as np
import time


'''
仅仅在linear regression的基础上改了hypothesis(sigmoid)和cost function (cross-entropy-like cost)
还没有测试运行效果怎样
因为还没想好怎样生成一个可供验证的数据集
'''


def sigmoid(z): return 1./(1. + np.exp(-z))


def inference(w, b, x_list):
    """
    定义hypothesis:
    y = w * x + b
    """
    z_list = w * x_list + b
    return sigmoid(z_list)


def eval_cost(x_list, gt_y_list, w, b):
    """
    定义cost function (quadratic cost here)
    Cost = 1/2m * sum((pred_y - y_t) ** 2)
    """
    cost_list = -(gt_y_list * np.log(inference(w, b, x_list))
                  + (1-gt_y_list) * np.log(1 - inference(w, b, x_list)))
    avg_cost = cost_list.mean()
    return avg_cost


def cal_gradient(pred_y_list, gt_y_list, x_list):
    """
    计算梯度
    dJ/dw  --> dw (偏导)
    dJ/db  --> db
    """
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
    pred_y_list = np.array([inference(w, b, x_list[i]) for i in range(len(x_list))])
    avg_dw, avg_db = cal_gradient(pred_y_list, gt_y_list, x_list)
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
    for i in range(max_iter):
        batch_idxs = np.random.choice(len(x_list), batch_size, replace=False)
        batch_x = [x_list[j] for j in batch_idxs]
        batch_y = [gt_y_list[k] for k in batch_idxs]
        w, b = gradient_descent(batch_x, batch_y, w, b, lr)
        if not i % 100:
            print("w: {:.4f} | b: {:.4f}".format(w, b))
            print("current loss: {}".format(eval_cost(x_list, gt_y_list, w, b)))


def generate_sample_data():
    """
    模拟产生数据集用于训练:
    先产生w, b
    再产生x_list
    再根据w, b和x_list产生y_list
    """
    w = np.random.randint(0, 10) + np.random.random()
    b = np.random.randint(0, 5) + np.random.random()
    num_samples = 1000
    x_list = np.random.randint(0, 10, num_samples) * np.random.random()
    y_list = w * x_list + b + np.random.random() * np.random.randint(-1, 1)
    return x_list, y_list, w, b


def run():
    x_list, y_list, w, b = generate_sample_data()
    lr = 0.01
    max_iter = 2000
    batch_size = 50
    train(x_list, y_list, batch_size, lr, max_iter)
    print("true_w = {:.4f} | true_b = {:.4f}".format(w, b))


if __name__ == '__main__':
    time_start = time.clock()
    run()
    time_end = time.clock()
    print("time_used: {}".format(time_end - time_start))

