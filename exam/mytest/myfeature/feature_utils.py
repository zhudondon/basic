import numpy as np

# 标准化
def normalize(features):

    # 复制 特征，转成 float
    features_normalized = np.copy(features).astype(float)

    # 计算均值 0->行 1->列
    features_mean = np.mean(features, 0)

    # 计算标准差
    features_deviation = np.std(features, 0)

    # 标准化操作 减去平均值
    if features.shape[0] > 1:
        features_normalized -= features_mean

    # 防止除以0
    features_deviation[features_deviation == 0] = 1
    features_normalized /= features_deviation

    return features_normalized, features_mean, features_deviation

#sin函数处理
def generate_sinusoids(dataset, sinusoid_degree):
    """
    sin(x).
    """

    num_examples = dataset.shape[0]
    #初始化一个空矩阵
    sinusoids = np.empty((num_examples, 0))

    # 根据次数 转换生成多个特征值，axis=1 就是 往y坐标 新增
    for degree in range(1, sinusoid_degree + 1):
        sinusoid_features = np.sin(degree * dataset)
        sinusoids = np.concatenate((sinusoids, sinusoid_features), axis=1)

    return sinusoids


# 二项式 生成
def generate_polynomials(dataset, polynomial_degree, normalize_data=False):
    """变换方法：
    x1, x2, x1^2, x2^2, x1*x2, x1*x2^2, etc.
    """
    #
    features_split = np.array_split(dataset, 2, axis=1)
    dataset_1 = features_split[0]
    dataset_2 = features_split[1]

    (num_examples_1, num_features_1) = dataset_1.shape
    (num_examples_2, num_features_2) = dataset_2.shape

    if num_examples_1 != num_examples_2:
        raise ValueError('Can not generate polynomials for two sets with different number of rows')

    if num_features_1 == 0 and num_features_2 == 0:
        raise ValueError('Can not generate polynomials for two sets with no columns')

    if num_features_1 == 0:
        dataset_1 = dataset_2
    elif num_features_2 == 0:
        dataset_2 = dataset_1

    num_features = num_features_1 if num_features_1 < num_examples_2 else num_features_2
    dataset_1 = dataset_1[:, :num_features]
    dataset_2 = dataset_2[:, :num_features]

    polynomials = np.empty((num_examples_1, 0))

    for i in range(1, polynomial_degree + 1):
        for j in range(i + 1):
            polynomial_feature = (dataset_1 ** (i - j)) * (dataset_2 ** j)
            polynomials = np.concatenate((polynomials, polynomial_feature), axis=1)

    if normalize_data:
        polynomials = normalize(polynomials)[0]

    return polynomials



def prepare_for_training(data, polynomial_degree = 0, sinusoid_degree=0, normalize_data=True):
    #计算样本总数，data是个矩阵
    shape_ = data.shape[0]

    ndarray = np.copy(data)

    #预处理 均值，偏差
    features_mean = 0
    features_deviation = 0
    data_normalized = ndarray

    if normalize_data:
        (
            data_normalized,
            features_mean,
            features_deviation
        ) = normalize(ndarray)

    # 特征变换sinusoidal
    if sinusoid_degree > 0:
        sinusoids = generate_sinusoids(data_normalized, sinusoid_degree)
        data_processed = np.concatenate((data_processed, sinusoids), axis=1)