# MachineLearning
part 1 监督学习
已知输入样本集学习结果，建立预测模型，推演目标变量可能结果
目标变量一般有两种类型：标称型和数值型。

分类：
一 K-近邻算法
    对未知类别属性数据集中的每个点依次执行以下操作：
    1）计算已知类别数据集中的每个点到当前点的距离,
    需要归一化特征值，防止不同数量级的属性对分类影响，newvalue = (oldvalue-min)）/(max-min)
    2）按照距离递增排序
    3）选取与当前点最近的k个点
    4）确定前k个点的类别出现频率
    5）选择其中出现频率最高的点作为当前点的类别预测分类

二 决策树
    检测数据集中的每个子项是否属于同一个分类：
    如果是，则返回标签类
    否则
        寻找划分数据集的最好特征，（划分之后的信息增益最大，熵：信息的期望值,描述状态的不确定性，信息增益即熵减）
        创建分支节点
            对每个划分的子集
                递归调用创建决策树过程
        返回 分支节点

    根据生成的决策树，对待预测数据按照各个节点进行分类

三 朴素贝叶斯
    贝叶斯准则：p(c|x) = p(x|c)*p(c)/p(x), p(ci|x,y) = p(x,y|ci)*p(ci)/p(x,y)
    根据已有样本数据计算已知分类结果，到属性的条件概率；已知分类结果的概率；可以计算出已知属性，属于某个分类的概率，选择其中概率最大的作为预测分类

四 Logistic回归
    根据现有数据对分类边界线建立回归公式。训练分类器时的做法就是寻找最佳拟合参数，使用的是最优化算法。
    sigmoid函数：𝝈(z) = 1/(1+e*e^(-z))  将阶跃函数平滑化
    z = W0 * X0 + W1 * X1 + ... + Wn * Xn，即z = WT * X；sigmoid函数计算结果小于0.5，归入0类；大于0.5归入1类
    最优化算法：
    1）梯度上升法
        函数f(x,y)的梯度，∇f(x,y) = ⦗∂f(x,y)/∂x, ∂f(x,y)/∂y⦘，函数对各个变量的偏导数构成的向量，沿各个坐标轴的函数增长率，梯度算子始终指向函数值增长最快的方向。
        误差e(w) = (y - w*x)T * (y - w*x), 利用梯度下降求误差最小值或梯度上升算法求-e(w)最大值，得到如下公式：
        w := w + 𝜶∇f(w), 𝜶为迭代步长，w为数据点，用来求函数最大值
        梯度下降算法：w := w - 𝜶∇f(w)，用来求函数最小值
        步骤：
        每个回归系数初始化为1
        重复R次：
            计算整个数据集的梯度
            使用alpha*gradient更新回归系数的向量: w = w + 𝜶 * dataMatrix.transpose()*error,error为列向量，包含每个样本的误差
            返回回归系数
        缺点：每次更新回归系数需要遍历整个数据集，计算量比较大
    2）随机梯度上升算法
        所有回归系数初始化为1
        对数据集中每个样本
            计算该样本的梯度
            使用alpha*gradient更新回归系数，相对于梯度上升算法，这里每次只使用一个样本数据进行更新
        返回回归系数
        缺点：定长的步长导致数据来回波动，异常样本数据导致回归系数异常变化
    3）改进随机梯度上升算法
        根据迭代次数减少步长，
        随机选择样本值进行回归系数更新

五 支持向量机
    将数据集分割开来的“直线”称为分隔超平面，表述形式：W.T * X + b
    支持向量：离分隔超平面最近的那些点
    点A到超平面的距离：|W.T * A + b|/||W||，目标求支持向量到分隔平面的最大距离
    arg MAXw,b{MINn(label*(W.T*X + b))*1/||W||}，找出到分隔超平面最近的点即支持向量，选择合适的超平面使得支持向量到超平面距离最大化
    给定约束条件：label*(W.T*X + b) >= 1.0
    拉格朗日乘子法优化目标函数
    SMO算法(Sequential Minimal Optimization)将大优化问题分解为多个小优化问题,每次只优化2个alpha来加速SVM的训练速度
    利用核函数将非线性可分的数据映射成线性可分的数据点，常用的核函数有：径向基函数。
    将数据从一个特征空间映射到另一个特征空间，通常情况将低维特征空间映射到高维特征空间。这种映射是通过核函数实现的。
    径向基函数的高斯版本，将数据从其特征空间映射到更高维的空间，实际上是一个无穷维的空间：
        k(x,y) = exp{-||x-y||^2/(2𝝈)^2}, 𝝈是达到率，或者说函数值跌落到0的速度参数。

六 AdaBoost元算法
    元算法是对其他算法进行组合的一种方式，可以不同算法组合，也可以同一算法的不同设置下组合；有人认为AdaBoost是最好的监督学习方法
    自举汇聚法(bootstrap aggregating)，也成为bagging方法，是在从原始数据集选择S次后得到S个新数据集的一种技术。新数据集和原数据集大小相等。每个数据集都是通过在原始数据集中随机选择一个样本进行替换得到。意味着新数据集可以有重复值。
        在S个数据集建好之后，可以将某个学习算法分别作用于每个数据集得到S个分类器，使用S个分类器对新数据进行分类。更先进的bagging方法有 随机森林(random forest)
    boosting是一种与bagging很类似的技术，使用的多个分类器的类型是一致的。不同的是boosting的不同的分类器是通过串行训练获得的，每个新的分类器都根据已训练出来的分类器的性能来进行训练。Boosting是通过集中关注被已有分类器错分的那些数据来获得新的分类器。
    Boosting分类的结果是基于所有分类器的加权求和结果的，bagging中的分类器权重是相等的，每个权重代表其对应分类器在上一轮迭代中的成功度。
    boosting方法中最流行的一个版本AdaBoost(adaptive boosting)
    算法如下：
    训练数据中的每个样本，并赋予其一个权重，这些权重向量构成了向量D。一开始这些权重都初始化成相等值。
    首先在训练数据上训练出一个弱分类器(可以使用任意分类器作为弱分类器，简单分类器效果更好)并计算该分类器的错误率，然后在同一数据集上再次训练弱分类器。
    在分类器的第二次训练当中，将会重新调整每个样本的权重，其中第一次分对的样本权重降低，分错的样本权重提高；
    为了从所有弱分类器中得到最终的分类结果，AdaBoost为每个分类器都分配了一个权重值alpha，这些alpha是基于每个弱分类器的错误率进行计算的。其定义如下：
        ℇ = 未正确分类的样本数/所有样本数
    alpha计算公式：错误率越小权重越大
        𝜶 = 1/2 * ln((1-ℇ)/ℇ)
    计算出alpha值后，可以对权重向量D进行更新，以使正确分类的样本权重降低，错误分类样本的权重增加，以便于后续分类器对错误样本有更强的作用力。D计算方法如下：
    正确分类样本权重更新：Di(t+1) = Di(t)*e^-𝜶/Sum(D)
    错分样本权重更新：    Di(t+1) = Di(t)*e^𝜶/Sum(D)
    在计算出D之后，AdaBoost又开始进行下一轮迭代。算法会不断地重复训练和调整权重过程，知道训练错误率为0或者弱分类器的数目达到用户指定值为止。

    基于单层决策树（decision stump）构建弱分类器：
    将最小错误率minError设为+∞
    对数据集中的每一个特征（第一层循环）：
        对每个步长（第二层循环按步长遍历某个特征的所有值）：
            对每个不等号（第三层循环，根据与比较值的大小进行分类）：
                建立一颗单层决策树并利用加权数据集对它进行测试
                如果错误率低于minError，则将当前单层决策树设为最佳单层决策树
    返回最佳单层决策树

    AdaBoost算法实现：
    对每次迭代：
        利用上述单层决策树构建方法找到最佳的单层决策树
        将最佳单层决策树加入到单层决策树组
        计算alpha（分类器的权重）
        计算新的权重向量D（样本的权重）
        更新累计类别估计值
        如果错误率为0.0，则退出循环

    利用AdaBoost算法进行分类：
    将待分类数据应用到训练出来的多个弱分类器上进行简单分类，分类估计值乘以每个分类器的权重alpha，再求和，根据求和结果的符号确定预测分类是+1，或者-1类。

    非均衡分类问题：
    对马的死亡预测会导致马的安乐死，且预测的准确率不是100%；
    对垃圾邮件的过滤准确率也非100%，会导致正常邮件接收不到
    不同类别的分类代价不一样，需要根据分类结果的代价来调整分类的权重。
    其他的分类性能度量指标：
                    预测结果
                    +1              -1
    真实结果：  +1  真正例（TP）    伪反例（FN）
                -1  伪正例（FP）    真反例（TN）
    正确率：TP/(TP+FP)，预测为正例的样本中真实正例的比例。
    召回率：TP/(TP+FN)，预测为正例的真实正例占所有真实正例的比例。
    分类器很难同时保证高正确率和高召回率。
    ROC曲线，ROC代表接收者操作特征，横轴为伪正例比例FP/(FP+TN)，纵轴是真正例的比例TP/(TP+FN).
    在理想情况下，最佳的分类器应该尽可能处于左上角，意味着分类器在假阳率很低的同时获取了很高的真阳率。
    对不同的ROC曲线进行比较的一个指标是曲线下面积AUC，反映了分类器的平均性能值

    处理分均衡问题的数据抽样方法：
    欠抽样或者过抽样，解决在分类器训练时正例数目和反例数目相差较大时的情况。
    欠抽样删除正例较多的样本，过抽样则复制反例较少的样本。




开发机器学习程序的基本步骤：
1）收集数据
2）准备输入数据，准备合适的格式便于处理
3）分析输入数据，除异补漏
4）训练算法，根据输入样本训练算法，从中抽取知识或信息，建立模型
    对于无监督学习，因为没有目标变量，所以不需要训练算法
5）测试算法，取出适当的样本对建立的模型验证成功率，对于无监督学习要找其他方法验证
6）使用算法，利用建立的模型运用在实际环境，看能否正常工作

概述：
| 算法         | 优点                                                                           | 缺点                                                             | 适用数据范围   | 适用场景                 |
| k-近邻算法   | 精度高，对异常值不敏感，无数据输入假定                                         | 计算复杂度高，空间复杂度高                                       | 数值型和标称型 | 手写识别                 |
| 决策树       | 计算复杂度不高，输出结果容易理解，对中间值的缺失不敏感，可以处理不相关特征数据 | 可能会产生过度匹配问题                                           | 数值型和标称型 | 预测患者佩戴何种隐形眼镜 |
| 朴素贝叶斯   | 在数据较少情况下有效，可以处理多类别问题                                       | 对输入数据的准备方式较为敏感                                     | 标称型数据     | 过滤垃圾邮件             |
| Logistic回归 | 计算代价不高，易于理解和实现                                                   | 容易欠拟合                                                       | 数值型和标称型 | 疝气病预测马的死亡率     |
| 支持向量机   | 泛化错误率低，计算开销不大，结果易解释                                         | 对参数选择和核函数的选择敏感，原始分类器不加修改仅适用于二类问题 | 数值型和标称型 | 手写识别问题             |
| AdaBoost     | 泛化错误率低，易编码，可以应用在大部分分类器上，无参数调整                     | 对离群点敏感                                                     | 数值型和标称型 | 各种分类问题             |

