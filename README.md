# Recommendation-System-for-Songs
音乐推荐系统

我们从网上找到两个数据集，一个是用户和音乐的点播数据集，另一个是音乐的详细信息数据集。

利用上述数据，我们可以做一个音乐推荐系统。其中涉及到数据的读取，召回阶段和排序阶段。

在召回阶段，我们尝试了基于排行榜的推荐，基于协同过滤的推荐以及基于矩阵分解的推荐。我们选择矩阵分解来获得召回阶段的结果。

在排序阶段，我们用gbdt+lr的方式来对召回阶段的候选集进行排序。选择打分最高的几个作为最终排序结果。

详细介绍：https://blog.csdn.net/qq_30841655/article/details/107989560
