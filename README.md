# Charben.GAN
下一代简化版深度学习· 探究深度生成对抗网络模型及其衍生
原文见：![GAN](https://arxiv.org/pdf/1406.2661.pdf) 2014年六月份由“GANs之父”Ian Goodfellow提出，Yoshua Bengio合作提出的一篇用于视觉领域的新一代深度模型。

##### 引言

我们通过(vis)一种对抗处理过程提出了一个新的框架用于预估(estimating)生成模型,即同时地(simultaneously)训练两个模型,一种生成模型G_model捕捉(捕捉)数据分布，一种判别模型D_model判别从训练集中由生成模型G_model生成样本的可能性。对G_model的训练过程是最大化D_model犯错的可能性。这个框架十分符合最小最大的二人博弈游戏，
