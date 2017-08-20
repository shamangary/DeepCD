# DeepCD
Code Author: Tsun-Yi Yang

Last update: 2017/08/17 (Training and testing codes are both uploaded.)

Platform: Ubuntu 14.04, Torch7

Paper
-
**[ICCV17] DeepCD: Learning Deep Complementary Descriptors for Patch Representations**

**Authors: [Tsun-Yi Yang](http://shamangary.logdown.com/), Jo-Han Hsu, [Yen-Yu Lin](https://www.citi.sinica.edu.tw/pages/yylin/index_zh.html), and [Yung-Yu Chuang](https://www.csie.ntu.edu.tw/~cyy/)**

**PDF:** 
+ Link1: http://www.csie.ntu.edu.tw/~cyy/publications/papers/Yang2017DLD.pdf
+ Link2: https://github.com/shamangary/DeepCD/blob/master/1557.pdf

Code abstract
-
This is the source code of DeepCD. The training is done on Brown dataset.

Two distinct descriptors are learned for the same network.

Product late fusion in distance domain is performed before the final ranking.

DeepCD project is heavily inspired by pnnet https://github.com/vbalnt/pnnet

***This respository:***
***Author: Tsun-Yi Yang***
+ Brown dataset (Training and testing) https://github.com/shamangary/DeepCD/tree/master/main

***Related respositories:***
***Author: Jo-Han Hsu***
+ MVS dataset (testing) https://github.com/Rohan8288/DeepCD_MVS
+ Oxford dataset (testing) https://github.com/Rohan8288/DeepCD_Oxford

Model
-
<img src="https://github.com/shamangary/DeepCD/blob/master/models_word.png" height="400"/>

Training with Data-Dependent Modulation (DDM) layer
-
+ DDM layer dynamically adapt the learning rate of the complementary stream.

+ It consider information of the whole batch by considering both leading and complementary distances.

The backward gradient value is scaled by a factor η (1e-3~1e-4). This step not only let us to slow down the learning of fully connected layer inside DDM layer, but also let us to approximately ignore the effect of DDM layer on the forward propagation of the complementary stream and make it an identity operation. The update equation is basically the the backward equation derived from multipling a parameter w from the previous layer.

<img src="https://github.com/shamangary/DeepCD/blob/master/DDM.png" height="300"/><img src="https://github.com/shamangary/DeepCD/blob/master/DeepCD_triplet.png" height="300"/>

```
a_DDM = nn.Identity()
output_layer_DDM = nn.Linear(pT.batch_size*2,pT.batch_size)
output_layer_DDM.weight:fill(0)
output_layer_DDM.bias:fill(1)
b_DDM = nn.Sequential():add(nn.Reshape(pT.batch_size*2,false)):add(output_layer_DDM):add(nn.Sigmoid())
DDM_ct1 = nn.ConcatTable():add(a_DDM:clone()):add(b_DDM:clone())
DDM_layer = nn.Sequential():add(DDM_ct1):add(nn.DataDependentModule(pT.DDM_LR))
```
Testing stage
-
+ A **hard threshold** will be appied on the complementary descriptor before the Hamming distance calculation.

+ **DDM layer is not involved in the testing stage** since we only need the trained model from the triplet structure.

+ **Product late fusion at distance domain** is computed before the final ranking.

Brown dataset results
-
<img src="https://github.com/shamangary/DeepCD/blob/master/DeepCD_brown.png" height="400"/>

Pretrained models
-
coming soon...
