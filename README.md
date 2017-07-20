# DeepCD
Code Author: Tsun-Yi Yang

Last update: 2017/07/20 (Full training and testing will be released soon...)

Platform: Ubuntu 14.04, Torch7

Paper
-
**[ICCV17] DeepCD: Learning Deep Complementary Descriptors for Patch Representations**

**Authors: [Tsun-Yi Yang](http://shamangary.logdown.com/), Jo-Han Hsu, [Yen-Yu Lin](https://www.citi.sinica.edu.tw/pages/yylin/index_zh.html), and [Yung-Yu Chuang](https://www.csie.ntu.edu.tw/~cyy/)**

Code abstract
-
This is the source code of DeepCD. The training is done on Brown dataset.

DeepCD project is heavily inspired by pnnet https://github.com/vbalnt/pnnet



Model
-
<img src="https://github.com/shamangary/DeepCD/blob/master/models_word.png" height="400"/>

Data-Dependent Modulation (DDM) layer
-
DDM layer dynamically adapt the learning rate of the complementary stream
by considering both leading and complementary distances.

<img src="https://github.com/shamangary/DeepCD/blob/master/DDM.png" height="300"/>

(Coming soon...)


Brown dataset results
-
<img src="https://github.com/shamangary/DeepCD/blob/master/DeepCD_brown.png" height="400"/>

Pretrained models
-
coming soon...
