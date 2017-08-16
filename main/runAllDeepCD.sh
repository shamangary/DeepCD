
# Training DeepCD 2-stream (lead: real, complementary: binary) with DDM
th ./train/fun_DeepCD_2S.lua "rb" 128 256 1.41 1 5 "liberty" true 1e-4
th ./train/fun_DeepCD_2S.lua "rb" 128 256 1.41 1 5 "notredame" true 1e-4
th ./train/fun_DeepCD_2S.lua "rb" 128 256 1.41 1 5 "yosemite" true 1e-4

# Training DeepCD 2-stream (lead: real, complementary: binary) without DDM
th ./train/fun_DeepCD_2S.lua "rb" 128 256 1.41 1 5 "liberty" false
th ./train/fun_DeepCD_2S.lua "rb" 128 256 1.41 1 5 "notredame" false
th ./train/fun_DeepCD_2S.lua "rb" 128 256 1.41 1 5 "yosemite" false


# Training DeepCD 2-stream (lead: binary, complementary: binary) with DDM
th ./train/fun_DeepCD_2S.lua "bb" 512 256 1.41 1 2 "liberty" true 5e-4
th ./train/fun_DeepCD_2S.lua "bb" 512 256 1.41 1 2 "notredame" true 5e-4
th ./train/fun_DeepCD_2S.lua "bb" 512 256 1.41 1 2 "yosemite" true 1e-3

# Training DeepCD 2-stream (lead: binary, complementary: binary) without DDM
th ./train/fun_DeepCD_2S.lua "bb" 512 256 1.41 1 2 "liberty" false
th ./train/fun_DeepCD_2S.lua "bb" 512 256 1.41 1 2 "notredame" false
th ./train/fun_DeepCD_2S.lua "bb" 512 256 1.41 1 2 "yosemite" false
