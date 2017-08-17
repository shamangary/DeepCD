# Uncomment the command you want to operate.
# Example: 
#----------------------------------------------------------------------------------------------------------
# th ./eval/fun_evalDeepCD_2S.lua "rb" 128 256 "liberty" "notredame" true 1 
#----------------------------------------------------------------------------------------------------------
# "rb": real-valued for leading and binary for complementary
# 128 256: 128 dim for leading and 256 bits for complementary
# true: DDM training is true. Otherwise its false
# The last input is the epoch number of the model.


#----------------------------------------------------------------------------------------------------------
# Evaluate DeepCD 2-stream (lead: real, complementary: binary) with DDM
th ./eval/fun_evalDeepCD_2S.lua "rb" 128 256 "liberty" "notredame" true 1 


#----------------------------------------------------------------------------------------------------------
# Evaluate DeepCD 2-stream (lead: real, complementary: binary) without DDM
#th ./eval/fun_evalDeepCD_2S.lua "rb" 128 256 "liberty" "notredame" false 1 


#----------------------------------------------------------------------------------------------------------
# Training DeepCD 2-stream (lead: binary, complementary: binary) with DDM
#th ./eval/fun_evalDeepCD_2S.lua "bb" 512 256 "liberty" "notredame" true 1 


#----------------------------------------------------------------------------------------------------------
# Training DeepCD 2-stream (lead: binary, complementary: binary) without DDM
#th ./eval/fun_evalDeepCD_2S.lua "bb" 512 256 "liberty" "notredame" false 1 
