require 'nn'
require '../utils.lua'
require 'image'
require 'optim'
require './DistanceRatioCriterion_allW.lua'
require 'cudnn'
require 'cutorch'
require 'cunn'
require './allWeightedMSECriterion.lua'
require './DataDependentModule.lua'

local matio = require 'matio'

des_type = arg[1]
if des_type == 'rb' then
  dim1 = tonumber(arg[2])
  bits2 = tonumber(arg[3])
elseif des_type == 'bb' then
  bits1 = tonumber(arg[2])
  bits2 = tonumber(arg[3])
end
w_com = tonumber(arg[4])
ws_lead = tonumber(arg[5])
ws_pro = tonumber(arg[6])
name = arg[7]
if arg[8] == 'true' then
  isDDM = true
elseif arg[8] == 'false' then
  isDDM = false
end
DDM_LR = tonumber(arg[9])

pT = {

  --dim1 = 128,
  --bits2 = 256,
  --w_com= 1.41,
  --ws_lead = 1,
  --ws_pro = 5,
  scal_sigmoid = 100,
  isNorm = true,
  --isSTN = false,
  --name = 'liberty',
  batch_size = 128,
  num_triplets = 1280000,
  max_epoch = 100

}


if des_type == 'rb' then
  pT.dim1 = dim1
  pT.bits2 = bits2
elseif des_type == 'bb' then
  pT.bits1 = bits1
  pT.bits2 = bits2
end
pT.w_com = w_com
pT.ws_lead = ws_lead
pT.ws_pro = ws_pro
pT.name = name
pT.isDDM = isDDM
pT.DDM_LR = DDM_LR

-- optim parameters
optimState = {
  learningRate = 0.1,
  weightDecay = 1e-4,
  momentum = 0.9,
  learningRateDecay = 1e-6
}
pT.optimState = optimState

if pT.isDDM then
  addName = 'DDM_'
else
  addName = ''
end

if des_type == 'rb' then
  folder_name = './train_epoch/DeepCD_2S_'..addName..pT.dim1..'dim1_'..pT.bits2..'bits2_'..pT.name
elseif des_type == 'bb' then
  folder_name = './train_epoch/DeepCD_2S_'..addName..pT.bits1..'bits1_'..pT.bits2..'bits2_'..pT.name
end

print(pT)
os.execute("mkdir -p " .. folder_name)
if pT.isDDM then
  os.execute("mkdir -p " .. folder_name.."/DDM_vec/")
end
torch.save(folder_name..'/ParaTable_DeepCD_2S_'..addName..pT.name..'.t7',pT)


------------------------------------------------------------------------------------------------


-- number of threads
torch.setnumthreads(13)

-- read training data, save mu and sigma & normalize

traind = read_brown_data('../UBCdataset/'..pT.name)
stats = get_stats(traind)
print(stats)
if pT.isNorm then
  norm_data(traind,stats)
end
print("==> read the dataset")

-- generate random triplets for training data

training_triplets = generate_triplets(traind, pT.num_triplets)
print("==> created the tests")
------------------------------------------------------------------------------------------------
if des_type == 'rb' then
  paths.dofile('../models/model_DeepCD_2stream.lua')
elseif des_type == 'bb' then
  paths.dofile('../models/model_DeepCD_2S_binary_binary.lua')
end
model1 = createModel(pT)
model1:training()

--  clone the other two networks in the triplet
model2 = model1:clone('weight', 'bias','gradWeight','gradBias')
model3 = model1:clone('weight', 'bias','gradWeight','gradBias')

-- add them to a parallel table
prl = nn.ParallelTable()
prl:add(model1)
prl:add(model2)
prl:add(model3)
prl:cuda()
------------------------------------------------------------------------------------------------
mlp= nn.Sequential()
mlp:add(prl)

-- get feature distances 
cc = nn.ConcatTable()


-- feats 1 with 2 leading
cnn_left1_lead = nn.Sequential()

cnnneg1_dist_lead = nn.ConcatTable()
a_neg1_lead = nn.Sequential()
a_neg1_lead:add(nn.SelectTable(1))
a_neg1_lead:add(nn.SelectTable(1))
b_neg1_lead = nn.Sequential()
b_neg1_lead:add(nn.SelectTable(2))
b_neg1_lead:add(nn.SelectTable(1))
cnnneg1_dist_lead:add(a_neg1_lead)
cnnneg1_dist_lead:add(b_neg1_lead)

cnn_left1_lead:add(cnnneg1_dist_lead)
cnn_left1_lead:add(nn.PairwiseDistance(2))
cnn_left1_lead:add(nn.View(pT.batch_size ,1))
cnn_left1_lead:cuda()
cc:add(cnn_left1_lead)

-- feats 1 with 2 completing
cnn_left1_com = nn.Sequential()

cnnneg1_dist_com = nn.ConcatTable()
a_neg1_com = nn.Sequential()
a_neg1_com:add(nn.SelectTable(1))
a_neg1_com:add(nn.SelectTable(2))
b_neg1_com = nn.Sequential()
b_neg1_com:add(nn.SelectTable(2))
b_neg1_com:add(nn.SelectTable(2))
cnnneg1_dist_com:add(a_neg1_com)
cnnneg1_dist_com:add(b_neg1_com)

cnn_left1_com:add(cnnneg1_dist_com)
cnn_left1_com:add(nn.PairwiseDistance(2))
cnn_left1_com:add(nn.View(pT.batch_size ,1))
cnn_left1_com:cuda()
cc:add(cnn_left1_com)

-- feats 2 with 3 leading
cnn_left2_lead = nn.Sequential()
cnnneg2_dist_lead = nn.ConcatTable()
a_neg2_lead = nn.Sequential()
a_neg2_lead:add(nn.SelectTable(2))
a_neg2_lead:add(nn.SelectTable(1))
b_neg2_lead = nn.Sequential()
b_neg2_lead:add(nn.SelectTable(3))
b_neg2_lead:add(nn.SelectTable(1))
cnnneg2_dist_lead:add(a_neg2_lead)
cnnneg2_dist_lead:add(b_neg2_lead)
cnn_left2_lead:add(cnnneg2_dist_lead)
cnn_left2_lead:add(nn.PairwiseDistance(2))
cnn_left2_lead:add(nn.View(pT.batch_size,1))
cnn_left2_lead:cuda()
cc:add(cnn_left2_lead)

-- feats 2 with 3 completing
cnn_left2_com = nn.Sequential()
cnnneg2_dist_com = nn.ConcatTable()
a_neg2_com = nn.Sequential()
a_neg2_com:add(nn.SelectTable(2))
a_neg2_com:add(nn.SelectTable(2))
b_neg2_com = nn.Sequential()
b_neg2_com:add(nn.SelectTable(3))
b_neg2_com:add(nn.SelectTable(2))
cnnneg2_dist_com:add(a_neg2_com)
cnnneg2_dist_com:add(b_neg2_com)
cnn_left2_com:add(cnnneg2_dist_com)
cnn_left2_com:add(nn.PairwiseDistance(2))
cnn_left2_com:add(nn.View(pT.batch_size ,1))
cnn_left2_com:cuda()
cc:add(cnn_left2_com)

-- feats 1 with 3 leading
cnn_right_lead = nn.Sequential()
cnnpos_dist_lead = nn.ConcatTable()
a_pos_lead = nn.Sequential()
a_pos_lead:add(nn.SelectTable(1))
a_pos_lead:add(nn.SelectTable(1))
b_pos_lead = nn.Sequential()
b_pos_lead:add(nn.SelectTable(3))
b_pos_lead:add(nn.SelectTable(1))
cnnpos_dist_lead:add(a_pos_lead)
cnnpos_dist_lead:add(b_pos_lead)
cnn_right_lead:add(cnnpos_dist_lead)
cnn_right_lead:add(nn.PairwiseDistance(2))
cnn_right_lead:add(nn.View(pT.batch_size ,1))
cnn_right_lead:cuda()
cc:add(cnn_right_lead)

-- feats 1 with 3 completing
cnn_right_com = nn.Sequential()
cnnpos_dist_com = nn.ConcatTable()
a_pos_com = nn.Sequential()
a_pos_com:add(nn.SelectTable(1))
a_pos_com:add(nn.SelectTable(2))
b_pos_com = nn.Sequential()
b_pos_com:add(nn.SelectTable(3))
b_pos_com:add(nn.SelectTable(2))
cnnpos_dist_com:add(a_pos_com)
cnnpos_dist_com:add(b_pos_com)
cnn_right_com:add(cnnpos_dist_com)
cnn_right_com:add(nn.PairwiseDistance(2))
cnn_right_com:add(nn.View(pT.batch_size ,1))
cnn_right_com:cuda()
cc:add(cnn_right_com)


cc:cuda()

mlp:add(cc)
------------------------------------------------------------------------------------------------
last_layer = nn.ConcatTable()


-- select leading min negative distance inside the triplet
mined_neg = nn.Sequential()
mining_layer = nn.ConcatTable()
mining_layer:add(nn.SelectTable(1))
mining_layer:add(nn.SelectTable(3))
mined_neg:add(mining_layer)
mined_neg:add(nn.JoinTable(2))
mined_neg:add(nn.Min(2))
mined_neg:add(nn.View(pT.batch_size ,1))
last_layer:add(mined_neg)

-- add leading positive distance
pos_layer = nn.Sequential()
pos_layer:add(nn.SelectTable(5))
pos_layer:add(nn.View(pT.batch_size ,1))
last_layer:add(pos_layer)


------------------------------------------------------------------------------------------------
a_DDM = nn.Identity()
output_layer_DDM = nn.Linear(pT.batch_size*2,pT.batch_size)
output_layer_DDM.weight:fill(0)
output_layer_DDM.bias:fill(1)
b_DDM = nn.Sequential():add(nn.Reshape(pT.batch_size*2,false)):add(output_layer_DDM):add(nn.Sigmoid())
DDM_ct = nn.ConcatTable():add(a_DDM:clone()):add(b_DDM:clone())
DDM_layer = nn.Sequential():add(DDM_ct):add(nn.DataDependentModule(pT.DDM_LR))
------------------------------------------------------------------------------------------------


--add neg1 (real,binary) distance
neg1_RB_layer = nn.Sequential()
temp_neg1_RB = nn.ConcatTable()

S1_neg1 = nn.Sequential()
S1_neg1:add(nn.SelectTable(1))

S2_neg1 = nn.Sequential()
S2_neg1:add(nn.SelectTable(2))

temp_neg1_RB:add(S1_neg1)
temp_neg1_RB:add(S2_neg1)
neg1_RB_layer:add(temp_neg1_RB)
if pT.isDDM then
  neg1_RB_layer:add(nn.JoinTable(2))
  neg1_RB_layer:add(DDM_layer)
  neg1_RB_layer:add(nn.SplitTable(2))
end
neg1_RB_layer:add(nn.CMulTable())
neg1_RB_layer:add(nn.Sqrt())
neg1_RB_layer:add(nn.View(pT.batch_size ,1))
neg1_RB_layer:cuda()
last_layer:add(neg1_RB_layer)

--add neg2 (real,binary) distance
neg2_RB_layer = nn.Sequential()
temp_neg2_RB = nn.ConcatTable()

S1_neg2 = nn.Sequential()
S1_neg2:add(nn.SelectTable(3))

S2_neg2 = nn.Sequential()
S2_neg2:add(nn.SelectTable(4))

temp_neg2_RB:add(S1_neg2)
temp_neg2_RB:add(S2_neg2 )
neg2_RB_layer:add(temp_neg2_RB)
if pT.isDDM then
  neg2_RB_layer:add(nn.JoinTable(2))
  neg2_RB_layer:add(DDM_layer)
  neg2_RB_layer:add(nn.SplitTable(2))
end
neg2_RB_layer:add(nn.CMulTable())
neg2_RB_layer:add(nn.Sqrt())
neg2_RB_layer:add(nn.View(pT.batch_size ,1))
neg2_RB_layer:cuda()
last_layer:add(neg2_RB_layer)


--add pos (real,binary) distance
pos_RB_layer = nn.Sequential()
temp_pos_RB = nn.ConcatTable()

S1_pos = nn.Sequential()
S1_pos:add(nn.SelectTable(5))
S2_pos = nn.Sequential()
S2_pos:add(nn.SelectTable(6))

temp_pos_RB:add(S1_pos)
temp_pos_RB:add(S2_pos)
pos_RB_layer:add(temp_pos_RB)
pos_RB_layer:add(nn.CMulTable())
pos_RB_layer:add(nn.Sqrt())

pos_RB_layer:add(nn.View(pT.batch_size ,1))
pos_RB_layer:cuda()
last_layer:add(pos_RB_layer)
------------------------------------------------------------------------------------------------
mlp:add(last_layer)

mlp:add(nn.JoinTable(2))
mlp:cuda()
------------------------------------------------------------------------------------------------
-- setup the criterion: ratio of min-negative to positive
epoch = 1



x=torch.zeros(pT.batch_size,1,32,32):cuda()
y=torch.zeros(pT.batch_size,1,32,32):cuda()
z=torch.zeros(pT.batch_size,1,32,32):cuda()



parameters, gradParameters = mlp:getParameters()







-- main training loop

Loss = torch.zeros(1,pT.max_epoch)



w = torch.zeros(128,5)
w[{{},{1,2}}]:fill(pT.ws_lead)
w[{{},{3,5}}]:fill(pT.ws_pro)
crit=nn.DistanceRatioCriterion_allW(w):cuda()
   
for epoch=epoch, pT.max_epoch do



   Gerr = 0
   shuffle = torch.randperm(pT.num_triplets)   
   nbatches = pT.num_triplets/pT.batch_size

   for k=1,nbatches-1 do
      xlua.progress(k+1, nbatches)

      s = shuffle[{ {k*pT.batch_size,k*pT.batch_size+pT.batch_size} }]
      for i=1,pT.batch_size do 
      	 x[i] = traind.patches32[training_triplets[s[i]][1]]
      	 y[i] = traind.patches32[training_triplets[s[i]][2]]
	 z[i] = traind.patches32[training_triplets[s[i]][3]]
      end

      local feval = function(f)
	 if f ~= parameters then parameters:copy(f) end
	 gradParameters:zero()
	 inputs = {x,y,z}
	 local outputs = mlp:forward(inputs)
	
	 local f = crit:forward(outputs, 1)
	 Gerr = Gerr+f
	 local df_do = crit:backward(outputs)
	 mlp:backward(inputs, df_do)
	 return f,gradParameters
      end
      optim.sgd(feval, parameters, optimState)

   end
   loss = Gerr/nbatches
   Loss[{{1},{epoch}}] = loss
   if pT.isDDM then
     print(DDM_vec)
     matio.save(folder_name..'/DDM_vec/DDM_vec_epoch'..epoch..'.mat',{DDM_vec=DDM_vec:double()})
   end

   print('==> epoch '..epoch)
   print(loss)
   print('')

   --remain = math.fmod(epoch,3)
   --if epoch == 1 or remain ==0 then
      net_save = mlp:get(1):get(1):clone()
      torch.save(folder_name..'/NET_DeepCD_2S_'..addName..pT.name..'_epoch'..epoch..'.t7',net_save:clearState())
   --end

end

torch.save(folder_name..'/Loss_DeepCD_2S_'..addName..pT.name..'.t7',Loss)



