require 'cutorch'
require 'xlua'
require 'trepl'
require 'cunn'
require 'cudnn'
require 'image'
require 'nn'
require 'torch'
require 'gnuplot'
require '../utils.lua'

des_type = arg[1]
if des_type == 'rb' then
  dim1 = tonumber(arg[2])
  bits2 = tonumber(arg[3])
elseif des_type == 'bb' then
  bits1 = tonumber(arg[2])
  bits2 = tonumber(arg[3])
end
network = arg[4]
eval_data = arg[5]
if arg[6] == 'true' then
  isDDM = true
elseif arg[6] == 'false' then
  isDDM = false
end
checkEpoch = arg[7]

-- load default 128 out tanh-maxpooling network trained on liberty dataset
-- for more details http://phototour.cs.washington.edu/patches/default.htm

if isDDM then
   addName = '_DDM'
else
   addName = ''
end

if des_type == 'rb' then
   net = torch.load('./train_epoch/DeepCD_2S'..addName..'_'..dim1..'dim1_'..bits2..'bits2_'..network..'/NET_DeepCD_2S'..addName..'_'..network..'_epoch'..checkEpoch..'.t7')
elseif des_type == 'bb' then
   net = torch.load('./train_epoch/DeepCD_2S'..addName..'_'..bits1..'bits1_'..bits2..'bits2_'..network..'/NET_DeepCD_2S'..addName..'_'..network..'_epoch'..checkEpoch..'.t7')
end
net:evaluate()

-- test on the testing gt (100k pairs from Brown's dataset)
ntest = 100000
R = torch.ones(2,ntest)
print(net)
net:get(1):get(2):remove(12)

trained = torch.load('./UBCdataset/'..network..'.t7')
dataset = torch.load('./UBCdataset/'..eval_data..'.t7')
stats = get_stats(trained)
print(stats)
norm_data(dataset,stats)
npatches =  (dataset.patches32:size(1))
print(npatches)

   -- normalize data
   patches32 = dataset.patches32
   
   -- split the patches in batches to avoid memory problems
   BatchSize = 128



for iter=1, 2 do
   

   if iter == 1 then
      if des_type == 'rb' then
         dim = dim1
         isBinary = false
      elseif des_type == 'bb' then
         dim = bits1
         isBinary = true
      end
      net:add(nn.SelectTable(1))

   elseif iter == 2 then
      dim = bits2
      isBinary = true
      net:remove(2)
      net:add(nn.SelectTable(2))

   end



  

   local Descrs = torch.CudaTensor(npatches,dim)
   local DescrsSplit = Descrs:split(BatchSize)
   for i,v in ipairs(patches32:split(BatchSize)) do
      temp = v:clone():cuda()
      DescrsSplit[i]:copy(net:forward(temp))
      --print(net:get(1):get(2):get(1):get(1):get(2):get(11).output)
--os.exit()
   end



   

   for j=1,ntest do
      l = dataset.gt100k[j]
      lbl = l[2]==l[5] and 1 or 0
      id1 = l[1]+1
      id2 = l[4]+1
      dl = Descrs[{ {id1},{} }]
      dr = Descrs[{ {id2},{} }]

      if isBinary then
         dl = dl:gt(0.5):float()
         dr = dr:gt(0.5):float()
      end
      d = torch.dist(dl,dr)


         if iter == 1 then


            R[{{1},{j}}] = lbl
            R[{{2},{j}}] = R[{{2},{j}}]*d
         elseif iter ==2 then

               R[{{2},{j}}] = R[{{2},{j}}]*d

            
         end

      --io.write(string.format("%d %.4f \n", lbl,d))
   end
end








--FPR95(FPR at TPR or recall 95%) and ROC curve
val_sorted, temp_id_sorted = torch.sort(R[{{2},{}}])
id_sorted = temp_id_sorted[{1,{}}]
pn_sorted = R[{{1},{}}]:index(2,id_sorted)


pos_all = torch.sum(R[{{1},{}}])
print('pos_all:'..pos_all)
neg_all =  ntest-pos_all
pos_95 = torch.floor(pos_all*0.95)

pos_acc = pn_sorted:clone() --Don't forget "clone()"

for j=2,ntest do
	pos_acc[{1,j}] = pos_acc[{1,j-1}] + pos_acc[{1,j}]
end


print('pos_95:'..pos_95)
id_95 = pos_acc[{1,{}}]:eq(pos_95):nonzero():min()
print(id_95)



tpr = torch.Tensor(1,ntest)
fpr = torch.Tensor(1,ntest)
for k=1,ntest do
	tpr[{1,k}] = pos_acc[{1,k}]/pos_all
	fpr[{1,k}] = torch.abs(pos_acc[{1,k}]-k)/neg_all
end



Result = torch.cat(fpr,tpr,1)
gnuplot.plot(Result:t())
gnuplot.title('ROC curve')
gnuplot.xlabel('FPR')
gnuplot.ylabel('TPR')
FPR95 = fpr[{1,id_95}]
print('FPR95%:'.. FPR95*100)




