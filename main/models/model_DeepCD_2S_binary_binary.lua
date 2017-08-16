function createModel(pT)
	-- setup the CNN
	model = nn.Sequential() 
	local CT = nn.ConcatTable()

	local lead_net = nn.Sequential()
	lead_net:add(cudnn.SpatialConvolution(1, 32, 7, 7))
	lead_net:add(cudnn.Tanh(true))
	lead_net:add(cudnn.SpatialMaxPooling(2,2,2,2)) 
	lead_net:add(cudnn.SpatialConvolution(32, 64, 6, 6))
	lead_net:add(cudnn.Tanh(true))
	lead_net:add(nn.View(64*8*8))
	lead_net:add(nn.Linear(64*8*8, math.max(128,pT.bits1/2)))
	lead_net:add(cudnn.Tanh(true))
	lead_net:add(nn.Linear(math.max(128,pT.bits1/2), pT.bits1))
	lead_net:add(nn.MulConstant(pT.scal_sigmoid, true))
	lead_net:add(cudnn.Sigmoid())



	local com_net = nn.Sequential()
	com_net:add(cudnn.SpatialConvolution(1, 32, 7, 7))
	com_net:add(cudnn.Tanh(true))
	com_net:add(cudnn.SpatialMaxPooling(2,2,2,2)) 
	com_net:add(cudnn.SpatialConvolution(32, 64, 6, 6))
	com_net:add(cudnn.Tanh(true))
	com_net:add(nn.View(64*8*8))
	com_net:add(nn.Linear(64*8*8, math.max(128,pT.bits2/2)))
	com_net:add(cudnn.Tanh(true))
	com_net:add(nn.Linear(math.max(128,pT.bits2/2), pT.bits2))
	com_net:add(nn.MulConstant(pT.scal_sigmoid, true))
	com_net:add(cudnn.Sigmoid())
	com_net:add(nn.MulConstant(pT.w_com,true))

	CT:add(lead_net)
	CT:add(com_net)
	model:add(CT)
	return model
	
end