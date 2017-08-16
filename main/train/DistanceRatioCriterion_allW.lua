-- Taken from Elad Hoffer's TripletNet https://github.com/eladhoffer 
-- Hinge loss ranking could also be used, see below
-- https://github.com/torch/nn/blob/master/doc/criterion.md#nn.MarginRankingCriterion
		
local DistanceRatioCriterion_allW, parent = torch.class('nn.DistanceRatioCriterion_allW', 'nn.Criterion')

function DistanceRatioCriterion_allW:__init(w)
    parent.__init(self)
    self.SoftMax = nn.SoftMax()
    self.wMSE = nn.allWeightedMSECriterion(w)
    -- wMSE:forward(a,b) equals to "sum((a*w-b*w).^2)/dim"
    self.Target = torch.Tensor()
end

function DistanceRatioCriterion_allW:createTarget(input, target)
    local target = target or 1
    self.Target:resizeAs(input):typeAs(input):zero()
    self.Target[{{},target}]:add(1)
    --self.Target[{{},target+1}]:add(1)
    
    self.Target[{{},3}]:add(1)
    self.Target[{{},4}]:add(1)
    --self.Target[{{},5}]:add(1)
    --self.Target[{{},7}]:add(1)
--print(input:size()) 
--print(self.Target:size()) 
--os.exit()
--[[
    print(input) 
    ...
     0.2617  0.5276
     0.1764  0.3031
     0.4771  0.3169
     0.3398  0.0905
     0.3689  0.1940
    --[torch.CudaTensor of size 128x2]
    
    print(self.Target) 
    ...
     1  0
     1  0
     1  0
     1  0
     1  0
    [torch.CudaTensor of size 128x2]

    --The first column contains negtive distance
    --while the second one the is positive distance
    os.exit()
    --]]
end

function DistanceRatioCriterion_allW:updateOutput(input, target)
    if not self.Target:isSameSizeAs(input) then
        self:createTarget(input, target)
    end
    self.output = self.wMSE:updateOutput(self.SoftMax:updateOutput(input),self.Target)
    return self.output
end

function DistanceRatioCriterion_allW:updateGradInput(input, target)
    if not self.Target:isSameSizeAs(input) then
        self:createTarget(input, target)
    end

    self.gradInput = self.SoftMax:updateGradInput(input, self.wMSE:updateGradInput(self.SoftMax.output,self.Target))
    return self.gradInput
end

function DistanceRatioCriterion_allW:type(t)
    parent.type(self, t)
    self.SoftMax:type(t)
    self.wMSE:type(t)
    self.Target = self.Target:type(t)
    return self
end
