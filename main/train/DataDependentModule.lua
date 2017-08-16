local DataDependentModule, parent = torch.class('nn.DataDependentModule', 'nn.Module')

function DataDependentModule:__init(DDM_learning_rate)
   parent.__init(self)
   self.gradInput = {}
   self.DDM_learning_rate = DDM_learning_rate or 0
end

function DataDependentModule:updateOutput(input)
   self.output = input[1]
   return self.output
end


function DataDependentModule:updateGradInput(input, gradOutput)


      self.gradInput[1] = gradOutput:clone()

      DDM_vec = input[2]
      --print(DDM_vec)
      self.gradInput[2] = gradOutput[{{},2}]:clone()
      for i=1, self.gradInput[1]:size(1) do
         self.gradInput[1][{i,2}] = self.gradInput[1][{i,2}]*DDM_vec[{i}]
         self.gradInput[2][{i}] = self.gradInput[2][{i}]*input[1][{i,2}]*self.DDM_learning_rate
      end

   return self.gradInput
end
