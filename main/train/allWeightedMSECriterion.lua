local allWeightedMSECriterion, parent = torch.class('nn.allWeightedMSECriterion','nn.MSECriterion')

function allWeightedMSECriterion:__init(w)
   parent.__init(self)
   self.weight = w:clone()
end

function allWeightedMSECriterion:updateOutput(input,target)

   self.buffer1 = self.buffer1 or input.new()
   self.buffer1:resizeAs(input):copy(input)
   if input:dim() - 1 == self.weight:dim() then
      for i=1,input:size(1) do
         self.buffer1[i]:cmul(self.weight)
      end
   else
      self.buffer1:cmul(self.weight)
   end

   self.buffer2 = self.buffer2 or input.new()
   self.buffer2:resizeAs(input):copy(target)
   if input:dim() - 1 == self.weight:dim() then
      for i=1,input:size(1) do
         self.buffer2[i]:cmul(self.weight)
      end
   else
      self.buffer2:cmul(self.weight)
   end

   self.output_tensor = self.output_tensor or input.new(1)
   input.THNN.MSECriterion_updateOutput(
      self.buffer1:cdata(),
      self.buffer2:cdata(),
      self.output_tensor:cdata(),
      self.sizeAverage
   )
   self.output = self.output_tensor[1]
   return self.output
end

function allWeightedMSECriterion:updateGradInput(input, target)
   self.buffer1 = self.buffer1 or input.new()
   self.buffer1:resizeAs(input):copy(input)
   if input:dim() - 1 == self.weight:dim() then
      for i=1,input:size(1) do
         self.buffer1[i]:cmul(self.weight)
      end
   else
      self.buffer1:cmul(self.weight)
   end

   self.buffer2 = self.buffer2 or input.new()
   self.buffer2:resizeAs(input):copy(target)
   if input:dim() - 1 == self.weight:dim() then
      for i=1,input:size(1) do
         self.buffer2[i]:cmul(self.weight)
      end
   else
      self.buffer2:cmul(self.weight)
   end

   input.THNN.MSECriterion_updateGradInput(
      self.buffer1:cdata(),
      self.buffer2:cdata(),
      self.gradInput:cdata(),
      self.sizeAverage
   )
   return self.gradInput
end