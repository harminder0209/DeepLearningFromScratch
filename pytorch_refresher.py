import torch

# random

x = torch.randn(3,requires_grad=True)
print(x)

""" 
requires_grad=True creates the computation grapgh .

when you apply a operation(forward pass) to the x(to calculate the output y) then it will automatically create a function for us .

THis function is used in the back propogation to get the gradient( dy/dx in this case). 

 
"""

y = x+2

print(y) # check the output for grad_fn=<AddBackward0>

z = y*y*2
z= z.mean()
print(z)
z.backward() #dz/dx  -- in the background this is a jacobian product so if the value is scalar then you hav to provide a vector


print(x.grad)


# imp

"""if the varaibale has underscore in the end then it will modify the variable in place   
example
x.requires_grad_(False)
"""


