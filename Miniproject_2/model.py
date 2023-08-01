from torch import empty , cat , arange
from torch.nn.functional import fold, unfold
from pathlib import Path

### THIS CODE INCLUDES ALL NECESSARY CLASSES FOR MINIPROJECT2
### EACH CLASS INCLUDES A FORWARD, BACKWARD AND A PARAM METHOD (EXCEPT FOR SGD)
### THE FOLLOWING CLASSES ARE LISTED AS FOLLOWS: Conv2d, NearestUpsampling, ReLU, Sigmoid, MSE, SGD , Sequential, and Model

### THIS CODE ONLY WORKS FOR Conv2d WITH A STRIDE EQUAL TO 1 OR 2 AND A FIXED KERNEL SIZE OF SIZE (3,3)

class Conv2d():
    # Conv2d has as input the number of channels of the input (int), the number of channels of the output (int), the kernel size (int), the padding (int), the stride (int).
    # The weights and biases are initialized with a normal distribution (just like in standard Pytorch initialization for convolutions)
    # The gradients of the weights and biases are initialized to 0
    def __init__(self,c_in,c_out,k_sz,padding=0,stride=1):                                              
        
        self.ci = c_in                      
        self.co = c_out                     
        self.ksz = k_sz                     
        self.pad = padding                 
        self.strid = stride                 
        k = 1/(self.ci*self.co*self.ksz**2)
        self.weight, self.bias = empty([self.co,self.ci,self.ksz,self.ksz]).uniform_(-k**(1/2),k**(1/2)), empty(self.co).uniform_(-k**(1/2),k**(1/2))
        self.dw = empty(1).fill_(0)
        self.db = empty(1).fill_(0)
    
    # This function is used to compute the convolution operation for the forward pass
    # It takes as input the tensors of inputs y, of weights and biases w,b and has additional inputs for the padding, the stride and the dilatation
    # It returns the tensor resulting from the convolution operation with the right output dimension
    def c2d(self,y,w,b,pad = 0, strid=1, dilat = 1):
        
        if b is None:
            b = empty(w.size(0)).fill(0)

        ksz = w.size(-1)
        yw = int((y.shape[-1] + 2*pad - dilat*(ksz-1) -1)/strid + 1)
        yh = int(((y.shape[-2] + 2*pad - dilat*(ksz-1) -1)/strid + 1))
        
        y = unfold(y,kernel_size=ksz,padding = pad, stride=strid, dilation = dilat)
        y = w.view(w.size(0),-1) @ y + b.view(1 , -1 , 1)
        y = fold(y,output_size=(yh, yw),kernel_size=1)
        if dilat == 1:
            return y
        else:
            return y[:,:,:-1,:-1]

    # This function is used to compute the convolution operation for the backward pass: to compute dL/dw
    # It takes as input the tensors of inputs y, of weights and biases w,b and has additional inputs for the padding and the dilatation
    # It returns the tensor resulting from the convolution operation with the right output dimension
    def c2dt(self,y,w,b,pad = 0, strid=1, dilat = 1):
        ksz = w.size(-1)
        yw = int((y.shape[-1] + 2*pad - dilat*(ksz-1) -1)/strid + 1)
        yh = int(((y.shape[-2] + 2*pad - dilat*(ksz-1) -1)/strid + 1))

        y = unfold(y, kernel_size=ksz, padding = pad, stride=strid, dilation = dilat)
        
        y = w.view(w.size(0),-1).contiguous().view(-1) @ y.view(y.size(1)*y.size(0),-1)
        
        y = y.view(1,y.size(-1))
        
        y = fold(y,output_size=(yh, yw),kernel_size=1)
        
        y = y.view(1,1,y.size(-2),y.size(-1))
        if dilat == 1:
            return y
        else:
            return y[:,:,:-1,:-1]

    # This function is used to compute the convolution operation for the backward pass: to compute dL/dx^(l-1)
    # It takes as input the tensors of inputs y, of weights and biases w,b and has additional inputs for the padding and the dilatation
    # It returns the tensor resulting from the convolution operation with the right output dimension
    def c2d_x(self,y,w,b,pad = 0, strid=1, dilat = 1, padding=0):
        ksz = w.size(-1)
        yw = int((y.shape[-1] + 2*pad - dilat*(ksz-1) -1)/strid + 1)
        yh = int(((y.shape[-2] + 2*pad - dilat*(ksz-1) -1)/strid + 1))
        y = unfold(y,kernel_size=ksz,padding = pad, stride=strid, dilation = dilat)
        y = w.view(w.size(0),-1) @ y 
        y = fold(y,output_size=(yh, yw),kernel_size=1)
        if dilat == 1:
            if padding ==1:
                return y[:,:,1:-1,1:-1]
            else: 
                return y
        else:
            return y[:,:,1:,1:]

    # The forward function uses c2d to compute the forward pass and returns the convuluted output tensor
    def forward(self,x):
        self.x = x
        y = x
        y = self.c2d(y,self.weight,self.bias,pad = self.pad,strid = self.strid)
        return y

    # The backward pass uses c2dt and c2dx to compute the gradients of the loss wrt to the weights w, the biases b and the inputs x
    # The gradients wrt w and b are stored in self variables while the gradient wrt x are returned by the function
    def backward(self,dl):
        
        dw = empty(self.weight.shape)
  
        dx = empty([self.x.size(0),self.co,self.x.size(1),self.x.size(2),self.x.size(3)])

        db = empty(self.bias.shape)


        for i in range(self.co):
            for j in range(self.ci): 
                dw[i:i+1,j:j+1,:,:] = self.c2dt(self.x[:,j:j+1,:,:],dl[:,i:i+1,:,:],None,pad = self.pad, dilat=self.strid)
                dx[:,i,j,:,:] = self.c2d_x(self.weight[i:i+1,j:j+1,:,:],dl[:,i:i+1,:,:].flip([2,3]),None, pad = (dl.shape[-1]-1)*self.strid, dilat=self.strid, padding=self.pad) 
        dx = dx.sum(1)
        for i in range(self.co):
            db[i] = dl[:,i:i+1,:,:].sum() 
        self.dw = dw
        self.db = db
        return dx

    # param creates a list of all weights and biases with dl/dw and dl/db respectively
    def param(self):
        return [(self.weight, self.dw),(self.bias, self.db)]


# This class takes as input a scale factor and upsamples an input x accordingly in its forward method.
# The backward method computes the gradients of the loss wrt to the input by summin these gradients in order to downsample
# The weights and biases and respective gradients are initialized at 0 and will stay at 0 throughout training
class NearestUpsampling():
    def __init__(self,scale_factor):
        
        self.sf = scale_factor
        self.weight = empty(1).fill_(0)
        self.bias = empty(1).fill_(0)
        self.dw = empty(1).fill_(0)
        self.db = empty(1).fill_(0)

    def forward(self,x):
        
        self.x=x
        upmat = empty([1,1,x.size(-2),self.sf,x.size(-1),self.sf]).fill_(1)
        upmat = upmat
        y = upmat*x.view(x.size(0),x.size(1),x.size(-2),1,x.size(-1),1)
        y = y.view(x.size(0),x.size(1),x.size(-2)*self.sf,x.size(-1)*self.sf)
        return y

    def backward(self,dl):
        # dx = dl[:,:,1::self.sf,1::self.sf]
        dx = dl.view(self.x.size(0),self.x.size(1),self.x.size(2),self.sf,self.x.size(3),self.sf)
        dx = dx.sum(dim=(3,5))
        return dx
    
    def param(self):
        return []

# This class includes the standard relu function in its forward method and its derivative in its backward method
# The weights and biases and respective gradients are initialized at 0 and will stay at 0 throughout training
class ReLU():
    def __init__(self):
        
        self.weight = empty(1).fill_(0)
        self.bias = empty(1).fill_(0)
        self.dw = empty(1).fill_(0)
        self.db = empty(1).fill_(0)

    def forward(self,x):
        self.x = x
        y = x
        y[y<0]=0
        return y

    def backward(self, dl):
        y=self.x
        y[y<0]=0
        y[y>0]=1
        y = y * dl
        return y

    def param(self):
        return []

# This class includes the sigmoid function in its forward method and its derivative in its backward method
# The weights and biases and respective gradients are initialized at 0 and will stay at 0 throughout training
class Sigmoid():
    def __init__(self):
        
        self.weight = empty(1).fill_(0)
        self.bias = empty(1).fill_(0)
        self.dw = empty(1).fill_(0)
        self.db = empty(1).fill_(0)

    def forward(self,x):
        self.x=x
        y = 1/(1+(-x).exp())
        return y
    
    def backward(self,dl):
        y = (-self.x).exp()/((1+(-self.x).exp())**2)
        y = y * dl
        return y

    def param(self):
        return []



# This class includes the MSE function in its forward method and its derivative in its backward method
# The user can choose between "mean" and "sum" just like in Pytorch
# The weights and biases and respective gradients are initialized at 0 and will stay at 0 throughout training
class MSE():
    def __init__(self, string='mean'):
        
        self.str = string
    
    def forward(self, out, tgt):
        self.out = out
        self.tgt = tgt
        y = (out-tgt)**2
        nb = 1
        if self.str == 'mean':
            for i in range(len(tgt.shape)):
                nb = nb*tgt.shape[i] 
        elif self.str =='sum':
            nb = tgt.shape[0]
        y = y.sum()
        y = 1/nb*y
        return y

    def backward(self):
        DL = 2*(self.out-self.tgt)
        nb = 1
        if self.str == 'mean':
            for i in range(len(self.tgt.shape)):
                nb = nb*self.tgt.shape[i] 
        elif self.str =='sum':
            nb = self.tgt.shape[0]
        DL = 1/nb*DL
        return DL


# This class updates the weights and biases of the convolution layers using the Stochastic Gradient Descent method
# It takes as input a learning rate eta and the sequence of the blocks in the network defined by sequential
class SGD():
    def __init__(self, seq, lr):
        self.seq = seq
        self.eta = lr

    def step(self):
        for arg in self.seq.arg:
            arg.weight = arg.weight - self.eta*arg.dw
            arg.bias = arg.bias - self.eta*arg.db

# This class gathers the modules in a sequence
# Its forward method calls the forward methods of all layers in the network to compute the forward pass on the input
# Its backward method calls the backward methods of all layers in the network to compute the gradients of the loss wrt w,b,x^(l-1)
# Its param method gathers all the network parameters and gradients wrt w and b in a list

class Sequential():
    def __init__(self, *args):
        
        self.arg = args
        

    def forward(self, img):
        y = img
        for arg in self.arg:
            y = arg.forward(y)
        return y
    
    def backward(self, DL):
        dl = DL
        for index_arg, arg in reversed(list(enumerate(self.arg))):
            dl = arg.backward(dl)
        return dl


    def param(self):
        l = []
        for arg in self.arg:
            for param in arg.param:
                l.append(param)
        return l
    

# This is the Model class suggested by the assignment
class Model():

    def __init__(self):
        
        self.mini_batch_size = 5
        self.eta = 0.5
        self.model = Sequential(Conv2d(3,16,3,padding = 1, stride=2), ReLU(), Conv2d(16,32,3,padding = 1, stride=2), ReLU(), NearestUpsampling(scale_factor=2), Conv2d(32,16,3,padding = 1, stride=1), ReLU(), NearestUpsampling(scale_factor=2), Conv2d(16,3,3,padding = 1, stride=1), Sigmoid())
        self.criterion = MSE()
        self.optim = SGD(self.model, lr=self.eta)
        pass

    def load_pretrained_model(self):
        import pickle
        model_path = Path(__file__).parent / "bestmodel.pth"
        with open(model_path, 'rb', ) as f:
            x= pickle.load(f)
        for empty,trained in zip(self.model.arg,x):
            empty.weight = trained[0]
            empty.bias = trained[1]

    def train(self, train_input, train_target, num_epochs):
        train_input = train_input
        train_target = train_target
        train_input = train_input/255
        train_target = train_target/255
        split_input = train_input.split(self.mini_batch_size)
        split_target = train_target.split(self.mini_batch_size)
        for epoch in range(num_epochs):
            for id,input in enumerate(split_input):
                output = self.model.forward(input)
                loss = self.criterion.forward(output, split_target[id])
                self.model.backward(self.criterion.backward())
                self.optim.step()

    def predict(self, test_input):
        test_input = test_input
        test_input = test_input/255
        test_output = self.model.forward(test_input)
        test_output = test_output*255
        return test_output

