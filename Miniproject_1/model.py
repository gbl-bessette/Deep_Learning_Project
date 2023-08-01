import torch
from pathlib import Path
import torch.nn.functional as F

def initialize_weights(m):
  if isinstance(m, torch.nn.Conv2d or torch.nn.ConvTranspose2d):
      torch.nn.init.xavier_normal_(m.weight.data,gain=torch.nn.init.calculate_gain('relu', param=None))
      torch.nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, torch.nn.BatchNorm2d):
      torch.nn.init.constant_(m.weight.data, 1)
      torch.nn.init.constant_(m.bias.data, 0)


class Network(torch.nn.Module):


    def __init__(self):

        super().__init__()

        # encoder
        self.conv1 = torch.nn.Conv2d(3, 25,    kernel_size=3, stride=1, padding=0)    # 3x32x32 to dim1x32x32
        self.conv2 = torch.nn.Conv2d(25, 50,   kernel_size=3, stride=1, padding=0)  # dim1x32x32 to dim2x32x32
        
        # ??
        self.convint = torch.nn.Conv2d(50,50,kernel_size=3, stride=1, padding=1)     # dim2x32x32 to dim2x32x32

        # decoder

        self.conv2T = torch.nn.ConvTranspose2d(50,25,kernel_size=3, stride=1, padding=0)     # dim2x32x32 to dim1x32x32
        self.conv1T = torch.nn.ConvTranspose2d(25,3,   kernel_size=3, stride=1, padding=0)     #  dim1x32x32 to 3x32x32
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,x):
        # encoding
        y = x
        y = F.relu(self.conv1(y))

        y = F.relu(self.conv2(y))

        y = F.relu(self.convint(y))

        # decoding
        y = F.relu(self.conv2T(y))

        y = self.sigmoid(self.conv1T(y))
        return y

class Model():

    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.mini_batch_size = 5
        self.eta = 0.001
        self.criterion = torch.nn.MSELoss().to(self.device)
        self.model = Network().to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.eta)

    def load_pretrained_model(self):
        model_path = Path(__file__).parent / "bestmodel.pth"
        # self.model = torch.load(model_path,map_location=self.device)
        md = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(md['model_state_dict'])
        self.model.eval()
    
    def train(self, train_input, train_target, num_epochs):
        # Initializing the weights
        self.model.apply(initialize_weights)
        # Split the input in batches
        train_input = (train_input/255).to(self.device)
        train_target = (train_target/255).to(self.device)
        split_input = train_input.split(self.mini_batch_size)
        split_target = train_target.split(self.mini_batch_size)
        # Train the model
        for epoch in range(num_epochs):
            acc_loss = 0
            for id,input in enumerate(split_input):
                output = self.model(input)
                loss = self.criterion(output, split_target[id])
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                acc_loss += loss.item()
            # print("it.", epoch+1, "acc_loss", acc_loss)
        
    def predict(self, test_input):
        test_input = test_input.to(self.device)
        test_input = test_input/255
        test_output = self.model.forward(test_input)
        test_output = test_output*255
        return test_output