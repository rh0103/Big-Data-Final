import torch
from tqdm.auto import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

"""
Implementation of Autoencoder
"""
class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, encoding_dim: int) -> None:
        """
        Modify the model architecture here for comparison
        """

        
        super(Autoencoder, self).__init__()
        # 
        # 
        # # 原式，記得改回來
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            # nn.Linear(encoding_dim, encoding_dim//2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            # nn.Linear(encoding_dim//2, encoding_dim),
            nn.Linear(encoding_dim, input_dim),
        )

        #淺層
        # self.encoder = nn.Sequential(
        #     nn.Linear(input_dim, encoding_dim),
        #     nn.ReLU()
        # )
        # self.decoder = nn.Sequential(
        #     nn.Linear(encoding_dim, input_dim)
        # )

        #胖層
        # self.encoder = nn.Sequential(
        #     nn.Linear(input_dim, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, encoding_dim),
        # )

        # self.decoder = nn.Sequential(
        #     nn.Linear(encoding_dim, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, input_dim),
        # )
    
    def forward(self, x):
        #TODO: 5%
        zipper = self.encoder(x)
        decoded = self.decoder(zipper)
        return decoded
        # raise NotImplementedError
    
    def fit(self, X, epochs=10, batch_size=32):
        #TODO: 5%
        model = self
        model.train()


        Xtensor = torch.tensor(X, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(Xtensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        datalen = len(dataloader)

        losslist = list()
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        mseloss = nn.MSELoss()

        for epochs in range(epochs):
            totalloss = 0.0
            for batch in dataloader:
                Xb = batch[0]

                xnhat = model(Xb)
                error = mseloss(Xb, xnhat)

                # backward and optimize pp
                for param in self.parameters():
                    if param.grad is not None:
                        param.grad.zero_()
                error.backward()
                optimizer.step()
                totalloss += error.item()
            
            averror = totalloss / datalen

            losslist += [averror]
            print(f'Epoch {epochs+1}/{epochs}, Loss: {error.item()}')

        # # happy draw time!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # plt.plot(losslist)
        # plt.ylabel('averaged squared error')
        # plt.title('Autoencoder Loss Curve')
        # plt.xlabel('epoch')
        # plt.show()
        





        # raise NotImplementedError
    
    def transform(self, X):
        #TODO: 2%
        
        tenX = torch.tensor(X, dtype=torch.float32)
        self.encoder.eval()
        with torch.no_grad():
            zipped = self.encoder(tenX)
        return zipped.numpy()
        # raise NotImplementedError
    
    def reconstruct(self, X):
        #TODO: 2%
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            zipped = self.encoder(X)
            decoded = self.decoder(zipped)
        return decoded.numpy()
        # raise NotImplementedError


"""
Implementation of DenoisingAutoencoder
"""
class DenoisingAutoencoder(Autoencoder):
    def __init__(self, input_dim, encoding_dim, noise_factor=0.2):
        super(DenoisingAutoencoder, self).__init__(input_dim,encoding_dim)
        self.noise_factor = noise_factor
    
    def add_noise(self, x):
        #TODO: 3%
        noise = torch.randn_like(x) * self.noise_factor
        x_noisy = x + noise
        return x_noisy
        # raise NotImplementedError
    
    def fit(self, X, epochs=10, batch_size=32):
        #TODO: 4%
        


        Xtensor = torch.tensor(X, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(Xtensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        datalen = len(dataloader)
        
        self.train()

        losslist = list()
        
        # optimizer = optim.Adam(model.parameters(), lr=0.001)
        # mseloss = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        # self.optimizer  = optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        self.criterion = nn.MSELoss()

        for epochs in range(epochs):
            totalloss = 0.0
            for batch in dataloader:
                Xb = batch[0]
                
                xnoisse = self.add_noise(Xb)

                self.optimizer.zero_grad()

                x_hat = self(xnoisse)
                error = self.criterion(x_hat, Xb) 

                error.backward()
                self.optimizer.step()
                totalloss += error.item()
            
            averror = totalloss / datalen

            losslist += [averror]
            print(f'Epoch {epochs+1}/{epochs}, Loss: {error.item()}')

        # # happy draw time!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # plt.plot(losslist)
        # plt.ylabel('loss')
        # plt.title('denoisingAutoencoder Loss Curve')
        # plt.xlabel('epoch')
        # plt.show()
        # # raise NotImplementedError

