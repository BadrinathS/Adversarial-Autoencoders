import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import numpy as np
import wandb
from torchvision.utils import save_image

wandb.init(job_type = 'train', project = 'Adverserial AutoEncoder')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

bs = 100

train_dataset = datasets.MNIST(root='./mnist_data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./mnist_data', train=False, transform=transforms.ToTensor(), download=False)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size =bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=True)

class Encoder(nn.Module):
    def __init__(self, input_dim, h1_dim, h2_dim, z_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, h1_dim)
        self.fc2 = nn.Linear(h1_dim, h2_dim)
        self.fc3 = nn.Linear(h2_dim, z_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        return self.fc3(x)

class Decoder(nn.Module):
    def __init__(self, z_dim, h1_dim, h2_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(z_dim, h1_dim)
        self.fc2 = nn.Linear(h1_dim, h2_dim)
        self.fc3 = nn.Linear(h2_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        return torch.sigmoid(self.fc3(x))


class Discriminator(nn.Module):
    def __init__(self, z_dim, h1_dim, h2_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(z_dim, h1_dim)
        self.fc2 = nn.Linear(h1_dim, h2_dim)
        self.fc3 = nn.Linear(h2_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return torch.sigmoid(self.fc3(x))


z_dim = 10
encoder = Encoder(784, 100, 200, z_dim).to(device)
decoder = Decoder(z_dim, 200, 100, 784).to(device)
discriminator = Discriminator(z_dim, 100, 50).to(device)

lr = 0.001

optim_encoder = optim.Adam(encoder.parameters(), lr=lr)
optim_decoder = optim.Adam(decoder.parameters(), lr=lr)
optim_discriminator = optim.Adam(discriminator.parameters(), lr=lr/2)
optim_generator = optim.Adam(encoder.parameters(), lr=lr/2)

criterion_auto_encoder = nn.BCELoss()

for epoch in range(100):
    
    autoencoder_loss_epoch = 0
    generator_loss_epoch = 0
    discriminator_loss_epoch = 0

    for batch_id, (image, label) in enumerate(train_loader):

        #Auto Encoder Loss

        image = image.view(-1,784).to(device)
        z = encoder(image)
        recon_image = decoder(z)

        recon_loss = F.binary_cross_entropy(recon_image, image, reduction='sum')
        autoencoder_loss_epoch += recon_loss.item()

        recon_loss.backward()
        optim_encoder.step()
        optim_decoder.step()

        #Discriminator Loss
        encoder.eval()

        z_real = torch.randn(bs,z_dim).to(device)
        discriminator_real = discriminator(z_real)

        z_fake = encoder(image)
        discriminator_fake = discriminator(z_fake)

        discriminator_loss = -torch.mean(torch.log(discriminator_real + 1e-6) + torch.log(1 - discriminator_fake + 1e-6))
        discriminator_loss_epoch += discriminator_loss.item()

        discriminator_loss.backward()
        optim_discriminator.step()

        #Generator Loss

        encoder.train()
        z_fake = encoder(image)
        discriminator_fake = discriminator(z_fake)

        generator_loss = -torch.mean(torch.log(discriminator_fake + 1e-6))
        generator_loss_epoch += generator_loss.item()

        generator_loss.backward()
        optim_generator.step()

    print('Epoch {} \n Autoencoder Loss {} \t Generator Loss {} \t Discriminator Loss {}'.format(epoch, autoencoder_loss_epoch/len(train_loader.dataset), generator_loss_epoch/len(train_loader.dataset), discriminator_loss_epoch/len(train_loader.dataset)))

    wandb.log({'Autoencoder Loss': autoencoder_loss_epoch/len(train_loader.dataset), 'Generator Loss':generator_loss_epoch/len(train_loader.dataset), 'Discriminator Loss':discriminator_loss_epoch/len(train_loader.dataset)}, step=epoch)



    with torch.no_grad():
        z = torch.randn(bs,z_dim).to(device)
        sample = decoder(z)
        sample = sample.view(bs,1,28,28).to(device)
        wandb.log({"Images": [wandb.Image(sample, caption="Images for epoch: "+str(epoch))]}, step=epoch)
        if epoch % 10 == 0:
            save_image(sample, './images/sample_'+str(epoch)+'.png')

    torch.save(encoder.state_dict(), './ckpt/encoder.pth')
    torch.save(decoder.state_dict(), './ckpt/decoder.pth')
    torch.save(discriminator.state_dict(), './ckpt/discriminator.pth')