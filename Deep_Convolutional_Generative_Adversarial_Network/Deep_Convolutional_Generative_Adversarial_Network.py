# Deep Convolutional GANs

# Importing the libraries
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

# To enable GPU in this particular case:
c = torch.rand(20000,20000).cuda()
print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')

"""
This is a simple trick to keep GPU active when the input to the NN is small which doesn't activate GPU usage,
we can simply activate it by storing a big tensor in GPU and once it is active, other functions will use that for their purpose.
Though didn't help after some time.
"""

# Setting some hyperparameters
batchSize = 64 
# We set the size of the batch.
imageSize = 64 
# We set the size of the generated images (64x64).

# Creating the transformations
transform = transforms.Compose([transforms.Scale(imageSize), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]) 
# We create a list of transformations (scaling, tensor conversion, normalization) to apply to the input images.
"""
transform.scale=scaling the image

transform.totensor conversion to tensor

transform.normalize:
The values of mean and std for transform.normalize are not the desired mean and std, but rather the values to subtract and divide by, i.e., the estimated mean and std.
In our example we subtract 0.5 and then divide by 0.5 yielding an image with mean zero and values in range [-1, 1]
Similar to training other neural networks, we normalize GAN inputs to improve convergence of the nets, and giving an equal range of value to all features so to not make some features dominate others because of wide range of value.
Also, generator in most GAN models tend to use tanh function as final activation, which maps output to [-1, 1] value range. Scaling our image data to the same value range will give equal data for discriminator to be trained upon
"""
# Loading the dataset
dataset = dset.CIFAR10(root = './data', download = True, transform = transform) 
# We download the training set in the ./data folder and we apply the previous transformations on each image.
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchSize, shuffle = True, num_workers = 2) 
# We use dataLoader to get the images of the training set batch by batch.

# Defining the weights_init function that takes as input a neural network m and that will initialize all its weights.
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        # Initilizing the weight of any convolution layer. Similarly for batch normalizer
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# Defining the generator
        
class G(nn.Module): 
    # We introduce a class to define the generator.
    # As usually CNN takes input and give results, here in generator we're using reverse technique. From noise to image

    def __init__(self): 
        # We introduce the __init__() function that will define the architecture of the generator.
        super(G, self).__init__() 
        # We inherit from the nn.Module tools.
        self.main = nn.Sequential( 
                # We create a meta module of a neural network that will contain a sequence of modules (convolutions, full connections, etc.).
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias = False), 
            # We start with an inversed convolution (to get noise).
            # 100 -- input
            # 512 -- feature map of output which is used as input in 2nd inverse conv
            # 4 -- 4x4 size of kernel
            # 1 -- 1x1 stride
            # 0 -- padding
            # parameters selected through experimentation.
            nn.BatchNorm2d(512), 
            # We normalize all the features along the dimension of the batch.
            nn.ReLU(True), 
            # We apply a ReLU rectification to break the linearity.
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False), 
            # We add another inversed convolution.
            nn.BatchNorm2d(256), 
            # We normalize again.
            nn.ReLU(True), 
            # We apply another ReLU.
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False), 
            # We add another inversed convolution.
            nn.BatchNorm2d(128), 
            # We normalize again.
            nn.ReLU(True), 
            # We apply another ReLU.
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False), 
            # We add another inversed convolution.
            nn.BatchNorm2d(64), 
            # We normalize again.
            nn.ReLU(True), 
            # We apply another ReLU.
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias = False), 
            # We add another inversed convolution.
            nn.Tanh() 
            # We apply a Tanh rectification to break the linearity and stay between -1 and +1.
        )

    def forward(self, input): 
        # We define the forward function that takes as argument an input that will be fed to the neural network, and that will return the output containing the generated images.
        output = self.main(input) 
        # We forward propagate the signal through the whole neural network of the generator defined by self.main.
        return output 
    # We return the output containing the generated images.

# Creating the generator
netG = G()
# We create the generator object.
netG.apply(weights_init) 
# We initialize all the weights of its neural network.

# Defining the discriminator

class D(nn.Module): 
    # We introduce a class to define the discriminator.

    def __init__(self): 
        # We introduce the __init__() function that will define the architecture of the discriminator.
        super(D, self).__init__() 
        # We inherit from the nn.Module tools.
        self.main = nn.Sequential( 
                # We create a meta module of a neural network that will contain a sequence of modules (convolutions, full connections, etc.).
            nn.Conv2d(3, 64, 4, 2, 1, bias = False), 
            # We start with a convolution.
            # 3 -- dimention of input(dimention of image which is obiously 3 -- channel)
            # 64 -- feature map
            # 4 -- 4x4 kernel
            # 2 -- 2x2 stride
            # 1 -- 1x1 padding
            nn.LeakyReLU(0.2, inplace = True), 
            # We apply a LeakyReLU.
            # get the difference between them using formulae used and graph for clear view.
            nn.Conv2d(64, 128, 4, 2, 1, bias = False), 
            # We add another convolution.
            nn.BatchNorm2d(128), 
            # We normalize all the features along the dimension of the batch.
            nn.LeakyReLU(0.2, inplace = True), 
            # We apply another LeakyReLU.
            nn.Conv2d(128, 256, 4, 2, 1, bias = False), 
            # We add another convolution.
            nn.BatchNorm2d(256), 
            # We normalize again.
            nn.LeakyReLU(0.2, inplace = True), 
            # We apply another LeakyReLU.
            nn.Conv2d(256, 512, 4, 2, 1, bias = False), 
            # We add another convolution.
            nn.BatchNorm2d(512), 
            # We normalize again.
            nn.LeakyReLU(0.2, inplace = True), 
            # We apply another LeakyReLU.
            nn.Conv2d(512, 1, 4, 1, 0, bias = False), 
            # We add another convolution.
            nn.Sigmoid() 
            # We apply a Sigmoid rectification to break the linearity and stay between 0 and 1.
        )

    def forward(self, input): 
        # We define the forward function that takes as argument an input that will be fed to the neural network, and that will return the output which will be a value between 0 and 1.
        output = self.main(input) 
        # We forward propagate the signal through the whole neural network of the discriminator defined by self.main.
        return output.view(-1) 
    # We return the output which will be a value between 0 and 1.
    # Flattening the output using .view(-1)

# Creating the discriminator
netD = D() 
# We create the discriminator object.
netD.apply(weights_init) 
# We initialize all the weights of its neural network.

# Training the DCGANs

criterion = nn.BCELoss() 
# We create a criterion object that will measure the error between the prediction and the target.
optimizerD = optim.Adam(netD.parameters(), lr = 0.0002, betas = (0.5, 0.999)) 
# We create the optimizer object of the discriminator.
optimizerG = optim.Adam(netG.parameters(), lr = 0.0002, betas = (0.5, 0.999)) 
# We create the optimizer object of the generator.

for epoch in range(50): 
    # We iterate over 50 epochs.

    for i, data in enumerate(dataloader, 0): 
        # We iterate over the images of the dataset.
        # 0 is the start point
        
        # 1st Step: Updating the weights of the neural network of the DISCRIMINATOR

        netD.zero_grad() 
        # We initialize to 0 the gradients of the discriminator with respect to the weights.
        
        # Training the discriminator with a real image of the dataset
        real, _ = data 
        # We get a real image of the dataset which will be used to train the discriminator.
        # 2nd value is lables which we're not caring in this case.
        # Right now real is input images in minibatch
        input = Variable(real) 
        # We wrap it in a torch variale of images in minibatch without gradient
        target = Variable(torch.ones(input.size()[0])) 
        # We get the target.
        # As we're now training the discriminator with real images creating torch tensor of ones as output (target)
        output = netD(input) 
        # We forward propagate this real image into the neural network of the discriminator to get the prediction (a value between 0 and 1).
        errD_real = criterion(output, target) 
        # We compute the loss between the predictions (output) and the target (equal to 1).
        
        # Training the discriminator with a fake image generated by the generator
        noise = Variable(torch.randn(input.size()[0], 100, 1, 1)) 
        # We make a random input vector (noise) of the generator.
        # 1st argument -- dimention
        # 2nd -- input size as mentioned in generator's first layer
        # 3rd, 4th -- fake dimentions to the tensor.
        fake = netG(noise) 
        # We forward propagate this random input vector into the neural network of the generator to get some fake generated images.
        target = Variable(torch.zeros(input.size()[0])) 
        # We get the target.
        output = netD(fake.detach()) 
        # Detaching the gradient inorder to make the model perform fast.
        # We forward propagate the fake generated images into the neural network of the discriminator to get the prediction (a value between 0 and 1).
        errD_fake = criterion(output, target) 
        # We compute the loss between the prediction (output) and the target (equal to 0).

        # Backpropagating the total error
        errD = errD_real + errD_fake 
        # We compute the total error of the discriminator.
        errD.backward() 
        # We backpropagate the loss error by computing the gradients of the total error with respect to the weights of the discriminator.
        optimizerD.step() 
        # We apply the optimizer to update the weights according to how much they are responsible for the loss error of the discriminator.

        # 2nd Step: Updating the weights of the neural network of the GENERATOR

        netG.zero_grad() 
        # We initialize to 0 the gradients of the generator with respect to the weights.
        target = Variable(torch.ones(input.size()[0])) 
        # We get the target.
        output = netD(fake) 
        # We're not detaching the gradient of this output as it is required to update the weights to the NN
        # We forward propagate the fake generated images into the neural network of the discriminator to get the prediction (a value between 0 and 1).
        errG = criterion(output, target) 
        # We compute the loss between the prediction (output between 0 and 1) and the target (equal to 1).
        errG.backward() 
        # We backpropagate the loss error by computing the gradients of the total error with respect to the weights of the generator.
        optimizerG.step() 
        # We apply the optimizer to update the weights according to how much they are responsible for the loss error of the generator.
        
        # 3rd Step: Printing the losses and saving the real images and the generated images of the minibatch every 100 steps

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch, 50, i, len(dataloader), errD.data, errG.data)) 
        # We print les losses of the discriminator (Loss_D) and the generator (Loss_G).
        if i % 100 == 0: # Every 100 steps:
            vutils.save_image(real, '%s/real_samples_%03d.png' % ("./results/50Epoch", epoch), normalize = True) 
            # We save the real images of the minibatch.
            fake = netG(noise) 
            # We get our fake generated images.
            vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % ("./results/50Epoch", epoch), normalize = True) 
            # We also save the fake generated images of the minibatch.