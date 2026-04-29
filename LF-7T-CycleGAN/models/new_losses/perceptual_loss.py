"""

Implementation of the perceptual loss component, as described by Yang et al. in: https://doi.org/10.1109/TMI.2025.3597401

"""


import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch




class PerceptualLoss(nn.Module):
    """
    Class to manage computing style and content perceptual losses

    """

    def __init__(self, device):

        super(PerceptualLoss, self).__init__()

        # Import the pre-trained ResNet34 Model
        resNet34 = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        # Extract layers
        self.resNet34conv1 = resNet34.conv1
        self.resNet34bn1 = resNet34.bn1
        self.resNet34relu = resNet34.relu
        self.resNet34maxpool = resNet34.maxpool

        # Get only feature extraction layers
        self.resNet34Layer1 = resNet34.layer1
        self.resNet34Layer2 = resNet34.layer2
        self.resNet34Layer3 = resNet34.layer3
        self.resNet34Layer4 = resNet34.layer4

        # Set the parameters in ResNet34 to not need gradients
        for param in self.parameters():
            param.requires_grad = False

        self.eval()

        # Send to GPU
        self.to(device)

    def _prepare_input(self, x):
        """
        Enforces input to 3-channel RGB format for ResNet34

        """

        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        return x


    def extract_features(self, x):
        """
        Performs feature extraction by passing input through ResNet34

        """

        # Ensure input has correct channels
        x = self._prepare_input(x)

        # Extract features by passing through ResNet34

        features = []

        x = self.resNet34conv1(x)
        x = self.resNet34bn1(x)
        x = self.resNet34relu(x)
        x = self.resNet34maxpool(x)

        x = self.resNet34Layer1(x)
        features.append(x)
        
        x = self.resNet34Layer2(x)
        features.append(x)
        
        x = self.resNet34Layer3(x)
        features.append(x)
        
        x = self.resNet34Layer4(x)
        features.append(x)
        
        return features
    

    def gram_matrix(self, features):
        """
        Compute the gram matrix of the features for comparing style

        """

        B, C, H, W  = features.shape
        
        # Re-shape features
        features_reshaped = features.view(B, C, H * W)
        
        # Calculate gram matrix
        gram_matrix = torch.bmm(features_reshaped, features_reshaped.transpose(1, 2))
        
        # Normalise to account for feature map size
        gram_matrix = gram_matrix / (H * W)
        
        return gram_matrix


    def loss_per_C(self, fake_B, fake_A, real_A, real_B):
        """
        Returns the content percepptual loss for the given inputs

        Args:
            fake_B: The generated 7T image
            fake_A: The generated 0.3T image
            real_A: The real 0.3T image
            real_B: The real 7T image

        Returns:
            Content perceptual loss value


        """

        # Extract features for each image
        fake_B_features = self.extract_features(fake_B)
        real_A_features  = self.extract_features(real_A)

        fake_A_features  = self.extract_features(fake_A)
        real_B_features = self.extract_features(real_B)
            
        # Apply equation as per Yang et al.
        return (
            F.mse_loss(fake_B_features[3], real_A_features[3]) +  F.mse_loss(fake_A_features[3], real_B_features[3])
        )


    def loss_per_S(self, fake_B, fake_A, real_A, real_B):
        """
        Returns the style percepptual loss for the given inputs

        Args:
            fake_B: The generated 7T image
            fake_A: The generated 0.3T image
            real_A: The real 0.3T image
            real_B: The real 7T image

        Returns:
            Style perceptual loss value


        """
        
        # Extract features for each image
        fake_B_features = self.extract_features(fake_B)
        real_B_features = self.extract_features(real_B)
        fake_A_features = self.extract_features(fake_A)
        real_A_features = self.extract_features(real_A)
        

        # Apply the equatin from the reference paper
        
        loss = 0

        for i in range(4):
            
            loss += F.mse_loss(
                self.gram_matrix(fake_B_features[i]),
                self.gram_matrix(real_B_features[i])
            )

            loss += F.mse_loss(
                self.gram_matrix(fake_A_features[i]),
                self.gram_matrix(real_A_features[i])
            )

        return loss


