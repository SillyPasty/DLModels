from torchvision import datasets, transforms
import torch

def get_data():
    # transform = transforms.Compose([transforms.ToTensor(),
    #                                 transforms.Normalize(mean = (0.5,), std = (0.5,))])
    data_train = datasets.MNIST(root = r"D:\Programme\Python\DL\Pytorch\dataset", 
                                transform=transforms.ToTensor(), 
                                train=True,
                                download=True)
    
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=4, shuffle=True)
    
    data_test = datasets.MNIST(root = r"D:\Programme\Python\DL\Pytorch\dataset",
                               transform = transforms.ToTensor(),
                               train = False,
                               download=True)

    test_loader = torch.utils.data.DataLoader(data_test, batch_size=4, shuffle=True)

    return train_loader, test_loader