from torchvision.models import squeezenet1_0
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from CustomSqueezenet import CustomSqueezeNet
from DataLoader_Functions import Rescale,ToTensor,WeldDefectXRayDataSet
from torch.utils.data import DataLoader
from torchvision import transforms
import tqdm
from tqdm import trange
# net=squeezenet1_0()
# state_dict=torch.load(r'squeezenet1_0-b66bff10.pth')
# net.load_state_dict(state_dict, strict=True)
# print(net)
# for name, child in net.named_children():
#         for x, y in child.named_children():
#             print(name,x)


#Define the Model 
def train_model(net, criterion, optimizer, trainloader, validloader, num_epochs=2, device=None):
    train_losses = []
    val_accuracies = []

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = net.to(device)

    for epoch in trange(num_epochs, desc='Epoch'):
        running_loss = 0.0
        train_batches = tqdm(trainloader, desc='Training', leave=False)

        for i, data in enumerate(train_batches):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_batches.set_postfix(loss=running_loss / (i + 1))

        train_losses.append(running_loss / len(trainloader))

        net.eval()
        correct = 0
        total = 0
        val_batches = tqdm(validloader, desc='Validation', leave=False)
        with torch.no_grad():
            for data in val_batches:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_accuracy = 100 * correct / total
                val_batches.set_postfix(accuracy=val_accuracy)

        val_accuracies.append(val_accuracy)

    print('Finished Training')

    # Plot training and validation accuracies
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_epochs), train_losses, '-b', label='Training Loss')
    plt.plot(range(num_epochs), val_accuracies, '-r', label='Validation Accuracy')
    plt.title('Training and Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Loss/Accuracy')
    plt.legend()
    plt.show()

    return train_losses, val_accuracies


transform = transforms.Compose([
        Rescale(224),
        ToTensor()
])

batch_size=32

trainset=WeldDefectXRayDataSet(csv_file='Processed_train.csv',
                              img_root_dir=r'Combined_RIAWELC_Dataset\training',
                              transform=transform)

trainloader=DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=0)

valset=WeldDefectXRayDataSet(csv_file='Processed_val.csv',
                              img_root_dir=r'Combined_RIAWELC_Dataset\validation',
                              transform=transform)

valloader=DataLoader(valset,batch_size=batch_size,shuffle=True,num_workers=0)


net=CustomSqueezeNet(num_classes=4)

for p in net.classifier.parameters():
		p.requires_grad = True
for p in net.features.parameters():
      p.requires_grad= False


num_epochs=1

learning_rate=0.001
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(net.classifier.parameters(),learning_rate)

# train_loss , val_acc= train_model(net, loss_fn, optimizer,trainloader,valloader , num_epochs , device=None)
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        print('This is i: ',i)
        print('This is inputs: ',inputs)
        print('This is labels: ',labels)
        # zero the parameter gradients
        # optimizer.zero_grad()

        # # forward + backward + optimize
        # outputs = net(inputs)
        # loss = criterion(outputs, labels)
        # loss.backward()
        # optimizer.step()

        # # print statistics
        # running_loss += loss.item()
        # if i % 2000 == 1999:    # print every 2000 mini-batches
        #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
        #     running_loss = 0.0

print('Finished Training')




print(net)
print("Model's state_dict:")
# for param_tensor in net.state_dict():
#     print(param_tensor, "\t", net.state_dict()[param_tensor].size())

print('\n')
n=3
print(f'Weights for a {n} layer')
# net.classifier[1]=nn.Conv2d(512,4,kernel_size=(1,1),stride=(1,1))
# net.classifier[3]=nn.
# print(type(net.classifier))

'''Add a softmax layer at the end of squeezenet model by navigating to the model definition of squeezenet'''
 
# print(net)

# print("Model's state_dict:")
# for param_tensor in net.state_dict():
#     print(param_tensor, "\t", net.state_dict()[param_tensor].size())





# trainable_params = [p for p in net.parameters() if p.requires_grad==True]
print('This is trainable params')
# print(trainable_params)



