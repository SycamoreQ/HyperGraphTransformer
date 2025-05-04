import torch 
from torch_geometric.data import Data, Batch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn 
from torch import optim
from models import Model_2
from train_eval import train_model , evaluate_model
from torch.utils.data import SubsetRandomSampler



def create_graph_data(features, labels=None):
    num_nodes = features.size(0)
    edge_index = torch.combinations(torch.arange(num_nodes), r=2).t()
    x = features.clone().detach()
    y = labels.clone().detach() if labels is not None else None
    return Data(x=x, edge_index=edge_index, y=y)

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2), 
    transforms.RandomRotation(30),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

datadir = 'C:/Users/admin/Desktop/DL_SEM_4_part2/DL_sem_4/brain_tumor_dataset'
train_dataset = ImageFolder(datadir, transform=transform_train)
val_dataset = ImageFolder(datadir, transform=transform_val)

from sklearn.model_selection import train_test_split

indices = list(range(len(train_dataset)))
labels = [train_dataset[i][1] for i in indices]


train_indices, temp_indices = train_test_split(
    indices, test_size=0.3, stratify=labels, random_state=42
)
val_indices, test_indices = train_test_split(
    temp_indices, test_size=0.5, stratify=[labels[i] for i in temp_indices], random_state=42
)
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = DataLoader(train_dataset, batch_size=2, sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=2, sampler=val_sampler)
test_loader = DataLoader(val_dataset, batch_size=32, sampler=test_sampler)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_3 = Model_2(num_classes=2)


criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(model_3.parameters(), lr=0.001, weight_decay=0.01)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=True
)


model, history = train_model(
    model=model_3,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=25,
    device=device,
    resume=False,  # Don't resume training
    checkpoint_dir='checkpoints',
    checkpoint_interval=2  # Save checkpoint every 2 epochs
)




    

