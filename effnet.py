from utils import *

mean = torch.Tensor([51.7598, 57.7707, 64.2816])
std = torch.Tensor([30.0926, 30.4241, 31.7076])

augmentations = Compose(
    [ConvertImageDtype(torch.float),
     RandomApply(torch.nn.ModuleList([RandomAffine(degrees = (-10, 10))]),p=0.3),
     RandomApply(torch.nn.ModuleList([GaussianBlur(kernel_size = (3, 7))]),p=0.4),
     Normalize(mean, std),
     Resize((384, 380)),
    ])


train_set = CustomImageDataset('/content/train.csv','/content/train_images',transform = augmentations)
val_set = CustomImageDataset('/content/train.csv','/content/train_images',transform =
                             transforms.Compose([transforms.ConvertImageDtype(torch.float),
                                                 torchvision.transforms.Normalize(mean,std),
                                                 torchvision.transforms.Resize((384, 380))
                                                ]))

traindata = Subset(train_set, indices=train_idx)
valdata = Subset(val_set, indices=valid_idx)
dataset_sizes = {'train':40826,'val':10207}


dataloaders = {'train':DataLoader(traindata,batch_size=128,num_workers=4),
               'val':DataLoader(valdata,batch_size=128,num_workers=4)}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_size = 0.8
num_train = len(train_set)
indices = list(range(num_train))
split = int(np.floor(train_size * num_train))
np.random.shuffle(indices)
train_idx, valid_idx = indices[:split], indices[split:]


val_losses = []
val_accs = []
train_losses = []
train_accs = []

model = models.efficientnet_b3(pretrained=True)
model.classifier[1] = nn.Linear(1536, 28)

for layer in model.parameters():
    layer.requires_grad = False

model.classifier[1].weight.requires_grad = True
model.classifier[1].bias.requires_grad = True

model.classifier[1].apply(weights_init_normal)

model.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0000)
#optimizer = optim.AdamW(model.parameters(), lr=0.0015, weight_decay=0.001)

exp_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)

warm_up_iter = 3
lr_min = 1e-3
lr_max = 1e-2
T_max = 10

lambda_0 = lambda curr_iter: curr_iter/warm_up_iter if curr_iter < warm_up_iter else \
             (lr_min + 0.5*(lr_max-lr_min)*(1.0+math.cos( (curr_iter - warm_up_iter)/(T_max - warm_up_iter)*math.pi)))/0.1
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda_0)

model_trained = train_model(model, criterion, optimizer,
                         scheduler, num_epochs=20)