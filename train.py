import glob
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch import nn
import os
from tqdm import tqdm
from PIL import Image
from torchvision import models, transforms
from model import MyModel
from torch.utils.tensorboard import SummaryWriter
from dataset import Dataset
import argparse


def predict(model, device, image_path):
    model.eval()
    val_acc_num = 0
    images_lenth = 0
    classes_dir = os.listdir(image_path)
    for label in classes_dir:
        images = []
        label_path = os.path.join(image_path, label)
        if os.path.isdir(label_path):
            images_png = glob.glob(os.path.join(label_path, "*.{}".format('png')))
            images_jpg = glob.glob(os.path.join(label_path, "*.{}".format('jpg')))
            images.extend(images_png)
            images.extend(images_jpg)
            images_lenth += len(images)
            for img in images:
                image = Image.open(img)
                image = image.convert("RGB")
                image = valid_transform(image)
                image = image.unsqueeze_(0).to(device)
                with torch.no_grad():
                    outputs = model(image)
                    outputs = outputs.to('cpu')
                predict_label = torch.max(outputs, dim=1)[1].data.numpy()[0]
                if predict_label == int(label):
                    val_acc_num += 1
    val_acc = val_acc_num / images_lenth
    return val_acc

def train(model, loss_func, optimizer, step_scheduler, train_transform, valid_transform, args):

    train_dataset = Dataset(args.train_image_path, train_transform)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
    )

    start_epoch = 0

    # 断点继续训练
    if args.resume:
        checkpoint = torch.load(args.chkpt)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
    
    for epoch in range(start_epoch + 1, args.epoch):
        #train
        model.train()
        batch_loss = 0
        item = 0
        train_acc_num = 0
        for _, (image, label) in tqdm(enumerate(train_dataloader)):
            image = image.to(args.device)
            label = label.to(args.device)
            optimizer.zero_grad()
            output = model(image)
            predict_label = torch.max(output, dim=1)[1]
            for index in range(predict_label.shape[0]):
                if predict_label[index] == label[index]:
                    train_acc_num += 1
            loss = loss_func(output, label)
            loss.backward()
            optimizer.step()
            print("Train Epoch = {} Loss = {}".format(epoch, loss.data.item()))
            batch_loss += loss.data.item()
            item += 1
            if args.wandb:
                wandb.log({'loss': loss.data.item()})

        train_acc = train_acc_num / (train_dataloader.batch_size * item)

        if args.wandb:
            wandb.log({'train acc': train_acc})
        train_epoch_loss =  batch_loss / item
        print("Epoch = {} Train Loss = {} Train acc = {}".format(epoch, train_epoch_loss, train_acc))

        if args.lr_update:
            step_scheduler.step()

        #validate
        val_acc = predict(model, args.device, args.valid_image_path)
        print('val acc: ', val_acc)
        if args.wandb:
            wandb.log({'epoch':epoch, 'val acc':val_acc})
        
    #test
    test_acc = predict(model, args.device, args.test_image_path)
    print(test_acc)
    if args.wandb:
        wandb.log({'test acc':test_acc})

    #save
    checkpoint = {
    "net": model.state_dict(),
    'optimizer': optimizer.state_dict(),
    "epoch": epoch
    }
    save_model_file = os.path.join(args.output_dir, "mymodel.pth")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    torch.save(checkpoint, save_model_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda:0')
    parser.add_argument('--model', type=str, default='densenet201', help='model name')
    parser.add_argument('--num_workers', type=int, default=8, help='cpu threads num')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_update', action="store_true", help='update lr or not')
    parser.add_argument('--lr_step_size', type=int, default=10, help='step of update')
    parser.add_argument('--lr_update_gamma', type=float, default=0.1, help='ratio of each update')
    parser.add_argument('--epoch', type=int, default=150, help='epoch num')
    parser.add_argument('--num_classes', type=int, default=19, help='classes number')

    parser.add_argument('--wandb', action="store_true",)
    parser.add_argument('--train_image_path', type=str, default='data/train')
    parser.add_argument('--valid_image_path', type=str, default='data/val')
    parser.add_argument('--test_image_path', type=str, default='data/test')
    parser.add_argument('--output_dir', type=str, default='checkpoints/')
    parser.add_argument('--resume', action="store_true", help='use a trained model or not')
    parser.add_argument('--chkpt', type=str, default='checkpoints/densenet201.pth', 
                                            help='there are two pretrained model in dir checkpoints/, densenet201.pth and mymodel.pth')
    args = parser.parse_args()
    print(args)
    if args.wandb:
        import wandb
        wandb.init(project='AI_project_2',
                    #entity='kfq20', 
                    name='densenet201')

    if args.model == 'mymodel':
        model = MyModel()
        train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 各通道颜色的均值和方差,用于归一化
        ])
        valid_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 各通道颜色的均值和方差,用于归一化
        ])
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    elif args.model == 'densenet201':
        model = models.densenet201(pretrained=True)
        model.classifier = nn.Linear(1920, args.num_classes)
        optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)
        train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 各通道颜色的均值和方差,用于归一化
        ])
        valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 各通道颜色的均值和方差,用于归一化
        ])
        
    elif args.model == 'resnet18':
        model = models.resnet18(pretrained=True)
        num_fits = model.fc.in_features
        model.fc = nn.Linear(num_fits, args.num_classes)
        optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
        train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 各通道颜色的均值和方差,用于归一化
        ])
        valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 各通道颜色的均值和方差,用于归一化
        ])
        
    elif args.model == 'resnet152':
        model = models.resnet152(pretrained=True)
        num_fits = model.fc.in_features
        model.fc = nn.Linear(num_fits, args.num_classes)
        train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 各通道颜色的均值和方差,用于归一化
        ])
        valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 各通道颜色的均值和方差,用于归一化
        ])
        optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)

    model = model.to(args.device)
    criterion = nn.CrossEntropyLoss()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_update_gamma)
    train(model, criterion, optimizer, scheduler, train_transform, valid_transform, args)