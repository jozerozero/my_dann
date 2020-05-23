import torch
import torch.optim as optim
import torch.nn as nn
import model
import transform as tran
import adversarial1 as ad
import numpy as np
from read_data import ImageList
import argparse
import os
import torch.nn.functional as F


torch.set_num_threads(1)
parser = argparse.ArgumentParser(description='PyTorch BSP Example')
parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
parser.add_argument('--src', type=str, default='a', metavar='S',
                    help='source dataset')
parser.add_argument('--tgt', type=str, default='d', metavar='S',
                    help='target dataset')
parser.add_argument('--num_iter', type=int, default=50002,
                    help='max iter_num')
args = parser.parse_args()

def get_datasetname(args):
    noe = 0
    office = False
    A = './data/Art.txt'
    C = './data/Clipart.txt'
    P = './data/Product.txt'
    R = './data/Real_World.txt'
    if args.src == 'A':
        src = A
    elif args.src == 'C':
        src = C
    elif args.src == 'P':
        src = P
    elif args.src == 'R':
        src = R
    if args.tgt == 'A':
        tgt = A
    elif args.tgt == 'C':
        tgt = C
    elif args.tgt == 'P':
        tgt = P
    elif args.tgt == 'R':
        tgt = R
    a = './data/amazon.txt'
    w = './data/webcam.txt'
    d = './data/dslr.txt'
    if args.src == 'a':
        src = a
        office = True
    elif args.src == 'w':
        src = w
        office = True
    elif args.src == 'd':
        src = d
        noe = noe + 1
        office = True
    if args.tgt == 'a':
        tgt = a
        noe = noe + 1
    elif args.tgt == 'w':
        tgt = w
    elif args.tgt == 'd':
        tgt = d
    return src, tgt, office,noe


src, tgt, office, noe = get_datasetname(args)

batch_size = {"train": 36, "val": 36, "test": 4}

batch_size = {"train": 36, "val": 36, "test": 4}
for i in range(10):
    batch_size["val" + str(i)] = 4

data_transforms = {
        'train': tran.transform_train(resize_size=256, crop_size=224),
        'val': tran.transform_train(resize_size=256, crop_size=224),
    }
data_transforms = tran.transform_test(data_transforms=data_transforms, resize_size=256, crop_size=224)
dsets = {"train": ImageList(open(src).readlines(), transform=data_transforms["train"]),
         "val": ImageList(open(tgt).readlines(), transform=data_transforms["val"]),
         "test": ImageList(open(tgt).readlines(),transform=data_transforms["val"])}
dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size[x],
                                                   shuffle=True, num_workers=4)
                    for x in ['train', 'val']}
dset_loaders["test"] = torch.utils.data.DataLoader(dsets["test"], batch_size=batch_size["test"],
                                                       shuffle=False, num_workers=4)

for i in range(10):
    dsets["val" + str(i)] = ImageList(open(tgt).readlines(),
                                          transform=data_transforms["val" + str(i)])
    dset_loaders["val" + str(i)] = torch.utils.data.DataLoader(dsets["val" + str(i)],
                                                                   batch_size=batch_size["val" + str(i)], shuffle=False,
                                                                   num_workers=4)

dset_sizes = {x: len(dsets[x]) for x in ['train', 'val'] + ["val" + str(i) for i in range(10)]}
dset_classes = range(65)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_iter = args.num_iter


def test_target(loader, model, test_iter=0):
    with torch.no_grad():
        start_test = True
        if test_iter > 0:
            iter_val = iter(loader['val0'])
            for i in range(test_iter):
                data = iter_val.next()
                inputs = data[0]
                labels = data[1]
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                if start_test:
                    all_output = outputs.data.float()
                    all_label = labels.data.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.data.float()), 0)
                    all_label = torch.cat((all_label, labels.data.float()), 0)
        else:
            iter_val = [iter(loader['val' + str(i)]) for i in range(10)]
            for i in range(len(loader['val0'])):
                data = [iter_val[j].next() for j in range(10)]
                inputs = [data[j][0] for j in range(10)]
                labels = data[0][1]
                for j in range(10):
                    inputs[j] = inputs[j].to(device)
                labels = labels.to(device)
                outputs = []
                for j in range(10):
                    output = model(inputs[j])
                    outputs.append(output)
                outputs = sum(outputs)
                if start_test:
                    all_output = outputs.data.float()
                    all_label = labels.data.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.data.float()), 0)
                    all_label = torch.cat((all_label, labels.data.float()), 0)
        _, predict = torch.max(all_output, 1)
        accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return accuracy

def inv_lr_scheduler(param_lr, optimizer, iter_num, gamma, power, init_lr=0.001, weight_decay=0.0005):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (1 + gamma * iter_num) ** (-power)
    i = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        param_group['weight_decay'] = weight_decay * 2
        i += 1
    return optimizer


class BSP_CDAN(nn.Module):

    def __init__(self, num_feature):
        super(BSP_CDAN, self).__init__()
        self.model_fc = model.Resnet50Fc()
        self.bottleneck_layer1 = nn.Linear(num_feature, 256)
        self.bottleneck_layer1.apply(init_weights)
        self.bottleneck_layer = nn.Sequential(self.bottleneck_layer1, nn.ReLU(), nn.Dropout(0.5))
        self.classifiler_layer = nn.Linear(256, len(dset_classes))
        self.classifiler_layer.apply(init_weights)
        self.predict_layer = nn.Sequential(self.model_fc, self.bottleneck_layer, self.classifiler_layer)

    def forward(self, x):
        feature = self.model_fc(x)
        out = self.bottleneck_layer(feature)
        outC = self.classifiler_layer(out)

        return out, outC


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)


def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()

    return fun1


class AdversarialNetwork(nn.Module):

    def __init__(self, input_feature, hidden_size, dropout_rate=0.5):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_features=input_feature, out_features=hidden_size)
        self.ad_layer2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.ad_layer3 = nn.Linear(in_features=hidden_size, out_features=1)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)

        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 10000.0

    def forward(self, x):
        if self.training == 1:
            self.iter_num += 1
        coeff = calc_coeff(iter_num=self.iter_num, high=self.high, low=self.low,
                           alpha=self.alpha, max_iter=self.max_iter)
        x.register_hook(grl_hook(coeff=coeff))

        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        y = self.sigmoid(y)

        return y

    def output_num(self):
        return 1

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 10, 'decay_mult': 2}]


num_feature = 2048
net = BSP_CDAN(num_feature=num_feature)
net = net.to(device)
ad_net = AdversarialNetwork(input_feature=256, hidden_size=100)
ad_net = ad_net.to(device)
net.train(True)
ad_net.train(True)

criterion = {"classifier": nn.CrossEntropyLoss(), "adversarial": nn.BCELoss()}
optimizer_dict = [{"params": filter(lambda p: p.requires_grad, net.model_fc.parameters()), "lr": 0.1},
                  {"params": filter(lambda p: p.requires_grad, net.bottleneck_layer.parameters()), "lr": 1},
                  {"params": filter(lambda p: p.requires_grad, net.classifiler_layer.parameters()), "lr": 1},
                  {"params": filter(lambda p: p.requires_grad, ad_net.parameters()), "lr": 1}]

optimizer = optim.SGD(optimizer_dict, lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)

# train_cross_loss = train_transfer_loss = train_total_loss = train_sigma = 0.0
train_classifier_loss = 0.0
train_domain_loss = 0.0
train_total_loss = 0.0


len_source = len(dset_loaders["train"]) - 1
len_target = len(dset_loaders["val"]) - 1

param_lr = []
iter_source = iter(dset_loaders["train"])
iter_target = iter(dset_loaders["val"])

for param_group in optimizer.param_groups:
    param_lr.append(param_group["lr"])

test_interval = 100
num_iter = max_iter

for iter_num in range(1, num_iter + 1):
    print(iter_num)
    net.train(True)

    optimizer = inv_lr_scheduler(param_lr, optimizer, iter_num, init_lr=0.003, gamma=0.001, power=0.75)
    optimizer.zero_grad()

    if iter_num % len_source == 0:
        iter_source = iter(dset_loaders["train"])
    if iter_num % len_target == 0:
        iter_target = iter(dset_loaders["val"])

    data_source = iter_source.next()
    data_target = iter_target.next()

    input_source, label_source = data_source
    input_target, label_target = data_target

    inputs = torch.cat([input_source, input_target], dim=0)

    dc_target = torch.from_numpy(np.array([[1], ] * batch_size["train"] + [[0], ] * batch_size["train"])).float()

    inputs = inputs.to(device)
    labels =label_source.to(device)
    dc_target = dc_target.to(device)

    feature, outC = net(inputs)

    classifier_loss = criterion["classifier"](outC.narrow(0, 0, batch_size["train"]), labels)
    domain_logits = ad_net(feature)
    domain_loss = criterion["adversarial"](domain_logits, dc_target)

    total_loss = classifier_loss + domain_loss
    total_loss.backward()
    optimizer.step()

    train_total_loss += total_loss.item()
    train_classifier_loss += classifier_loss.item()
    train_domain_loss += domain_loss.item()

    if iter_num % test_interval == 0:
        print("global stpes: %d\t total_loss:%f\t domain_loss:%f\t label_loss:%f"
              % (iter_num, train_total_loss, train_domain_loss, train_classifier_loss))
        train_classifier_loss = 0.0
        train_domain_loss = 0.0
        train_total_loss = 0.0

        net.eval()
        test_acc = test_target(loader=dset_loaders, model=net.predict_layer)
        print('test_acc:%.4f' % (test_acc))



