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
parser.add_argument('--src', type=str, default='A', metavar='S',
                    help='source dataset')
parser.add_argument('--tgt', type=str, default='C', metavar='S',
                    help='target dataset')
parser.add_argument('--num_iter', type=int, default=20002,
                    help='max iter_num')
args = parser.parse_args()

def get_datasetname(args):
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

    return src, tgt


src, tgt = get_datasetname(args)

# batch_size = {"train": 36, "val": 36, "test": 4}
batch_size = {"train": 36, "val": 36, "test": 36}
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_target(loader, model, domain_encoder, test_iter=0):
    with torch.no_grad():
        start_test = True
        if test_iter > 0:
            iter_val = iter(loader['test'])
            for i in range(test_iter):
                data = iter_val.next()
                inputs = data[0]
                print(data[0].size())
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
                total_inputs = torch.cat(inputs, dim=0)
                tgt_domain_input = torch.from_numpy(np.array([[1, 0]] * total_inputs.size()[0])).float()
                tgt_domain_input = tgt_domain_input.to(device)
                domain_code = domain_encoder(tgt_domain_input)
                _, total_outputs, _ = model(total_inputs, domain_code)
                outputs = torch.chunk(total_outputs, 10, dim=0)

                outputs = sum(outputs)
                # print(outputs.size())
                # exit()
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
    # lr = init_lr * (1 + gamma * iter_num) ** (-power)
    lr = init_lr * (1 + gamma * iter_num) ** (-power)
    i = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        param_group['weight_decay'] = weight_decay * 2
        i += 1
    return optimizer


class DomainEncoder(nn.Module):

    def __init__(self, domain_hidden_state=1):
        super(DomainEncoder, self).__init__()
        self.domain_hidden_state = 1
        self.domain_encoder = nn.Linear(2, domain_hidden_state)
        self.domain_encoder.apply(init_weights)

    def forward(self, domain_input):
        return self.domain_encoder(domain_input)
        pass


class TransPredictor(nn.Module):

    def __init__(self):
        super(TransPredictor, self).__init__()
        self.trans_layer1 = nn.Linear(2048, 1000)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)

        self.trans_layer2 = nn.Linear(1000, 1000)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)

        self.trans_layer3 = nn.Linear(1000, 65)

        self.train_predictor = nn.Sequential(self.trans_layer1, self.relu1, self.dropout1,
                                             self.trans_layer2, self.relu2, self.dropout2,
                                             self.trans_layer3)

    def forward(self, rec_feature):
        return self.train_predictor(rec_feature)


class Decoder(nn.Module):
    
    def __init__(self, input_size=1025, ):
        super(Decoder, self).__init__()
        self.decoder_layer1 = nn.Linear(input_size, 2048)
        self.decoder_layer1.apply(init_weights)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        pass

    def forward(self, inner_code, domain_code):
        decoder_input = torch.cat([inner_code, domain_code], dim=1)
        return self.dropout(self.relu(self.decoder_layer1(decoder_input)))


class BSP_CDAN(nn.Module):
    def __init__(self, num_features):
        super(BSP_CDAN, self).__init__()
        self.model_fc = model.Resnet50Fc()
        self.bottleneck_layer1 = nn.Linear(num_features+1, 2000)
        self.bottleneck_layer1.apply(init_weights)
        self.bottleneck_layer2 = nn.Linear(2000, 1024)
        self.bottleneck_layer2.apply(init_weights)
        self.bottleneck_layer = nn.Sequential(self.bottleneck_layer1, nn.ReLU(), nn.Dropout(0.5),
                                              self.bottleneck_layer2, nn.ReLU(), nn.Dropout(0.5))

        self.classifier_layer = nn.Linear(1024, len(dset_classes))
        self.classifier_layer.apply(init_weights)
        self.predict_layer = nn.Sequential(self.model_fc, self.bottleneck_layer, self.classifier_layer)




    def forward(self, x, domain_code):
        # x = torch.cat([x, domain_code], dim=0)
        resnet_feat = self.model_fc(x)

        feature = torch.cat([resnet_feat, domain_code], dim=1)
        out = self.bottleneck_layer(feature)
        outC = self.classifier_layer(out)
        return (out, outC, resnet_feat)


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


class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature, hidden_size):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 10000.0

    def forward(self, x):
        if self.training:
            self.iter_num += 1
        coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
        x = x * 1.0
        x.register_hook(grl_hook(coeff))
        x = self.ad_layer1(x)
        x = self.relu1(x)
        y = self.ad_layer3(x)
        y = self.sigmoid(y)
        return y

    def output_num(self):
        return 1

    def get_parameters(self):
        # return [{"params": self.parameters(), "lr_mult": 10, 'decay_mult': 2}]
        return [{"params": self.parameters()}]


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)


def CDAN(feature, ad_net,):

    # softmax_output = input_list[1].detach()
    # feature = input_list[0]
    batch_size = feature.size(0) // 2
    ad_out = ad_net(feature)
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    return nn.BCELoss()(ad_out, dc_target)

def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()

    return fun1

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n

        return mse


if __name__ == '__main__':
    num_features = 2048
    domain_encoder = DomainEncoder()
    domain_encoder = domain_encoder.to(device)
    net = BSP_CDAN(num_features)
    net = net.to(device)
    decoder = Decoder()
    decoder = decoder.to(device)
    # ad_net = AdversarialNetwork(256 * len(dset_classes), 1024)
    ad_net = AdversarialNetwork(1024, 100)
    ad_net = ad_net.to(device)

    transpredictor = TransPredictor()
    transpredictor = transpredictor.to(device)

    mse = MSE()
    mse = mse.to(device)

    net.train(True)
    ad_net.train(True)
    criterion = {"classifier": nn.CrossEntropyLoss(), "adversarial": nn.BCELoss()}
    optimizer_dict = [{"params": filter(lambda p: p.requires_grad, net.model_fc.parameters()), "lr": 0.1},
                      {"params": filter(lambda p: p.requires_grad, net.bottleneck_layer.parameters()), "lr": 1},
                      {"params": filter(lambda p: p.requires_grad, net.classifier_layer.parameters()), "lr": 1},
                      {"params": filter(lambda p: p.requires_grad, ad_net.parameters()), "lr": 1},
                      {"params": filter(lambda p: p.requires_grad, domain_encoder.parameters()), "lr": 1},
                      {"params": filter(lambda p: p.requires_grad, decoder.parameters()), "lr": 1},
                      {"params": filter(lambda p: p.requires_grad, transpredictor.parameters()), "lr": 1}]

    optimizer = optim.SGD(optimizer_dict, lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)
    train_cross_loss = train_transfer_loss = train_total_loss = train_sigma = 0.0
    len_source = len(dset_loaders["train"]) - 1
    len_target = len(dset_loaders["val"]) - 1
    param_lr = []
    iter_source = iter(dset_loaders["train"])
    iter_target = iter(dset_loaders["val"])
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    test_interval = 100
    num_iter = args.num_iter
    best_result = 0.0
    record_file_path = "best_result.txt"
    record = open(record_file_path, "a")

    for iter_num in range(1, num_iter + 1):
        # print(iter_num)
        net.train(True)
        optimizer = inv_lr_scheduler(param_lr, optimizer, iter_num, init_lr=0.002, gamma=0.001, power=0.75,
                                     weight_decay=0.0005)

        optimizer.zero_grad()
        if iter_num % len_source == 0:
            iter_source = iter(dset_loaders["train"])
        if iter_num % len_target == 0:
            iter_target = iter(dset_loaders["val"])
        data_source = iter_source.next()
        data_target = iter_target.next()
        inputs_source, labels_source = data_source
        inputs_target, labels_target = data_target
        inputs = torch.cat((inputs_source, inputs_target), dim=0)
        dc_target = torch.from_numpy(np.array([[1], ] * batch_size["train"] + [[0], ] * batch_size["train"])).float()
        src_domain_input = torch.from_numpy(np.array([[0, 1]] * batch_size['train'])).float()
        tgt_domain_input = torch.from_numpy(np.array([[1, 0]] * batch_size['train'])).float()
        trans_domain_input = torch.from_numpy(np.array([[1, 0]] * batch_size['train'])).float()

        domain_input = torch.cat([src_domain_input, tgt_domain_input], dim=0).to(device)
        trans_domain_input = trans_domain_input.to(device)
        inputs = inputs.to(device)
        labels = labels_source.to(device)
        dc_target = dc_target.to(device)
        # print(domain_input.size())
        # print(inputs.size())
        # print()
        # exit()

        domain_code = domain_encoder(domain_input)
        trans_domain_code = domain_encoder(trans_domain_input)
        cls_domain_code = domain_encoder(src_domain_input.to(device))
        feature, outC, resnet_feature = net(inputs, domain_code)

        rec_feature = decoder(feature, domain_code)

        trans_rec_feature = decoder(feature.narrow(0, 0, int(feature.size(0) / 2)), trans_domain_code)
        cls_rec_feature = decoder(feature.narrow(0, 0, int(feature.size(0) / 2)), trans_domain_code)
        rec_label_trans = transpredictor(trans_rec_feature)
        rec_label_cls = transpredictor(cls_rec_feature)

        trans_label_loss =\
            criterion["classifier"](rec_label_trans, labels) + criterion["classifier"](rec_label_cls, labels)

        mse_loss = mse(rec_feature, resnet_feature)

        classifier_loss = criterion["classifier"](outC.narrow(0, 0, batch_size["train"]), labels)
        total_loss = classifier_loss
        # softmax_out = nn.Softmax(dim=1)(outC)
        coeff = calc_coeff(iter_num)
        transfer_loss = CDAN(feature, ad_net)
        # total_loss = total_loss + transfer_loss + sigma_loss
        total_loss = total_loss + transfer_loss + 0.5 * mse_loss + 0.5 * trans_label_loss
        total_loss.backward()
        optimizer.step()
        train_cross_loss += classifier_loss.item()
        train_transfer_loss += transfer_loss.item()
        train_total_loss += total_loss.item()
        train_sigma += 0
        if iter_num % test_interval == 0:
            print(
            "Iter {:05d}, Average Cross Entropy Loss: {:.4f}; Average Transfer Loss: {:.4f}; Average Sigma Loss: {:.4f}; Average Training Loss: {:.4f}".format(
                iter_num, train_cross_loss / float(test_interval), train_transfer_loss / float(test_interval),
                          train_sigma / float(test_interval),
                          train_total_loss / float(test_interval)))
            train_cross_loss = train_transfer_loss = train_total_loss = train_sigma = 0.0
        if (iter_num % 100) == 0:
            net.eval()
            test_acc = test_target(dset_loaders, net, domain_encoder)
            if test_acc > best_result:
                best_result = test_acc
                pass
            print("test_acc:%f\t best_result:%f" % (test_acc, best_result))
            print("%s->%s:%f".format(args.src, args.tgt, best_result), file=record)
