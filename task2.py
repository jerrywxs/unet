import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import torch
from sklearn import metrics
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
from keras.utils import to_categorical
import imageio


class U_net(nn.Module):
    def crop(self, lower_conv, prev_conv):
        p = (prev_conv.size()[2] - lower_conv.size()[2]) // 2
        prev_conv = F.pad(prev_conv, [-p, -p, -p, -p])
        while prev_conv.size()[2] - lower_conv.size()[2]:
            prev_conv = F.pad(prev_conv, [-1, 0, -1, 0])
        while prev_conv.size()[3] - lower_conv.size()[3]:
            p = (prev_conv.size()[3] - lower_conv.size()[3])
            prev_conv = F.pad(prev_conv, [0, -p, 0, 0])
        return torch.cat((lower_conv, prev_conv), 1)

    def encode_block(self, in_ch, out_ch):
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=3, in_channels=in_ch, out_channels=out_ch),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.Conv2d(kernel_size=3, in_channels=out_ch, out_channels=out_ch),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_ch),
        )
        return block

    def decode_block(self, in_ch, mid_ch, out_ch):
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=3, in_channels=in_ch, out_channels=mid_ch),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(mid_ch),
            torch.nn.Conv2d(kernel_size=3, in_channels=mid_ch, out_channels=mid_ch),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(mid_ch),
            torch.nn.ConvTranspose2d(in_channels=mid_ch, out_channels=out_ch, kernel_size=3, stride=2,
                                     padding=1, output_padding=1)
        )
        return block

    def out_block(self, in_ch, mid_ch, out_ch):
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=3, in_channels=in_ch, out_channels=mid_ch),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(mid_ch),
            torch.nn.Conv2d(kernel_size=3, in_channels=mid_ch, out_channels=mid_ch),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(mid_ch),
            torch.nn.Conv2d(kernel_size=3, in_channels=mid_ch, out_channels=out_ch, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_ch),
        )
        return block

    def __init__(self, in_ch, out_ch):
        super(U_net, self).__init__()

        self.conv_encode1 = self.encode_block(in_ch, 64)
        self.conv_encode2 = self.encode_block(64, 128)
        self.conv_encode3 = self.encode_block(128, 256)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2)
        self.mid = self.decode_block(256, 512, 256)

        self.conv_decode3 = self.decode_block(512, 256, 128)
        self.conv_decode2 = self.decode_block(256, 128, 64)
        self.output = self.out_block(128, 64, out_ch)

    def forward(self, _input):
        encode_block1 = self.conv_encode1(_input)
        encode_block2 = self.conv_encode2(self.maxpool(encode_block1))
        encode_block3 = self.conv_encode3(self.maxpool(encode_block2))
        mid_block = self.mid(self.maxpool(encode_block3))
        decode_block3 = self.crop(mid_block, encode_block3)
        decode_block2 = self.crop(self.conv_decode3(decode_block3), encode_block2)
        decode_block1 = self.crop(self.conv_decode2(decode_block2), encode_block1)
        res = self.output(decode_block1)
        return res


def dice_loss(pred, label):
    dice = 0
    for i in range(pred.shape[1]):
        for j in range(pred.shape[0]):
            a = pred[j, i, :, :].numpy()
            b = (label == i)[j, :, :].numpy()
            dice += 2 * (a * b) / (np.sum(a) + np.sum(b))

    return dice


def bce(pred, label):
    _target = to_categorical(label)     # m * nb_of_class
    _target = torch.from_numpy(_target)
    pos = torch.eq(label, 1).float().resize_(label.shape[0], 1)
    neg = torch.eq(label, 0).float().resize_(label.shape[0], 1)
    total = torch.sum(pos) + torch.sum(neg)
    pos_cof = torch.sum(neg) / total
    neg_cof = torch.sum(pos) / total
    return F.binary_cross_entropy_with_logits(pred, _target, pos_cof * pos + neg_cof * neg, reduction='mean')


def calculate_performance(img, label):
    img = img.flatten()
    label = label.flatten()
    # print('accuracy: ', metrics.accuracy_score(label, img))
    precision = metrics.precision_score(label, img, average="binary")
    recall = metrics.recall_score(label, img, average="binary")
    f1 = 2 * precision * recall / (precision + recall)
    print('precision: ', precision)
    print('recall: ', recall)
    print('f1 measure: ', f1)
    with open('record.txt', 'a') as f:
        f.write(f'precision: {precision}\n')
        f.write(f'recall: {recall}\n')
        f.write(f'f1 measure: {f1}\n')


if __name__ == '__main__':
    dic = {'BG': 0, 'VS': 1}
    path = 'Training'

    image_dir = path + '/original_retinal_images'
    path_dic = dict()
    path_dic['VS'] = path + '/blood_vessel_segmentation_masks'

    image_mask_lis = []
    nb_of_img = 20
    nb_of_channel = 1
    nb_of_class = 2
    ratio = 2
    height, width = int(584 / ratio), int(565 / ratio)
    X_train = None
    y_train = None
    for i in range(21, nb_of_img+21):
        print(i)
        idx = str(i)
        if i < 10:
            idx = '0' + idx

        image = cv2.imread(image_dir + f'/{idx}_training.tif', 0)
        print(image.shape)
        image = cv2.resize(image, (width, height))
        # plt.subplot(2, 2, 1)
        # plt.imshow(image)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # from openCV tutorial
        image = clahe.apply(image)
        # print(image.shape)
        # plt.subplot(2, 2, 2)
        # plt.imshow(image)
        # plt.show()

        all_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        # for e in ['HE', 'EX', 'MA', 'SE']:
        flag = True
        for e in ['VS']:
            tmp = imageio.mimread(path_dic[e] + f'/{idx}_manual1.gif')
            tmp = np.array(tmp)[0, :, :]
            if tmp is None:
                nb_of_img -= 1
                flag = False
                continue
            # all_mask += cv2.threshold(tmp, thresh=1, maxval=dic[e], type=cv2.THRESH_BINARY)[1]
            # all_mask += cv2.resize(cv2.threshold(tmp, thresh=1, maxval=dic[e], type=cv2.THRESH_BINARY)[1], (width, height))
            # new = cv2.resize(cv2.threshold(tmp, thresh=1, maxval=dic[e], type=cv2.THRESH_BINARY)[1], (width, height))
            new = cv2.resize(cv2.threshold(tmp, thresh=1, maxval=1, type=cv2.THRESH_BINARY)[1], (width, height))
            all_mask += new * (all_mask == 0)
        # all_mask *= 50
        # plt.imshow(all_mask)
        # plt.show()
        # if i == 1:
        #     X_train = image
        #     y_train = all_mask
        # else:
        #     X_train = np.vstack([X_train, image])
        #     y_train = np.vstack([y_train, all_mask])
        if flag:
            image_mask_lis.append((image, all_mask))
        # print(np.max(all_mask), '-------------------------')
    # exit()

    X_train = np.hstack([i.flatten() for i, _ in image_mask_lis])       # why must flatten？
    y_train = np.hstack([i.flatten() for _, i in image_mask_lis])
    image_mask_lis = []
    X_train = torch.from_numpy(X_train).view(nb_of_img, height, width, nb_of_channel).float()
    X_train = X_train.permute(0, 3, 1, 2)
    y_train = torch.from_numpy(y_train).view(nb_of_img, 1, height, width).float()

    # for i in range(1, nb_of_img+1):
    nb_of_test = 3
    for i in range(1, nb_of_test+1):
        print(i)
        idx = str(i)
        if i < 10:
            idx = '0' + idx

        image = cv2.imread('Test' + '/original_retinal_images' + f'/{idx}_test.tif', 0)
        print(image.shape)
        image = cv2.resize(image, (width, height))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # from openCV tutorial
        image = clahe.apply(image)

        all_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        flag = True
        for e in ['VS']:
            tmp = imageio.mimread('Test' + '/blood_vessel_segmentation_masks' + f'/{idx}_manual1.gif')
            tmp = np.array(tmp)[0, :, :]
            if tmp is None:
                nb_of_img -= 1
                flag = False
                continue
            new = cv2.resize(cv2.threshold(tmp, thresh=1, maxval=1, type=cv2.THRESH_BINARY)[1], (width, height))
            all_mask += new * (all_mask == 0)
        if flag:
            image_mask_lis.append((image, all_mask))
    print(len(image_mask_lis))
    X_test = np.hstack([i.flatten() for i, _ in image_mask_lis])  # why must flatten？
    y_test = np.hstack([i.flatten() for _, i in image_mask_lis])
    image_mask_lis = []
    X_test = torch.from_numpy(X_test).view(nb_of_test, height, width, nb_of_channel).float()
    X_test = X_test.permute(0, 3, 1, 2)
    y_test = torch.from_numpy(y_test).view(nb_of_test, 1, height, width).float()


    # Define our model
    model = U_net(nb_of_channel, nb_of_class)
    # Define your learning rate
    learning_rate = 0.01
    # Define your optimizer
    momentum = 0.99
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)  # ?
    # Define your loss function
    # criterion = nn.NLLLoss  # since soft-max is in log form
    trainingloss = []
    testingloss = []
    # Define number of iterations
    epochs = 100
    batch_size = 1
    print(nb_of_img)
    for epoch in range(1, epochs + 1):
        model.train()  # turn into train mode
        # step one : fit your model by using training data and get predict label
        total_loss = 0
        for i in range(int(X_train.shape[0] / batch_size)):
            # print(data_chunk[:, 1:].view(batch_size, 1, 28, 28)[0, 0, :, :].shape)
            # image = data_chunk[:, 1:].view(batch_size, 1, 28, 28)[0, 0, :, :]
            # plt.imshow(image, cmap="Greys")
            # plt.show()
            # target, data = data_chunk[:, 0], data_chunk[:, 1:].view(data_chunk.shape[0], 1, 28, 28)
            data = X_train[i * batch_size:(i + 1) * batch_size, :].reshape(-1, nb_of_channel, height, width)
            target = y_train[i * batch_size:(i + 1) * batch_size, :]
            # plt.imshow(target[0, 0, :, :])
            # plt.show()
            output = model(data)
            # print(target.shape)
            #
            c = (target.size()[2] - output.size()[2]) // 2
            target = F.pad(target, (-c, -c, -c, -c))
            if target.size()[3] - output.size()[3]:
                c = (target.size()[3] - output.size()[3]) // 2
                target = F.pad(target, (-c, -c, 0, 0))
            # print(output.shape, target.shape)
            target = target.flatten()
            # output = output.permute(0, 2, 3, 1).reshape(-1, nb_of_class)    # 本来是batch * classs * row * col？
            # target = target.resize(66640)
            output = output.permute(0, 2, 3, 1)
            new_size = output.shape[0] * output.shape[1] * output.shape[2]
            output = output.resize(new_size, nb_of_class)
            # step two: calculate your training loss

            # print(output.shape, target.shape)
            # print(min(target), max(target))
            # loss = F.cross_entropy(output, target.long())  # ?????\
            # target = target.reshape((-1, 1))

            # output = output.reshape((-1, 1))
            # loss = F.binary_cross_entropy(F.sigmoid(output), target)
            loss = bce(output, target)
            total_loss += loss.item() * data.shape[0]  # o(︶︿︶)o this is already average loss4
            # output = output.permute(0, 2, 3, 1)
            # loss = dice_loss(output, target)
            # total_loss += loss

            # print(epoch, loss.item())
            # step three: calculate backpropagation
            loss.backward()
            # step four: update parameters
            optimizer.step()
            # step five: reset our optimizer
            optimizer.zero_grad()
        # step six: store your training loss
        trainingloss.append(total_loss / X_train.shape[0])
        print(f'{epoch, trainingloss[-1]}')
        # print(f'epoch: {epoch}', trainingloss[-1])
        # step seven: evaluation your model by using testing data and get the accuracy
        with torch.no_grad():
            model.eval()
            # predict testing data
            target = y_test

            if epoch % 1 == 0:
                plt.subplot(2, 2, 1)
                plt.imshow(target.numpy().astype(np.uint8)[1, 0, :, :] * 50)

            data = X_test.reshape(-1, nb_of_channel, height, width)
            output = model(data)
            c = (target.size()[2] - output.size()[2]) // 2
            target = F.pad(target[:3], (-c, -c, -c, -c))
            if target.size()[3] - output.size()[3]:
                c = (target.size()[3] - output.size()[3]) // 2
                target = F.pad(target, (-c, -c, 0, 0))

            if epoch % 1 == 0:
                plt.subplot(2, 2, 2)
                plt.imshow(target.numpy().astype(np.uint8)[1, 0, :, :] * 50)
                plt.subplot(2, 2, 3)
                print('output size: ', output.shape)
                y_hat = output.argmax(dim=1)
                plt.imshow(y_hat.numpy().astype(np.uint8)[1, :, :] * 50)
                # plt.show()
                plt.savefig(f'epoch {epoch}')
                plt.close()

            target = target.flatten()
            output = output.permute(0, 2, 3, 1).reshape(-1, nb_of_class)
            # calculate your testing loss
            loss = F.cross_entropy(output, target.long())
            # loss = bce2d(output, target)
            # store your testing loss       # testingloss += loss.item(), 这是什么操作。。怎么不报错?
            # if epoch % 10 == 0:
            if True:
                # get labels with max values
                # print(output.shape)
                y_hat = output.argmax(dim=1)
                # calculate the accuracy
                # print(y_hat)
                # print(target)
                acc = sum(int(i) for i in y_hat == target) / target.shape[0]
                print('Epoch:', epoch, 'Test Accuracy:', acc)
                calculate_performance(y_hat, target)
                print('----------------------------------------')




