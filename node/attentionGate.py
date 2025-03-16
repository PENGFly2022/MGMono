import torch
import torch.nn as nn
from torch.nn.functional import unfold

class AttentionGatedMSG(nn.Module):
    def __init__(self, width=512, norm_layer=None, up_kwargs=None):
        super(AttentionGatedMSG, self).__init__()
        self.up_kwargs = up_kwargs
        self.ks = 3

        # kernel prediction based on the combined two different scales of features
        self.kernel_prediction_1 = nn.Conv2d(2 * width, 9, kernel_size=3, dilation=1, bias=True,
                                             padding=1)  # 4 groups of kernels and each kernel with 9 kernel values
        self.kernel_prediction_2 = nn.Conv2d(2 * width, 9, kernel_size=3, dilation=4, bias=True,
                                             padding=4)  # 4 groups of kernels and each kernel with 9 kernel values
        self.kernel_prediction_3 = nn.Conv2d(2 * width, 9, kernel_size=3, dilation=8, bias=True,
                                             padding=8)  # 4 groups of kernels and each kernel with 9 kernel values

        # kernel prediction for attention
        self.kernel_se_1 = nn.Conv2d(width, 9, kernel_size=3, dilation=1, bias=True,
                                     padding=1)  # one channel attention map
        self.kernel_sr_1 = nn.Conv2d(width, 9, kernel_size=3, dilation=1, bias=True,
                                     padding=1)  # one channel attention map

        self.kernel_se_2 = nn.Conv2d(width, 9, kernel_size=3, dilation=4, bias=True, padding=4)
        self.kernel_sr_2 = nn.Conv2d(width, 9, kernel_size=3, dilation=4, bias=True, padding=4)

        self.kernel_se_3 = nn.Conv2d(width, 9, kernel_size=3, dilation=8, bias=True, padding=8)
        self.kernel_sr_3 = nn.Conv2d(width, 9, kernel_size=3, dilation=8, bias=True, padding=8)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.combination_msgs = nn.Sequential(nn.Conv2d(3 * width, width, kernel_size=1),
                                              nn.ReLU(inplace=True))

    def struc_att(self, att,epoch=100, rank=1):
        bs, W, h, w = att.size()
        output=torch.zeros(bs, W, h, w).cuda()
        if rank == 0:
            output = att
        else:
            for i in range(rank):
                ch_weights = torch.randn(bs, W).cuda()
                ch_ag_weights = self.softmax(ch_weights).unsqueeze(-1).unsqueeze(-1)
                sp_weights = (ch_ag_weights * att).sum(1, True)
                sp_ag_weights = self.sigmoid(sp_weights)
                ch_weights = (sp_ag_weights * att).sum(-1).sum(-1)
                # ch_ag_weights = self.softmax(ch_weights).unsqueeze(-1).unsqueeze(-1)
                output=sp_ag_weights * ch_ag_weights * att+output
        return output


    def forward(self, sr,se,epoch=100,rank=1):
        # input[0] is last scale feature map
        inputs_se = se  # the feature map sending message
        inputs_sr = sr  # the feature map receiving message
        input_concat = torch.cat((inputs_se, inputs_sr), 1)
        # print(input_concat.shape,"input_concat")
        # weight prediction for different dilation rates
        dy_weights_1 = self.kernel_prediction_1(input_concat)
        dy_weights_1_ = dy_weights_1.view(dy_weights_1.size(0), 1, self.ks ** 2, dy_weights_1.size(2),
                                          dy_weights_1.size(3))
        dy_weights_2 = self.kernel_prediction_2(input_concat)
        dy_weights_2_ = dy_weights_2.view(dy_weights_2.size(0), 1, self.ks ** 2, dy_weights_2.size(2),
                                          dy_weights_2.size(3))
        dy_weights_3 = self.kernel_prediction_3(input_concat)
        dy_weights_3_ = dy_weights_3.view(dy_weights_3.size(0), 1, self.ks ** 2, dy_weights_3.size(2),
                                          dy_weights_3.size(3))

        dy_kernel_se_1 = self.kernel_se_1(inputs_se).unsqueeze(1)
        dy_kernel_sr_1 = self.kernel_sr_1(inputs_sr).unsqueeze(1)
        dy_kernel_se_2 = self.kernel_se_2(inputs_se).unsqueeze(1)
        dy_kernel_sr_2 = self.kernel_sr_2(inputs_sr).unsqueeze(1)
        dy_kernel_se_3 = self.kernel_se_3(inputs_se).unsqueeze(1)
        dy_kernel_sr_3 = self.kernel_sr_3(inputs_sr).unsqueeze(1)
        # new add 2020 2 12
        # unfold inputs
        f_se = inputs_se.shape  ##feature maps have the same shape
        f_sr = inputs_sr.shape
        inputs_se_1 = unfold(inputs_se, kernel_size=3, dilation=1, padding=1).view(f_se[0], f_se[1], self.ks ** 2,
                                                                                   f_se[2], f_se[3])
        inputs_sr_1 = unfold(inputs_sr, kernel_size=3, dilation=1, padding=1).view(f_sr[0], f_sr[1], self.ks ** 2,
                                                                                   f_sr[2], f_sr[3])
        inputs_se_2 = unfold(inputs_se, kernel_size=3, dilation=4, padding=4).view(f_se[0], f_se[1], self.ks ** 2,
                                                                                   f_se[2], f_se[3])
        inputs_sr_2 = unfold(inputs_sr, kernel_size=3, dilation=4, padding=4).view(f_sr[0], f_sr[1], self.ks ** 2,
                                                                                   f_sr[2], f_sr[3])
        inputs_se_3 = unfold(inputs_se, kernel_size=3, dilation=8, padding=8).view(f_se[0], f_se[1], self.ks ** 2,
                                                                                   f_se[2], f_se[3])
        inputs_sr_3 = unfold(inputs_sr, kernel_size=3, dilation=8, padding=8).view(f_sr[0], f_sr[1], self.ks ** 2,
                                                                                   f_sr[2], f_sr[3])

        # attention prediction

        attention_map_1 = inputs_sr * ((dy_weights_1_ * inputs_se_1).sum(2)) + (dy_kernel_se_1 * inputs_se_1).sum(2) + (
                    dy_kernel_sr_1 * inputs_sr_1).sum(2)

        attention_map_2 = inputs_sr * ((dy_weights_2_ * inputs_se_2).sum(2)) + (dy_kernel_se_2 * inputs_se_2).sum(2) + (
                    dy_kernel_sr_2 * inputs_sr_2).sum(2)

        attention_map_3 = inputs_sr * ((dy_weights_3_ * inputs_se_3).sum(2)) + (dy_kernel_se_3 * inputs_se_3).sum(2) + (
                    dy_kernel_sr_3 * inputs_sr_3).sum(2)
        # print(attention_map_1)
        # print(attention_map_2)
        # sturcure attention
        attention_map_1 = self.struc_att(attention_map_1,epoch,rank=rank)
        attention_map_2 = self.struc_att(attention_map_2,epoch,rank=rank)
        attention_map_3 = self.struc_att(attention_map_3,epoch,rank=rank)

        # attention gated message calcultation with different dilation rate
        message_1 = attention_map_1 * ((dy_weights_1_ * inputs_se_1).sum(2))
        message_2 = attention_map_2 * ((dy_weights_2_ * inputs_se_2).sum(2))
        message_3 = attention_map_3 * ((dy_weights_3_ * inputs_se_3).sum(2))

        # final message
        message_f = self.combination_msgs(torch.cat([message_1, message_2, message_3], 1))
        return message_f

# input_1 = (2,512,192,640)
# input_2 = (2,512,192,640)
# # 随即生成两个形状为input_1和input_2的Tensor
# tensor_1 = torch.randn(input_1)
# tensor_2 = torch.randn(input_2)
# print(tensor_1.shape,"tensor_1")
#
# model = AttentionGatedMSG()
# x = model(tensor_1,tensor_2)
# print(x.shape,"x")