import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
from node.att import AttentionBlock
class AtLoc(nn.Module):
    def __init__(self, feature_extractor, droprate=0.5, pretrained=True, feat_dim=2048):
        super(AtLoc, self).__init__()
        self.droprate = droprate
        # replace the last FC layer in feature extractor
        self.feature_extractor = feature_extractor
        self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
        fe_out_planes = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(fe_out_planes, feat_dim)
        self.att = AttentionBlock(feat_dim)
        self.fc_xyz = nn.Linear(feat_dim, 3)
        self.fc_wpqr = nn.Linear(feat_dim, 3)
        # initialize
        if pretrained:
            init_modules = [self.feature_extractor.fc, self.fc_xyz, self.fc_wpqr]
        else:
            init_modules = self.modules()
        for m in init_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
    def forward(self, x):
        x = self.feature_extractor(x)
        x = F.relu(x)
        x = self.att(x.view(x.size(0), -1))
        if self.droprate > 0:
            x = F.dropout(x, p=self.droprate)
        xyz = self.fc_xyz(x)
        wpqr = self.fc_wpqr(x)
        return xyz, wpqr
