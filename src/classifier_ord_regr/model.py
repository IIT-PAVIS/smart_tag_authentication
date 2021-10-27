import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn.functional as F
import utils_3d_parties as ext_modules
from torch import nn


class HiQNanoClassifier_ord_regr(pl.LightningModule):
    def __init__(
        self,
        class_num=4,
        instances=225,
        learning_rate=1e-3,
        t_conv_size=5,
        weight=None,
    ):
        super().__init__()

        self.class_num = class_num
        self.learning_rate = learning_rate
        self.instances = instances

        self.in_planes = 3
        self.mid_planes1 = 64
        self.out_planes = 128

        self.t_conv_size = t_conv_size

        max_pool = nn.MaxPool3d((1, 2, 2))
        #max_pool_final = nn.MaxPool3d((1, 7, 7))
        max_pool_ad = nn.AdaptiveMaxPool3d((self.instances, 1, 1))
        #avg_pool = nn.AvgPool3d((1, 7, 7), stride=(1, 1, 1))

        self.model = nn.Sequential(
            # ext_modules.conv1x3x3(self.in_planes, self.in_planes, kernel_size=1,stride=1,padding=0),
            ext_modules.conv1x3x3(self.in_planes, self.mid_planes1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(self.mid_planes1),
            nn.ReLU(inplace=True),
            ext_modules.conv1x3x3(self.mid_planes1, self.mid_planes1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(self.mid_planes1),
            nn.ReLU(inplace=True),
            max_pool,
            # ext_modules.conv3x1x1(self.mid_planes1, self.mid_planes1, kernel_size=self.t_conv_size, stride=1, padding=self.t_conv_size // 2),
            nn.ReLU(inplace=True),
            ext_modules.conv1x3x3(self.mid_planes1, self.out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(self.out_planes),
            nn.ReLU(inplace=True),
            ext_modules.conv1x3x3(self.out_planes, self.out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(self.out_planes),
            nn.ReLU(inplace=True),
            max_pool,
            # ext_modules.conv3x1x1(self.out_planes, self.out_planes, kernel_size=self.t_conv_size, stride=1, padding=self.t_conv_size // 2),
            nn.ReLU(inplace=True),
            ext_modules.conv1x3x3(self.out_planes, self.out_planes, kernel_size=7, stride=1, padding=0),
            # max_pool_ad,
            # max_pool_final
            #avg_pool
        )
        
        self.model.apply(self.init_weights)
        self.fc = nn.Linear(self.out_planes,self.out_planes)
        # self.bn_fc = nn.BatchNorm1d(self.out_planes)
        # self.dropout = nn.Dropout(p=0.5)

        ## BIAS INIT is super important!!
        self.coral_weights = torch.nn.Linear(self.out_planes, 1, bias=False)
        self.coral_bias = torch.nn.Parameter( torch.range(self.class_num - 2, 0, -1).float() )
        # self.coral_bias = torch.nn.Linear(self.out_planes, self.class_num - 1)
        #self.coral_bias.requires_grad = False

        # Metrics
        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.train_f1 = pl.metrics.F1(self.class_num)
        self.valid_f1 = pl.metrics.F1(self.class_num)
        self.valid_conf = pl.metrics.ConfusionMatrix(self.class_num)

        if weight is not None:
            weight = torch.Tensor(weight)


    def init_weights(self, m):
        ## just for temporal convolutions: the middle element is 1, the rest is zero
        if isinstance(m, nn.Conv3d):
            if m.weight.data.shape[2] == self.t_conv_size:
                # m.weight.data.fill_(0.0)
                m.weight.data[:,:,:(self.t_conv_size//2),0,0] = 0.
                m.weight.data[:,:,(self.t_conv_size//2 + 1):,0,0] = 0.
                m.weight.data[:,:,self.t_conv_size//2,0,0] = torch.eye(m.weight.data.shape[0])
 

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.shape[0], x.shape[1], -1)
        x = x.permute((0, 2, 1))
        x = F.relu(self.fc(x))

        # x = x / torch.norm(x)
        x = self.coral_weights(x) + self.coral_bias #+ self.coral_bias(x)
        return x



    def training_step(self, train_batch, batch_idx):
        x, y, _ = train_batch
        logits = self.forward(x)

        y = y.view(-1)
        levels = ext_modules.levels_from_labelbatch(y, self.class_num).to(logits.device)
        logits = logits.view(-1, self.class_num - 1)

        loss = ext_modules.coral_loss(logits, levels)
        self.log("train_loss", loss, on_step=False, on_epoch=True)

        # self.train_acc(logits, y)
        # self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)

        labels = ext_modules.proba_to_label(torch.sigmoid(logits))
        # print(labels)
        # print(y)
        self.train_acc(labels, y)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y, _ = val_batch
        logits = self.forward(x)

        y = y.view(-1)
        levels = ext_modules.levels_from_labelbatch(y, self.class_num).to(logits.device)
        logits = logits.view(-1, self.class_num - 1)

        labels = ext_modules.proba_to_label(torch.sigmoid(logits))

        loss = ext_modules.coral_loss(logits, levels)
        # loss = self.cross_entropy_loss(logits, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True)

        # self.valid_acc(logits, y)
        # self.log("valid_acc", self.valid_acc, on_step=False, on_epoch=True)

        self.valid_acc(labels, y)
        self.log("valid_acc", self.valid_acc, on_step=False, on_epoch=True)

        conf_mat = self.valid_conf(labels, y)
        return conf_mat

    def validation_epoch_end(self, validation_step_outputs):
        # outs is a list of whatever you returned in `validation_step`
        conf_mat = sum(validation_step_outputs)

        # Logging Confusion Matrix
        matplotlib.use("Agg")
        df_cm = pd.DataFrame(
            conf_mat.detach().cpu().numpy(),
            index=range(self.class_num),
            columns=range(self.class_num),
        )
        plt.figure(figsize=(10, 7))
        fig_ = sns.heatmap(df_cm, annot=True, cmap="Spectral").get_figure()
        plt.close(fig_)
        self.logger.experiment.add_figure("Confusion matrix", fig_, self.current_epoch)

    def test_step(self, test_batch, batch_idx):
        x, y, _ = test_batch

        logits = self.forward(x)
        y = y.view(-1)
        levels = ext_modules.levels_from_labelbatch(y, self.class_num).to(logits.device)
        logits = logits.view(-1, self.class_num - 1)

        loss = ext_modules.coral_loss(logits, levels)
        # loss = self.cross_entropy_loss(logits, y)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
