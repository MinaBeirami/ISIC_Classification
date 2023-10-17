import torch
from torchvision import models
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchvision import models
from torchmetrics.classification import F1Score, AUROC, Recall, ROC, Accuracy, Precision, Specificity 


class ISICClassifier2(pl.LightningModule):
    def __init__(
        self,
        model_name,
        weights=None,
        learning_rate=0.001,
        weight_decay=1e-4,
        gamma=2,
        num_classes = 2,
        num_channels = 3,
    ):
        """Initialize with pretrained weights instead of None if a pretrained model is required."""
        super().__init__()
        self.training_step_losses = []
        self.training_step_accuracy = []

        self.validation_step_losses = []
        self.validation_step_accuracy = []

        self.weights = weights
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gamma = gamma
    
        self.num_classes = num_classes
        self.num_channels = num_channels

        self.model_name = model_name

        self.conv_reshape = []
        # models ###################
        if self.model_name == "resnet50":
            self.model = self._get_resnet50_model()

        elif self.model_name == "cnn":
            self._mina_cnn()       
        
        if self.num_classes >2:
            self.classification = 'multiclass'
        elif self.num_classes == 2:
            self.classification = 'binary'


        # metrics ###################
        self.f1 = F1Score(task=self.classification, num_classes=self.num_classes)
        self.auroc = AUROC(task=self.classification, num_classes=self.num_classes)
        self.recall = Recall(task=self.classification, num_classes=self.num_classes)
        self.accuracy = Accuracy(task=self.classification, num_classes=self.num_classes)
        self.precision = Precision(task=self.classification, num_classes=self.num_classes)
        self.specifity = Specificity(task=self.classification, num_classes=self.num_classes)
        self.save_hyperparameters()

    
    def _ce_loss(self, y_pred, y_true):
        criterion = torch.nn.CrossEntropyLoss()
        return criterion(y_pred, y_true)

    def _mina_cnn(self):
        #TODO : write layers for cnn
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.fc1 = nn.Linear(1600, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, self.num_classes)
        
    def _get_resnet50_model(self):
        model = models.resnet50(weights=self.weights)
        # Set last layer to return single output
        last_features = model.fc.in_features
        model.fc = torch.nn.Linear(last_features, self.num_classes)
        return model


    ### add more models above if necessary ###


    def _metrics(self, y_pred, y_true):
        f1 = self.f1(y_pred, y_true)
        auroc = self.auroc(y_pred, y_true)
        recall = self.recall(y_pred, y_true)
        precision = self.precision(y_pred, y_true)
        accuracy = self.accuracy(y_pred, y_true)
        specifity = self.specifity(y_pred, y_true)
        return f1, auroc, recall, precision, accuracy, specifity

    def forward(self, imgs):
        if self.model_name == 'cnn':
            out = self.conv_layer1(imgs)
            out = self.conv_layer2(out)
            out = self.max_pool1(out)

            out = self.conv_layer3(out)
            out = self.conv_layer4(out)
            out = self.max_pool2(out)

            out = out.reshape(out.size(0), -1)

            out = self.fc1(out)
            out = self.relu1(out)
            output = self.fc2(out)

        elif self.model_name == 'resnet50':
            output = self.model(imgs)
            
            
        return output
    
    
    def configure_optimizers(self):
       
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        inputs, y_true = batch
        y_pred = self.forward(inputs)
        #print('y_true shape:', y_true.shape, '\n', 'y_pred shape:', y_pred.shape)
        # loss
        loss = self._ce_loss(y_pred, y_true)

        f1, auroc, recall, precision, accuracy, specifity = self._metrics(y_pred, y_true)

        self.training_step_losses.append(loss)
        self.training_step_accuracy.append(accuracy)

        self.log("train_loss", loss, prog_bar=True, on_step=True)
        self.log("train_accuracy", accuracy, prog_bar=True, on_step=True)
        self.log("train_auroc", auroc, prog_bar=False, on_step=True)
        self.log("train_precision", precision, prog_bar=False, on_step=True)
        self.log("train_recall", recall, prog_bar=False, on_step=True)
        self.log("train_f1", f1, prog_bar=False, on_step=True)
        self.log("train_specifity", specifity, prog_bar=False, on_step=True)


        return loss

    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.training_step_losses).mean()
        avg_train_acc = torch.stack(self.training_step_accuracy).mean()

        self.log(
            "train_loss_epoch_end",
            avg_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_acc_epoch_end",
            avg_train_acc,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return avg_loss

    def validation_step(self, batch, batch_idx):
        inputs, y_true = batch
        #print('y_true shape:', y_true.shape)
        #y_true = y_true.unsqueeze(-1)
        # Forward pass
        y_pred = self.forward(inputs)
        # loss
        loss = self._ce_loss(y_pred, y_true)

        f1, auroc, recall, precision, accuracy, specifity = self._metrics(y_pred, y_true)

        self.validation_step_losses.append(loss)
        self.validation_step_accuracy.append(accuracy)

        self.log("val_loss", loss, prog_bar=True, on_step=True)
        self.log("val_accuracy", accuracy, prog_bar=True, on_step=True)
        self.log("val_auroc", auroc, prog_bar=False, on_step=True)
        self.log("val_precision", precision, prog_bar=False, on_step=True)
        self.log("val_recall", recall, prog_bar=False, on_step=True)
        self.log("val_f1", f1, prog_bar=False, on_step=True)
        self.log("val_specifity", specifity, prog_bar=False, on_step=True)


        return loss

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.validation_step_losses).mean()
        avg_train_acc = torch.stack(self.validation_step_accuracy).mean()

        self.log(
            "validation_loss_epoch_end",
            avg_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "validation_acc_epoch_end",
            avg_train_acc,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return avg_loss

    def predict_step(self, batch, batch_idx):
        inputs, y_true = batch
        y_pred = self.forward(inputs)
        y_pred = torch.nn.functional.sigmoid(y_pred)
        return y_pred, y_true

    def test_step(self, batch, batch_idx):
        inputs, y_true = batch
        #y_true = y_true.unsqueeze(-1)
        # Forward pass
        y_pred = self.forward(inputs)
        # loss
        
        loss = self._ce_loss(y_pred, y_true)

        f1, auroc, recall, precision, accuracy, specifity = self._metrics(y_pred, y_true)

        self.log("test_accuracy", accuracy, prog_bar=True, on_step=True)
        self.log("test_auroc", auroc, prog_bar=False, on_step=True)
        self.log("test_precision", precision, prog_bar=False, on_step=True)
        self.log("test_recall", recall, prog_bar=False, on_step=True)
        self.log("test_f1", f1, prog_bar=False, on_step=True)
        self.log("test_Specifity", specifity, prog_bar=False, on_step=True)
        return accuracy


class ISICClassifier(pl.LightningModule):
    def __init__(
        self,
        model_name,
        weights=None,
        learning_rate=0.001,
        weight_decay=1e-4,
        gamma=2,
        num_classes = 2,
        num_channels = 3,
    ):
        """Initialize with pretrained weights instead of None if a pretrained model is required."""
        super().__init__()
        self.training_step_losses = []
        self.training_step_accuracy = []

        self.validation_step_losses = []
        self.validation_step_accuracy = []

        self.weights = weights
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gamma = gamma

        if num_classes == 2:
            self.num_classes = num_classes - 1
        elif num_classes > 2:
            self.num_classes = num_classes

        self.num_channels = num_channels

        self.model_name = model_name

        self.conv_reshape = []
        # models ###################
        if self.model_name == "resnet50":
            self.model = self._get_resnet50_model()
     
        # possible bug with cross entropy and two classes
        if self.num_classes >2:
            self.classification = 'multiclass'
        elif self.num_classes == 1:
            self.classification = 'binary'


        # metrics ###################
        self.f1 = F1Score(task=self.classification, num_classes=self.num_classes)
        self.auroc = AUROC(task=self.classification, num_classes=self.num_classes)
        self.recall = Recall(task=self.classification, num_classes=self.num_classes)
        self.accuracy = Accuracy(task=self.classification, num_classes=self.num_classes)
        self.precision = Precision(task=self.classification, num_classes=self.num_classes)
        self.specifity = Specificity(task=self.classification, num_classes=self.num_classes)
        self.save_hyperparameters()

    def _bce_loss(self, y_pred, y_true):
        criterion = torch.nn.BCEWithLogitsLoss()
        return criterion(y_pred, y_true.float())

    def _ce_loss(self, y_pred, y_true):
        criterion = torch.nn.CrossEntropyLoss()
        return criterion(y_pred, y_true)

        
    def _get_resnet50_model(self):
        model = models.resnet50(weights=self.weights)
        # Set last layer to return single output
        last_features = model.fc.in_features
        model.fc = torch.nn.Linear(last_features, self.num_classes)
        return model


    ### add more models above if necessary ###


    def _metrics(self, y_pred, y_true):
        f1 = self.f1(y_pred, y_true)
        auroc = self.auroc(y_pred, y_true)
        recall = self.recall(y_pred, y_true)
        precision = self.precision(y_pred, y_true)
        accuracy = self.accuracy(y_pred, y_true)
        specifity = self.specifity(y_pred, y_true)
        return f1, auroc, recall, precision, accuracy, specifity

    def forward(self, imgs):
        output = self.model(imgs)
        return output
    
    
    def configure_optimizers(self):
       
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        inputs, y_true = batch
        y_pred = self.forward(inputs)
        # loss
        if self.classification == 'multiclass':
            loss = self._ce_loss(y_pred, y_true)
        elif self.classification == 'binary':
            y_true = y_true.unsqueeze(-1)
            loss = self._bce_loss(y_pred, y_true)

        f1, auroc, recall, precision, accuracy, specifity = self._metrics(y_pred, y_true)

        self.training_step_losses.append(loss)
        self.training_step_accuracy.append(accuracy)

        self.log("train_loss", loss, prog_bar=True, on_step=True)
        self.log("train_accuracy", accuracy, prog_bar=True, on_step=True)
        self.log("train_auroc", auroc, prog_bar=False, on_step=True)
        self.log("train_precision", precision, prog_bar=False, on_step=True)
        self.log("train_recall", recall, prog_bar=False, on_step=True)
        self.log("train_f1", f1, prog_bar=False, on_step=True)
        self.log("train_specifity", specifity, prog_bar=False, on_step=True)


        return loss

    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.training_step_losses).mean()
        avg_train_acc = torch.stack(self.training_step_accuracy).mean()

        self.log(
            "train_loss_epoch_end",
            avg_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_acc_epoch_end",
            avg_train_acc,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return avg_loss

    def validation_step(self, batch, batch_idx):
        inputs, y_true = batch
        # Forward pass
        y_pred = self.forward(inputs)
        # loss
        if self.classification == 'multiclass':
            loss = self._ce_loss(y_pred, y_true)
        elif self.classification == 'binary':
            y_true = y_true.unsqueeze(-1)
            loss = self._bce_loss(y_pred, y_true)

        f1, auroc, recall, precision, accuracy, specifity = self._metrics(y_pred, y_true)

        self.validation_step_losses.append(loss)
        self.validation_step_accuracy.append(accuracy)

        self.log("val_loss", loss, prog_bar=True, on_step=True)
        self.log("val_accuracy", accuracy, prog_bar=True, on_step=True)
        self.log("val_auroc", auroc, prog_bar=False, on_step=True)
        self.log("val_precision", precision, prog_bar=False, on_step=True)
        self.log("val_recall", recall, prog_bar=False, on_step=True)
        self.log("val_f1", f1, prog_bar=False, on_step=True)
        self.log("val_specifity", specifity, prog_bar=False, on_step=True)


        return loss

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.validation_step_losses).mean()
        avg_train_acc = torch.stack(self.validation_step_accuracy).mean()

        self.log(
            "validation_loss_epoch_end",
            avg_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "validation_acc_epoch_end",
            avg_train_acc,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return avg_loss

    # This is for binaryCE loss cuz one dimension 

    def predict_step(self, batch, batch_idx):
        inputs, y_true = batch
        y_pred = self.forward(inputs)
        if self.classification == 'binary':
            y_pred = torch.nn.functional.sigmoid(y_pred)
        return y_pred, y_true

    def test_step(self, batch, batch_idx):
        inputs, y_true = batch
        #y_true = y_true.unsqueeze(-1)
        # Forward pass
        y_pred = self.forward(inputs)
        # loss
        
        if self.classification == 'multiclass':
            loss = self._ce_loss(y_pred, y_true)
        elif self.classification == 'binary':
            y_true = y_true.unsqueeze(-1)
            loss = self._bce_loss(y_pred, y_true)

        f1, auroc, recall, precision, accuracy, specifity = self._metrics(y_pred, y_true)

        self.log("test_accuracy", accuracy, prog_bar=True, on_step=True)
        self.log("test_auroc", auroc, prog_bar=False, on_step=True)
        self.log("test_precision", precision, prog_bar=False, on_step=True)
        self.log("test_recall", recall, prog_bar=False, on_step=True)
        self.log("test_f1", f1, prog_bar=False, on_step=True)
        self.log("test_Specifity", specifity, prog_bar=False, on_step=True)
        return accuracy
