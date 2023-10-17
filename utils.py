import torch
from torch import nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import  matplotlib.pyplot as plt
from torchvision import transforms, models



transform = transforms.Compose([transforms.Resize((450, 600)), transforms.ColorJitter(), transforms.ToTensor(), transforms.Normalize(mean=0., std=1.)])
device = 'cuda'
                                

def plot_cm(predictions):
    display_labels = ["Lasion", "Tumour"]

    y_pred, y_true = zip(*predictions)
    y_pred = list(y_pred)
    y_true = list(y_true)
    y_pred = torch.cat(y_pred)
    y_true = torch.cat(y_true)

    y_pred = torch.round(y_pred.squeeze(-1).type(torch.float32)).detach().cpu()
    y_true = y_true.type(torch.float32)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(include_values=True, cmap="Blues", ax=None, xticks_rotation="horizontal")
    plt.show()



def plot_feature_map(model, image=(str('DATA/BCC/ISIC_0024331.jpg'))):
    # we will save the conv layer weights in this list
    model_weights =[]
    #we will save the conv layers in this list
    conv_layers = []
    # get all the model children as list
    model_children = list(model.children())
    #counter to keep count of the conv layers
    counter = 0
    #append all the conv layers and their respective wights to the list
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter+=1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                for child in model_children[i][j].children():
                    if type(child) == nn.Conv2d:
                        counter+=1
                        model_weights.append(child.weight)
                        conv_layers.append(child)
        print(f"Total convolution layers: {counter}")
        print("conv_layers")

    image = transform(image)
    print(f"Image shape before: {image.shape}")
    image = image.unsqueeze(0)
    print(f"Image shape after: {image.shape}")
    image = image.to(device)    
    outputs = []
    names = []
    for layer in conv_layers[0:]:
        image = layer(image)
        outputs.append(image)
        names.append(str(layer))
    print(len(outputs))
    #print feature_maps
    for feature_map in outputs:
        print(feature_map.shape)

    processed = []
    for feature_map in outputs:
        feature_map = feature_map.squeeze(0)
        gray_scale = torch.sum(feature_map,0)
        gray_scale = gray_scale / feature_map.shape[0]
        processed.append(gray_scale.data.cpu().numpy())
    for fm in processed:
        print(fm.shape)

    fig = plt.figure(figsize=(30, 50))
    for i in range(len(processed)):
        a = fig.add_subplot(5, 4, i+1)
        imgplot = plt.imshow(processed[i])
        a.axis("off")
        a.set_title(names[i].split('(')[0], fontsize=30)
    plt.savefig(str('plots/feature_maps.jpg'), bbox_inches='tight')