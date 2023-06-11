from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.types import Device
import torchvision
from torchvision import datasets, models, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.utils.data import DataLoader
import datetime

from load_celeba import *
from model import *

from torch.utils.mobile_optimizer import optimize_for_mobile

def label2onehot(self, labels, dim):
    """Convert label indices to one-hot vectors."""
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    out[np.arange(batch_size), labels.long()] = 1
    return out

def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def create_labels(c_org, c_dim=5, dataset='CelebA', selected_attrs=None):
    """Generate target domain labels for debugging and testing."""
    # Get hair color indices.
    if dataset == 'CelebA':
        hair_color_indices = []
        for i, attr_name in enumerate(selected_attrs):
            if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                hair_color_indices.append(i)

    c_trg_list = []
    for i in range(c_dim):
        if dataset == 'CelebA':
            c_trg = c_org.clone()
            if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                c_trg[:, i] = 1
                for j in hair_color_indices:
                    if j != i:
                        c_trg[:, j] = 0
            else:
                c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.
        elif dataset == 'RaFD':
            c_trg = label2onehot(torch.ones(c_org.size(0))*i, c_dim)

        c_trg_list.append(c_trg.to(device))
    return c_trg_list

def print_network(model, name):
    """Print out the network information."""
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model)
    print(name)
    print("The number of parameters: {}".format(num_params))

if __name__ == '__main__':

    SELECT_ATTRS_TEST = "1"            # 1, 2, 1+2, oppure, D1, D2, D1+2, 2D1+2

    debug = False
    original = False                # True se vogliamo salvare le immagini originali
    save_images = False
    cartella_unica = True
    train = False
    set_classifer = 'teacher'       #teacher, student, student_FR, student_FR_10, student_FR_100, student_FR135, student_OUT, student_FR5OUT

    # Number of epochs to train for
    num_epochs = 15
    batch_size = 1

    result_original = 'classifier/test_2/main/'

    if(SELECT_ATTRS_TEST == "1"):
        if(set_classifer == 'teacher'):
            g_conv_dim = 64     #teacher 64, student 32
            g_repeat_num = 6    #teacher 6, student 3
            model_save_dir = 'classifier/generatore/teacher/'           #directory del generatore
            result_dir_FID = 'classifier/FID/teacher/'                  #directory delle immagini generate per il FID
            result_dir = 'classifier/test_2/teacher/'                   #directory delle immagini generate
        else:
            g_conv_dim = 32
            g_repeat_num = 3
            if(set_classifer == 'student'):
                model_save_dir = 'classifier/generatore/student/'
                result_dir_FID = 'classifier/FID/student/'
                result_dir = 'classifier/test_2/student/'
            elif (set_classifer == 'student_FR'):
                model_save_dir = 'classifier/generatore/student_FR/'            
                result_dir_FID = 'classifier/FID/student_FR/'
                result_dir = 'classifier/test_2/student_FR/'
            elif (set_classifer == 'student_FR_10'):
                model_save_dir = 'classifier/generatore/student_FR_10/'            
                result_dir_FID = 'classifier/FID/student_FR_10/'
                result_dir = 'classifier/test_2/student_FR_10/'
            elif (set_classifer == 'student_FR_100'):
                model_save_dir = 'classifier/generatore/student_FR_100/'            
                result_dir_FID = 'classifier/FID/student_FR_100/'
                result_dir = 'classifier/test_2/student_FR_100/'
            elif (set_classifer == 'student_FR135'):                        #flusso applicato a 3 blocchi residual [1,3,5]
                model_save_dir = 'classifier/generatore/student_FR135/'            
                result_dir_FID = 'classifier/FID/student_FR135/'
                result_dir = 'classifier/test_2/student_FR135/'
            elif (set_classifer == 'student_OUT'):                        #flusso applicato all'immagine di output
                model_save_dir = 'classifier/generatore/student_OUT/'            
                result_dir_FID = 'classifier/FID/student_OUT/'
                result_dir = 'classifier/test_2/student_OUT/'
            elif (set_classifer == 'student_FR5OUT'):                        #flusso applicato all'ultimo blocco residual + immagine di output
                model_save_dir = 'classifier/generatore/student_FR5OUT/'            
                result_dir_FID = 'classifier/FID/student_FR5OUT/'
                result_dir = 'classifier/test_2/student_FR5OUT/'
    elif(SELECT_ATTRS_TEST == "2"):
        if(set_classifer == 'teacher_2attr'):
            g_conv_dim = 64     #teacher 64, student 32
            g_repeat_num = 6    #teacher 6, student 3
            model_save_dir = 'classifier/generatore/teacher_2attr/'           #directory del generatore
            result_dir_FID = 'classifier/FID/teacher_2attr/'                  #directory delle immagini generate per il FID
            result_dir = 'classifier/test_2/teacher_2attr/'                   #directory delle immagini generate
        else:
            g_conv_dim = 32
            g_repeat_num = 3
            if(set_classifer == 'student_2attr'):
                model_save_dir = 'classifier/generatore/student_2attr/'
                result_dir_FID = 'classifier/FID/student_2attr/'
                result_dir = 'classifier/test_2/student_2attr/'
            elif (set_classifer == 'student_FR_2attr'):
                model_save_dir = 'classifier/generatore/student_FR_2attr/'            
                result_dir_FID = 'classifier/FID/student_FR_2attr/'
                result_dir = 'classifier/test_2/student_FR_2attr/'
            elif (set_classifer == 'student_FR135_2attr'):                        #flusso applicato a 3 blocchi residual [1,3,5]
                model_save_dir = 'classifier/generatore/student_FR135_2attr/'            
                result_dir_FID = 'classifier/FID/student_FR135_2attr/'
                result_dir = 'classifier/test_2/student_FR135_2attr/'
            elif (set_classifer == 'student_OUT_2attr'):                        #flusso applicato all'immagine di output
                model_save_dir = 'classifier/generatore/student_OUT_2attr/'            
                result_dir_FID = 'classifier/FID/student_OUT_2attr/'
                result_dir = 'classifier/test_2/student_OUT_2attr/'
            elif (set_classifer == 'student_FR5OUT_2attr'):                        #flusso applicato all'ultimo blocco residual + immagine di output
                model_save_dir = 'classifier/generatore/student_FR5OUT_2attr/'            
                result_dir_FID = 'classifier/FID/student_FR5OUT_2attr/'
                result_dir = 'classifier/test_2/student_FR5OUT_2attr/'
    elif(SELECT_ATTRS_TEST == "1+2"):
        if(set_classifer == 'teacher_12attr'):
            g_conv_dim = 64     #teacher 64, student 32
            g_repeat_num = 6    #teacher 6, student 3
            model_save_dir = 'classifier/generatore/teacher_12attr/'           #directory del generatore
            result_dir_FID = 'classifier/FID/teacher_12attr/'                  #directory delle immagini generate per il FID
            result_dir = 'classifier/test_2/teacher_12attr/'                   #directory delle immagini generate
        else:
            g_conv_dim = 16
            g_repeat_num = 1
            if(set_classifer == 'student_12attr'):
                model_save_dir = 'classifier/generatore/student_12attr/'
                result_dir_FID = 'classifier/FID/student_12attr/'
                result_dir = 'classifier/test_2/student_12attr/'
            elif (set_classifer == 'student_FR_12attr'):
                model_save_dir = 'classifier/generatore/student_FR_12attr/'            
                result_dir_FID = 'classifier/FID/student_FR_12attr/'
                result_dir = 'classifier/test_2/student_FR_12attr/'
            elif (set_classifer == 'student_OUT_12attr'):                        #flusso applicato all'immagine di output
                model_save_dir = 'classifier/generatore/student_OUT_12attr/'            
                result_dir_FID = 'classifier/FID/student_OUT_12attr/'
                result_dir = 'classifier/test_2/student_OUT_12attr/'
            elif (set_classifer == 'student_FR5OUT_12attr'):                        #flusso applicato all'ultimo blocco residual + immagine di output
                model_save_dir = 'classifier/generatore/student_FR5OUT_12attr/'            
                result_dir_FID = 'classifier/FID/student_FR5OUT_12attr/'
                result_dir = 'classifier/test_2/student_FR5OUT_12attr/'
    elif(SELECT_ATTRS_TEST == "D1"):
        if(set_classifer == 'teacher'):
            g_conv_dim = 64     #teacher 64, student 32
            g_repeat_num = 6    #teacher 6, student 3
            model_save_dir = 'classifier/generatore/teacher/'           #directory del generatore
            result_dir_FID = 'classifier/FID/teacher/'                  #directory delle immagini generate per il FID
            result_dir = 'classifier/test_2/teacher/'                   #directory delle immagini generate
        else:
            g_conv_dim = 32
            g_repeat_num = 3
            if(set_classifer == 'student_D_1attr'):
                model_save_dir = 'classifier/generatore/student_D_1attr/'
                result_dir_FID = 'classifier/FID/student_D_1attr/'
                result_dir = 'classifier/test_2/student_D_1attr/'
            elif (set_classifer == 'student_D_FR_1attr'):
                model_save_dir = 'classifier/generatore/student_D_FR_1attr/'            
                result_dir_FID = 'classifier/FID/student_D_FR_1attr/'
                result_dir = 'classifier/test_2/student_D_FR_1attr/'
            elif (set_classifer == 'student_D_FR5OUT_1attr'):                        #flusso applicato all'ultimo blocco residual + immagine di output
                model_save_dir = 'classifier/generatore/student_D_FR5OUT_1attr/'            
                result_dir_FID = 'classifier/FID/student_D_FR5OUT_1attr/'
                result_dir = 'classifier/test_2/student_D_FR5OUT_1attr/'
    elif(SELECT_ATTRS_TEST == "D2"):
        if(set_classifer == 'teacher_2attr'):
            g_conv_dim = 64     #teacher 64, student 32
            g_repeat_num = 6    #teacher 6, student 3
            model_save_dir = 'classifier/generatore/teacher_2attr/'           #directory del generatore
            result_dir_FID = 'classifier/FID/teacher_2attr/'                  #directory delle immagini generate per il FID
            result_dir = 'classifier/test_2/teacher_2attr/'                   #directory delle immagini generate
        else:
            g_conv_dim = 32
            g_repeat_num = 3
            if(set_classifer == 'student_D_2attr'):
                model_save_dir = 'classifier/generatore/student_D_2attr/'
                result_dir_FID = 'classifier/FID/student_D_2attr/'
                result_dir = 'classifier/test_2/student_D_2attr/'
            elif (set_classifer == 'student_D_FR_2attr'):
                model_save_dir = 'classifier/generatore/student_D_FR_2attr/'            
                result_dir_FID = 'classifier/FID/student_D_FR_2attr/'
                result_dir = 'classifier/test_2/student_D_FR_2attr/'
            elif (set_classifer == 'student_D_FR5OUT_2attr'):                        #flusso applicato all'ultimo blocco residual + immagine di output
                model_save_dir = 'classifier/generatore/student_D_FR5OUT_2attr/'            
                result_dir_FID = 'classifier/FID/student_D_FR5OUT_2attr/'
                result_dir = 'classifier/test_2/student_D_FR5OUT_2attr/'
    elif(SELECT_ATTRS_TEST == "D1+2"):
        if(set_classifer == 'teacher_12attr'):
            g_conv_dim = 64     #teacher 64, student 32
            g_repeat_num = 6    #teacher 6, student 3
            model_save_dir = 'classifier/generatore/teacher_12attr/'           #directory del generatore
            result_dir_FID = 'classifier/FID/teacher_12attr/'                  #directory delle immagini generate per il FID
            result_dir = 'classifier/test_2/teacher_12attr/'                   #directory delle immagini generate
        else:
            g_conv_dim = 16
            g_repeat_num = 1
            if(set_classifer == 'student_D_12attr'):
                model_save_dir = 'classifier/generatore/student_D_12attr/'
                result_dir_FID = 'classifier/FID/student_D_12attr/'
                result_dir = 'classifier/test_2/student_D_12attr/'
            elif (set_classifer == 'student_D_FR_12attr'):
                model_save_dir = 'classifier/generatore/student_D_FR_12attr/'            
                result_dir_FID = 'classifier/FID/student_D_FR_12attr/'
                result_dir = 'classifier/test_2/student_D_FR_12attr/'
            elif (set_classifer == 'student_D_FR5OUT_12attr'):                        #flusso applicato all'ultimo blocco residual + immagine di output
                model_save_dir = 'classifier/generatore/student_D_FR5OUT_12attr/'            
                result_dir_FID = 'classifier/FID/student_D_FR5OUT_12attr/'
                result_dir = 'classifier/test_2/student_D_FR5OUT_12attr/'
    elif(SELECT_ATTRS_TEST == "2D1+2"):
        if(set_classifer == 'teacher_12attr'):
            g_conv_dim = 64     #teacher 64, student 32
            g_repeat_num = 6    #teacher 6, student 3
            model_save_dir = 'classifier/generatore/teacher_12attr/'           #directory del generatore
            result_dir_FID = 'classifier/FID/teacher_12attr/'                  #directory delle immagini generate per il FID
            result_dir = 'classifier/test_2/teacher_12attr/'                   #directory delle immagini generate
        else:
            g_conv_dim = 32
            g_repeat_num = 3
            if(set_classifer == 'student_2_12attr'):
                model_save_dir = 'classifier/generatore/student_2_12attr/'
                result_dir_FID = 'classifier/FID/student_2_12attr/'
                result_dir = 'classifier/test_2/student_2_12attr/'
            elif (set_classifer == 'student_2_FR5_12attr'):
                model_save_dir = 'classifier/generatore/student_2_FR5_12attr/'            
                result_dir_FID = 'classifier/FID/student_2_FR5_12attr/'
                result_dir = 'classifier/test_2/student_2_FR5_12attr/'
            elif (set_classifer == 'student_2_FR135_12attr'):
                model_save_dir = 'classifier/generatore/student_2_FR135_12attr/'            
                result_dir_FID = 'classifier/FID/student_2_FR135_12attr/'
                result_dir = 'classifier/test_2/student_2_FR135_12attr/'
            elif (set_classifer == 'student_2_OUT_12attr'):
                model_save_dir = 'classifier/generatore/student_2_OUT_12attr/'            
                result_dir_FID = 'classifier/FID/student_2_OUT_12attr/'
                result_dir = 'classifier/test_2/student_2_OUT_12attr/'
            elif (set_classifer == 'student_2_FR5OUT_12attr'):                        #flusso applicato all'ultimo blocco residual + immagine di output
                model_save_dir = 'classifier/generatore/student_2_FR5OUT_12attr/'            
                result_dir_FID = 'classifier/FID/student_2_FR5OUT_12attr/'
                result_dir = 'classifier/test_2/student_2_FR5OUT_12attr/'
            elif(set_classifer == 'student_2D_12attr'):
                model_save_dir = 'classifier/generatore/student_2D_12attr/'
                result_dir_FID = 'classifier/FID/student_2D_12attr/'
                result_dir = 'classifier/test_2/student_2D_12attr/'
            elif (set_classifer == 'student_2D_FR5_12attr'):
                model_save_dir = 'classifier/generatore/student_2D_FR5_12attr/'            
                result_dir_FID = 'classifier/FID/student_2D_FR5_12attr/'
                result_dir = 'classifier/test_2/student_2D_FR5_12attr/'
            elif (set_classifer == 'student_2D_FR135_12attr'):
                model_save_dir = 'classifier/generatore/student_2D_FR135_12attr/'            
                result_dir_FID = 'classifier/FID/student_2D_FR135_12attr/'
                result_dir = 'classifier/test_2/student_2D_FR135_12attr/'
            elif (set_classifer == 'student_2D_OUT_12attr'):
                model_save_dir = 'classifier/generatore/student_2D_OUT_12attr/'            
                result_dir_FID = 'classifier/FID/student_2D_OUT_12attr/'
                result_dir = 'classifier/test_2/student_2D_OUT_12attr/'
            elif (set_classifer == 'student_2D_FR5OUT_12attr'):                        #flusso applicato all'ultimo blocco residual + immagine di output
                model_save_dir = 'classifier/generatore/student_2D_FR5OUT_12attr/'            
                result_dir_FID = 'classifier/FID/student_2D_FR5OUT_12attr/'
                result_dir = 'classifier/test_2/student_2D_FR5OUT_12attr/'

    

    # Create directories if not exist.
    if not os.path.exists(result_original):
        os.makedirs(result_original)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    if not os.path.exists(result_dir_FID):
        os.makedirs(result_dir_FID)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    g_lr = 0.0001
    beta1 = 0.5
    beta2 = 0.999 #Mustache, Goatee, Rosy_Cheeks, Bushy_Eyebrows, Wearing_Lipstick 
    
    image_dir = os.path.abspath("./stargan-student/data/celeba/images")
    attr_path = os.path.abspath("./stargan-student/data/celeba/list_attr_celeba.txt")

    selected_attrs=["5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", 
    "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", 
    "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose",
    "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", 
    "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young" ]

    if(SELECT_ATTRS_TEST == "1" or SELECT_ATTRS_TEST == "D1"):
        selected_test_attrs=["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young"]
        c_dim = 5
    elif(SELECT_ATTRS_TEST == "2" or SELECT_ATTRS_TEST == "D2"):
        selected_test_attrs=["Mustache", "Goatee", "Rosy_Cheeks", "Bushy_Eyebrows", "Wearing_Lipstick"]
        c_dim = 5
    elif(SELECT_ATTRS_TEST == "1+2" or SELECT_ATTRS_TEST == "D1+2" or SELECT_ATTRS_TEST == "2D1+2"):
        selected_test_attrs=["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young", 
                             "Mustache", "Goatee", "Rosy_Cheeks", "Bushy_Eyebrows", "Wearing_Lipstick"]
        c_dim = 10
    #for el in all:
    #    if el not in selected_attrs:
    #        selected_attrs.append(el)
    
    print(selected_attrs, "\n")
    print(selected_test_attrs, "\n")
    print(device, "\n")
    print(set_classifer, "\n")

    def set_parameter_requires_grad(model):
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if "layer3." in name or "layer4." in name or "fc." in name:
                param.requires_grad = True

    model_ft = models.resnet50(pretrained=True)
    set_parameter_requires_grad(model_ft)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(selected_attrs))
    model_ft.to(device)

    # celeba_loader = get_loader(args.celeba_image_dir, args.attr_path, args.selected_attrs,
    #                             args.celeba_crop_size, args.image_size, args.batch_size,
    #                             'CelebA', 'train', args.num_workers)

    # celeba_loader_test = get_loader(args.celeba_image_dir, args.attr_path, args.selected_attrs,
    #                             args.celeba_crop_size, args.image_size, args.batch_size,
    #                             'CelebA', 'test', args.num_workers)                            

    # Define transforms for celebA dataset
    celeba_transforms = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(178),
            transforms.Resize(128),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    #datasets_train = [CelebA_custom(image_dir, attr_path, selected_attrs, celeba_transforms, 'train', idx)
    #             for idx in range(len(selected_attrs))]
    #datasets_train = torch.utils.data.ConcatDataset(datasets_train)
     
    datasets_train = CelebASkillato(image_dir, attr_path, selected_attrs, celeba_transforms, 'train')         

    celeba_loader = DataLoader(
                datasets_train,
                shuffle = True,
                batch_size=batch_size,
                num_workers=0
        )

    #datasets_test = [CelebA_custom(image_dir, attr_path, selected_attrs, celeba_transforms, 'test', idx)
    #             for idx in range(len(selected_attrs))]
    #datasets_test = torch.utils.data.ConcatDataset(datasets_test)
    
    datasets_test = CelebASkillato(image_dir, attr_path, selected_test_attrs, celeba_transforms, 'test')

    celeba_loader_test = DataLoader(
                datasets_test,
                shuffle = False,
                batch_size=batch_size,
                num_workers=0
        )

    #parameters to update
    params_to_update = model_ft.parameters()

    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)


    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    # Setup the loss fxn
    criterion = nn.BCEWithLogitsLoss()
    #criterion = nn.BCELoss()

    if train == True:
        correct = 0
        tot_it = 0
        #TRAIN
        for e in range(num_epochs):
            start_time = time.time()
            for i, (x,y) in enumerate(celeba_loader):
                x = x.to(device)
                y = y.to(device)

                y_out = model_ft(x)

                loss = criterion(y_out,y)

                optimizer_ft.zero_grad()

                loss.backward()
                optimizer_ft.step()

                y_out_sig = torch.sigmoid(y_out)
                result = y_out_sig > 0.5

                correct += (result.float() == y).sum().item()   
                
                tot_it += 1

                if((i + 1) % 100 == 0):
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    print("Elapsed: [{}]; Epochs: [{}/{}]; Iterations: [{}/{}] Loss = {:.3f} Acc = {:.3f}%".format(et,e,num_epochs,i+1,len(celeba_loader), loss.item(), 100*(correct/(len(selected_attrs) * batch_size * tot_it))))
                    start_time = time.time()

            #save the classification model
            torch.save(model_ft.state_dict(), "celeba_classifierALL.pth")

    else:
        model_ft.load_state_dict(torch.load("celeba_classifierALL.pth", map_location=device))
        model_ft.eval()

        G = Generator(None, g_conv_dim, c_dim, g_repeat_num)
        g_optimizer = torch.optim.Adam(G.parameters(), g_lr, [beta1, beta2])
        G.to(device)
        G_path = os.path.join(model_save_dir, '{}-G.ckpt'.format(200000))
        G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        
        correct_attrs = []
        for i in selected_test_attrs:
            correct_attrs.append(0)
        correct = 0
        tot_it = 0

        start_time = time.time()
        for i, (x,y) in enumerate(celeba_loader_test):
                x = x.to(device)
                y = y.to(device)

                if original:
                    x_or = torch.cat([x], dim=3)
                    #print(result_dir+selected_test_attrs[c_ind])
                    result_path = os.path.join(result_original, '{}_original-image.jpg'.format(i))
                    save_image(denorm(x_or.data.cpu()), result_path, nrow=1, padding=0)
                    #print('Saved real and fake images into {}...'.format(result_path))

                c_trg_list = create_labels(y, len(selected_test_attrs), "CelebA", selected_test_attrs)       #la lista delle label selezionate = 5
                if(i == 1394 or i == 1884 or i == 2270 or i == 2723 or i == 2743 or i == 3124 or i == 3241 or i == 3377):
                    print("i: ", i)
                    print(c_trg_list)
                    print()
                for c_ind, c_trg in enumerate(c_trg_list):
                    
                    x_fake = G(x, c_trg)            #genero l'immagine                                    
                    y_out = model_ft(x_fake)        #classifico l'immagine generata    
                    y_out_sig = torch.sigmoid(y_out)
                    result = y_out_sig > 0.5

                    correct += (result[0][selected_attrs.index(selected_test_attrs[c_ind])] == c_trg[0][c_ind]).sum().item()       #accuratezza totale  
                    correct_attrs[c_ind] += (result[0][selected_attrs.index(selected_test_attrs[c_ind])] == c_trg[0][c_ind]).sum().item()       #accuratezza per ogni attributo
                    
                    

                    if(debug and (i+1)%100==0):
                        #print(c_trg)
                        #print(c_trg.size())
                        #print(c_trg[0])
                        print("g_conv_dim: ", g_conv_dim)
                        print("c_dim: ", c_dim)
                        print("g_repeat_num: ", g_repeat_num)
                        print("model_save_dir: ", model_save_dir)
                        print(selected_attrs[selected_attrs.index(selected_test_attrs[c_ind])])
                        print(result[0][selected_attrs.index(selected_test_attrs[c_ind])])
                        print(c_trg[0][c_ind])
                        print(result[0][selected_attrs.index(selected_test_attrs[c_ind])] == c_trg[0][c_ind])
                        print("correct: ", correct)
                        print("len(selected_test_attrs): ", len(selected_test_attrs))
                        print("batch_size: ", batch_size)
                        print("tot_it: ", tot_it)
                        print("Test Acc = {:.3f}%".format( 100*(correct/(len(selected_test_attrs) * batch_size * tot_it))))
                        print()
                        #print(c_trg[0][c_ind].size())

                    if (save_images):
                        # se voglio salvare tutte le immagini in una cartella
                        if(cartella_unica):
                            result_path = os.path.join(result_dir_FID, '{}_{}_{}-image.jpg'.format(i+1,c_ind+1, set_classifer))
                        else:
                            result_path = os.path.join(result_dir+selected_test_attrs[c_ind], '{}_{}-image.jpg'.format(i+1,set_classifer))
                        
                        if(i<10000/c_dim and cartella_unica or not(cartella_unica)):
                            # Save the translated images.
                            x_concat = torch.cat([x_fake], dim=3)
                            save_image(denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)

                tot_it += 1             #stava dentro il for c_trg_list

                if((i + 1)%100 == 0):
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    print("Elapsed: [{}]; Iterations: [{}/{}]".format(et,i+1,len(celeba_loader_test)))
                    start_time = time.time()


        print("Test Acc = {:.3f}%".format( 100*(correct/(len(selected_test_attrs) * batch_size * tot_it))))
        for (i,j) in enumerate(selected_test_attrs):
            print("Test Acc \"{}\" = {:.3f}%".format(j, 100*(correct_attrs[i]/(batch_size * tot_it))))
            # print("Test Acc \"Blond_Hair\" = {:.3f}%".format( 100*(correct_attrs[1]/(batch_size * tot_it))))
            # print("Test Acc \"Brown_Hair\" = {:.3f}%".format( 100*(correct_attrs[2]/(batch_size * tot_it))))
            # print("Test Acc \"Male\" = {:.3f}%".format( 100*(correct_attrs[3]/(batch_size * tot_it))))
            # print("Test Acc \"Young\" = {:.3f}%".format( 100*(correct_attrs[4]/(batch_size * tot_it))))
