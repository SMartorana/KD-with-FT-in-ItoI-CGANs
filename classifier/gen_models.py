from inspect import trace
import os
from model import *
from torch.utils.data import DataLoader
from load_celeba import *
from torchvision import transforms
from torch.functional import Tensor
import torch.jit
from torch.utils.mobile_optimizer import optimize_for_mobile

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

""" def label2onehot(labels, dim):
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    out[np.arange(batch_size), labels.long()] = 1
    return out



def create_labels(c_org, c_dim=5, dataset='CelebA', selected_attrs=None):
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
 """
device = torch.device('cpu')

set_classifer = 'teacher'       #teacher, student, student_FR, student_FR135, student_OUT, student_FR5OUT
if(set_classifer == 'teacher'):
    model_save_dir = 'classifier/generatore/teacher/'           #directory del generatore
    g_conv_dim = 64     #teacher 64, student 32
    g_repeat_num = 6    #teacher 6, student 3
    c_dim = 5
else:
    g_conv_dim = 32     #teacher 64, student 32
    g_repeat_num = 3    #teacher 6, student 3
    c_dim = 5
    if(set_classifer == 'student'):
        model_save_dir = 'classifier/generatore/student/'
    elif (set_classifer == 'student_FR'):
        model_save_dir = 'classifier/generatore/student_FR_10/'                   
    elif (set_classifer == 'student_FR135'):                        #flusso applicato a 3 blocchi residual [1,3,5]
        model_save_dir = 'classifier/generatore/student_FR135/'            
    elif (set_classifer == 'student_OUT'):                        #flusso applicato all'immagine di output
        model_save_dir = 'classifier/generatore/student_OUT/'            
    elif (set_classifer == 'student_FR5OUT'):                        #flusso applicato all'ultimo blocco residual + immagine di output
        model_save_dir = 'classifier/generatore/student_FR5OUT/'            
""" 
image_dir = os.path.abspath("./stargan-student/data/celeba/images")
attr_path = os.path.abspath("./stargan-student/data/celeba/list_attr_celeba.txt")
selected_test_attrs=["Mustache", "Goatee", "Rosy_Cheeks", "Bushy_Eyebrows", "Wearing_Lipstick"]

celeba_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(178),
            transforms.Resize(128),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

datasets_test = CelebASkillato(image_dir, attr_path, selected_test_attrs, celeba_transforms, 'test')
 """
""" celeba_loader_test = DataLoader(
            datasets_test,
            shuffle = True,
            batch_size=1,
            num_workers=0
    ) """

G = Generator(None, g_conv_dim, c_dim, g_repeat_num)
print(model_save_dir)
G_path = os.path.join(model_save_dir, '{}-G.ckpt'.format(200000))
G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))

""" for i, (x,y) in enumerate(celeba_loader_test):

    c_trg_list = create_labels(y, len(selected_test_attrs), "CelebA", selected_test_attrs)       #la lista delle label selezionate = 5

    for c_ind, c_trg in enumerate(c_trg_list): """

x = Tensor(1,3,128,128)
c_trg = denorm(Tensor(1,c_dim))

print("Device:", device)
print("SAVED", set_classifer)
print("torch.__version__: ", torch.__version__)
print("c_trg: ", c_trg)
# print("c_trg.size(): ", c_trg.size())
# print("x: ", x)
# print("x.size(): ", x.size())
# print(Tensor(5).size())
# print(Tensor(1,3,128,128).size())

G_path_pt = os.path.join(model_save_dir, '{}.pt'.format(set_classifer))
G_path_ptl = os.path.join(model_save_dir, '{}.ptl'.format(set_classifer))

# Alternativa 1

# G = torch.quantization.convert(G)
print("Generatore ok.")
#traced_script_module = torch.jit.trace(G, (denorm(x.data), denorm(c_trg.data)))
traced_script_module = torch.jit.trace(G, (x, c_trg))
print("jit.trace ok.")
# print(traced_script_module)
#torch.jit.save(traced_script_module, G_path_pt)
traced_script_module = optimize_for_mobile(traced_script_module)
print("optimize_for_mobile ok.")
# traced_script_module.save(G_path_pt)
traced_script_module._save_for_lite_interpreter(G_path_ptl)
print("_save_for_lite_interpreter ok.")
# Alternativa 2
""" quantized_torchmodel = torch.jit.script(G)
quantized_torchmodel = optimize_for_mobile(quantized_torchmodel)
#torch.jit.save(quantized_torchmodel, G_path_pt)                     # com.facebook.jni.CppException: PytorchStreamReader failed locating file bytecode.pkl: file not found ()
quantized_torchmodel._save_for_lite_interpreter(G_path_ptl) """