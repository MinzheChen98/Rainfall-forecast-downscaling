from __future__ import print_function
import torch
import matplotlib as plt
import argparse
import sys
from torch.utils.data import DataLoader,random_split
from torchvision import datasets, models, transforms
import platform
from datetime import timedelta, date, datetime
from model import vdsr_dem
import torch.nn as nn
import numpy as np
import torch.optim as optim
import os
from math import log10
import time
from PrepareData import ACCESS_BARRA_vdsr_zg
import tqdm
from statsmodels.distributions.empirical_distribution import ECDF
# ===========================================================
# Training settings
# ===========================================================
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
# Hardware specifications
parser.add_argument('--n_threads', type=int, default=0,
                    help='number of threads for data loading')

parser.add_argument('--cpu', action='store_true',help='cpu only?') 

# hyper-parameters
parser.add_argument('--train_name', type=str, default="vdsr_pr", help='training name')

parser.add_argument('--batch_size', type=int, default=11, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=4, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.01')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')

# model configuration
parser.add_argument('--upscale_factor', '-uf',  type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--model', '-m', type=str, default='vdsr', help='choose which model is going to use')

#data
parser.add_argument('--pr', type=bool, default=True,help='add-on pr?')

parser.add_argument('--train_start_time', type=type(datetime(1990,1,25)), default=datetime(1990,1,2),help='r?')
parser.add_argument('--train_end_time', type=type(datetime(1990,1,25)), default=datetime(1990,2,9),help='?')
parser.add_argument('--test_start_time', type=type(datetime(2002,1,1)), default=datetime(2012,1,1),help='a?')
parser.add_argument('--test_end_time', type=type(datetime(2002,12,31)), default=datetime(2012,12,31),help='')

parser.add_argument('--dem', action='store_true',help='add-on dem?') 
parser.add_argument('--psl', action='store_true',help='add-on psl?') 
parser.add_argument('--zg', action='store_true',help='add-on zg?') 
parser.add_argument('--tasmax', action='store_true',help='add-on tasmax?') 
parser.add_argument('--tasmin', action='store_true',help='add-on tasmin?')
parser.add_argument('--leading_time_we_use', type=int,default=1
                    ,help='add-on tasmin?')
parser.add_argument('--ensemble', type=int, default=11,help='total ensambles is 11') 
parser.add_argument('--channels', type=float, default=0,help='channel of data_input must') 
#[111.85, 155.875, -44.35, -9.975]
parser.add_argument('--domain', type=list, default=[112.9, 154.25, -43.7425, -9.0],help='dataset directory')

parser.add_argument('--file_ACCESS_dir', type=str, default="../data/",help='dataset directory')
parser.add_argument('--file_BARRA_dir', type=str, default="../data/barra_aus/",help='dataset directory')
parser.add_argument('--file_DEM_dir', type=str, default="../DEM/",help='dataset directory')
parser.add_argument('--precision', type=str, default='single',choices=('single', 'half','double'),help='FP precision for test (single | half)')

args = parser.parse_args([])


#pr_dem
def write_log(log):
    print(log)
    if not os.path.exists("./save/"+args.train_name+"/"):
        os.mkdir("./save/"+args.train_name+"/")
    my_log_file=open("./save/"+args.train_name + '/train.txt', 'a')
#     log="Train for batch %d,data loading time cost %f s"%(batch,start-time.time())
    my_log_file.write(log + '\n')
    my_log_file.close()
    return




def evaluation(net,val_dataloders,loss,criterion):
    net.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_loss=0
    avg_psnr = 0
    start=time.time()
    with torch.no_grad():
        for batch, (pr,dem,hr) in enumerate(val_dataloders):
            pr,dem,hr = prepare([pr,dem, hr],device)
            sr = net(pr,dem)
            val_loss=criterion(sr, hr)
            test_loss+=val_loss.item()
            psnr = 10 * log10(1000 / (val_loss.item())**2)
            avg_psnr += psnr
        write_log("evalutaion: time cost %f s, test_loss: %f, psnr: avg_psnr %f"%(
                      time.time()-start,
                      test_loss/(batch + 1),
                      avg_psnr / len(val_dataloders)
                 ))
    return test_loss

def prepare( l, device=False):
    def _prepare(tensor):
        if args.precision == 'half': tensor = tensor.half()
        if args.precision == 'single': tensor = tensor.float()
        return tensor.to(device)

    return [_prepare(_l) for _l in l]

# find 50 satation # lat, lon



station_50_index={'CARNAMAH': [128, 28],
 'MULLEWA': [138, 24],
 'WONGAN HILLS': [117, 35],
 'BADGINGARRA RESEARCH STN': [122, 24],
 'MOUNT BARKER': [83, 44],
 'ESPERANCE': [90, 82],
 'BENCUBBIN': [118, 46],
 'KELLERBERRIN': [110, 44],
 'BEVERLEY': [106, 37],
 'CORRIGIN': [104, 46],
 'HYDEN': [103, 55],
 'NARROGIN': [99, 39],
 'ONGERUP': [89, 51],
 'RAVENSTHORPE': [93, 65],
 'SALMON GUMS RES.STN.': [98, 80],
 'CEDUNA AMO': [106, 190],
 'CLEVE': [92, 215],
 'KYANCUTTA': [97, 206],
 'STREAKY BAY': [100, 194],
 'YONGALA': [98, 236],
 'PRICE': [86, 229],
 'WAROOKA': [80, 223],
 'ROSEDALE (TURRETFIELD RESEARCH CENTRE)': [84, 236],
 'MENINGIE': [73, 241],
 'KEITH': [70, 250],
 'SPRINGSURE COMET ST': [179, 320],
 'OAKEY AERO': [149, 354],
 'SURAT': [151, 329],
 'LAKE VICTORIA STORAGE': [88, 258],
 'COLLARENEBRI (ALBERT ST)': [129, 325],
 'BALRANALD (RSL)': [83, 279],
 'PEAK HILL POST OFFICE': [100, 321],
 'CONDOBOLIN AG RESEARCH STN': [97, 313],
 'NYNGAN AIRPORT': [111, 312],
 'TRANGIE RESEARCH STATION AWS': [107, 319],
 'MUNGINDI POST OFFICE': [134, 329],
 'QUIRINDI POST OFFICE': [111, 344],
 'ORANGE AGRICULTURAL INSTITUTE': [95, 329],
 'WAGGA WAGGA AMO': [78, 315],
 'GRENFELL (MANGANESE RD)': [90, 321],
 'COROWA AIRPORT': [71, 305],
 'NARRANDERA AIRPORT AWS': [82, 306],
 'HILLSTON AIRPORT': [93, 297],
 'LAKE CARGELLIGO AIRPORT': [95, 305],
 'MILDURA AIRPORT': [87, 266],
 'OUYEN (POST OFFICE)': [79, 268],
 'WARRACKNABEAL MUSEUM': [68, 269],
 'ECHUCA AERODROME': [69, 290],
 'KERANG': [73, 282],
 'ARARAT PRISON': [59, 274]}


def crps(ensin,obs):
    '''
    @param ensin A vector of prediction
    @param obs  A vector of observations
    
'''

#     assert not np.isnan(ensin).any() and not np.isnan(obs).any(), "data contains nan"
         
    Fn = ECDF(ensin)
    xn=np.sort(np.unique(ensin))
    m=len(xn)
    dn=np.diff(xn)
    eq1=0
    eq2=0
    if(obs>xn[0] and obs<xn[m-1]): #obs在范围内
        k=np.max(np.where(xn<=obs))#小于obs的最大值下标
        x0 = xn[k] #小于obs的最大值
        if k>0:
            eq1=np.sum(Fn(xn[0:k+1])**2*np.append(dn[0:k], obs - xn[k]))#小于obs的所有值 的 百分比数 的平方
        else:
            eq1 =np.sum(Fn(xn[0])**2*(obs - xn[0]))
        if k<m-2:

            eq2=np.sum((1-Fn(xn[k:m-1]))**2*np.append(xn[k+1] - obs, dn[(k+1):(m-1)]))
        else:
            eq2 =np.sum((1-Fn(xn[m-2]))**2*(xn[m-1] - obs))

    if obs <= xn[0]: # 观测值在之外
        eq2 =np.sum(np.append(1, 1-Fn(xn[0:(m-1)]))**2*np.append(xn[0]-obs, dn))
    if obs >= xn[m-1]:
        eq1= np.sum(Fn(xn)**2*np.append(dn, obs - xn[m-1]))
            
    return eq1+eq2 




def vectcrps_v(fct_ens,obs):
    '''
    #' @param fct_ens A 2D prediction
    #' @param obs  A vector of observations
    #' @return a crps vector'''
    score =0

    
    fct_ens=fct_ens
    assert not np.isnan(fct_ens).any() and not np.isnan(obs).any(),"data contains nan"
    for i in range(obs.shape[0]):
#         print(fct_ens[:,i],obs[i])
        score+=crps(fct_ens[:,i],obs[i])
  
    return score


def vectcrps_m(fct_ens,obs):
    '''
    #' @param fct_ens A 2D prediction
    #' @param obs  A vector of observations
    #' @return a crps vector'''
    score =0
    assert not np.isnan(fct_ens).any() and not np.isnan(obs).any(),"data contains nan"
    score_map=np.zeros((obs.shape[0],obs.shape[1]))
    for i in range(obs.shape[0]):
        for j in range(obs.shape[1]):
            score_map[i,j]=crps(fct_ens[:,i,j],obs[i,j])
#             score+=crps(fct_ens[:,i,j],obs[i,j])
    return score_map
    return score/(obs.shape[0]*obs.shape[1])

def vectcrps_m_50_station(fct_ens,obs):
    '''
    #' @param fct_ens A 2D prediction
    #' @param obs  A vector of observations
    #' @return a crps vector'''
    score =0
    assert not np.isnan(fct_ens).any() and not np.isnan(obs).any(),"data contains nan"
    score_map=np.zeros((50))
    for idx,i in enumerate(station_50_index.values()):
        score_map[idx]=crps(fct_ens[:,i[0],i[1]],obs[i[0],i[1]])
#     for i in range(obs.shape[0]):
#         for j in range(obs.shape[1]):
#             score_map[i,j]=crps(fct_ens[:,i,j],obs[i,j])
#             score+=crps(fct_ens[:,i,j],obs[i,j])
    return score_map


def main():

    #     init_date=date(1970, 1, 1)
    #     start_date=date(1990, 1, 2)
    #     end_date=date(2011,12,25)
    sys = platform.system()
    args.dem=True
    args.train_name="pr_dem"
    args.channels=0
    if args.pr:
        args.channels+=1
    if args.zg:
        args.channels+=1
    if args.psl:
        args.channels+=1
    if args.tasmax:
        args.channels+=1
    if args.tasmin:
        args.channels+=1
    if args.dem:
        args.channels+=1
    print("training statistics:")
    print("  ------------------------------")
    print("  trainning name  |  %s"%args.train_name)
    print("  ------------------------------")
    print("  num of channels | %5d"%args.channels)
    print("  ------------------------------")
    print("  num of threads  | %5d"%args.n_threads)
    print("  ------------------------------")
    print("  batch_size     | %5d"%args.batch_size)
    print("  ------------------------------")
    print("  using cpu only | %5d"%args.cpu)

    lr_transforms = transforms.Compose([
        transforms.Resize((316, 376)),
    #     transforms.RandomResizedCrop(IMG_SIZE),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomRotation(30),
        transforms.ToTensor()
    #     transforms.Normalize(IMG_MEAN, IMG_STD)
    ])

    hr_transforms = transforms.Compose([
    #         transforms.Resize((316, 376)),
    #     transforms.RandomResizedCrop(IMG_SIZE),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomRotation(30),
        transforms.ToTensor()
    #     transforms.Normalize(IMG_MEAN, IMG_STD)
    ])

    data_set=ACCESS_BARRA_vdsr_zg(args.test_start_time,args.test_end_time,lr_transform=lr_transforms,hr_transform=hr_transforms,shuffle=False,args=args)



    #     #######################################################################

    test_data=DataLoader(data_set,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                num_workers=args.n_threads,drop_last=True)

    #     #######################################################################


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # net=torch.load("./save/vdsr_pr/best_test.pth")
    # net=torch.load("../data/model/vdsr_pr/best_test.pth")
    net=torch.load("./save/zgpsl/best_test_40.pth",map_location=torch.device('cpu'))['model']




    # #     criterion = nn.MSELoss(size_average=False)
    #     criterion=nn.L1Loss()

    #     optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    # #     optimizer_my = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    #     scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    if torch.cuda.device_count() > 1:
        write_log("!!!!!!!!!!!!!Let's use"+str(torch.cuda.device_count())+"GPUs!")
        net = nn.DataParallel(net,range(torch.cuda.device_count()))
    else:
        write_log("Let's use"+str(torch.cuda.device_count())+"GPUs!")

    net.to(device)
    #     ##############################################
    write_log("start")
    #     max_error=np.inf
    #     val_max_error=np.inf

    #     print(data_set.filename_list)

    # for e in range(args.nEpochs):
    #         loss=0
    for lead in range(1,7):
        args.leading_time_we_use=lead

        data_set=ACCESS_BARRA_vdsr_zg(args.test_start_time,args.test_end_time,lr_transform=lr_transforms,hr_transform=hr_transforms,shuffle=False,args=args)


        test_data=DataLoader(data_set,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                    num_workers=args.n_threads,drop_last=False)


        crps_score_vsdr=[]
        start=time.time()
        fmt = '%Y%m%d'
        save_dir="./save/crps/s0/"
        np.save(save_dir+"lead_time0_50station",crps_score_vsdr)
        test_data=tqdm.tqdm(test_data)
        for batch, (pr,hr,zg,t,en,data_time,idx) in enumerate(test_data):

            pr,zg,hr,t= prepare([pr,zg,hr,t],device)
        #             print(en,data_time,idx)
        #             optimizer.zero_grad()
            with torch.set_grad_enabled(False):

                sr = net(pr,zg,t)
                sr_np=sr.cpu().numpy()
                hr_np=hr.cpu().numpy()
                print(hr_np.shape,sr_np.shape)
                for i in range(args.batch_size//args.ensemble):
                    a=np.squeeze( sr_np[i*args.ensemble:(i+1)*args.ensemble])
                    b=np.squeeze(hr_np[i*args.ensemble])
        #             print(a.shape,b.shape)
        #             skil=vectcrps_m(a,b)
                    skil=vectcrps_m_50_station(a,b)
    #                 print(skil.shape)
                    time_tuple = time.strptime(str(data_time[i*args.ensemble].item()), fmt)
                    year, month, day = time_tuple[:3]
                    a_date = date(year, month, day)
    #                 print(data_time,idx)
        #             np.save("../crps/vsdr_pr/"+(a_date+timedelta(idx[i*args.ensemble].item())).strftime("%Y%m%d")+'_50station',skil)
                    crps_score_vsdr.append(skil)
        np.save(save_dir+"lead_time"+str(lead)+'_50station',crps_score_vsdr)
        print(str(lead)+" : "+str(np.array(crps_score_vsdr).mean()))

            
if __name__=='__main__':
    main()
