import os
import data_processing_tool as dpt
from datetime import timedelta, date, datetime
# import args_parameter as args
import torch,torchvision
import numpy as np
import random

from torch.utils.data import Dataset,random_split
from torchvision import datasets, models, transforms

import time
import xarray as xr
from PIL import Image
# from sklearn.model_selection import StratifiedShuffleSplit

# file_ACCESS_dir="/g/data/ub7/access-s1/hc/raw_model/atmos/pr/daily/"
# file_BARRA_dir="/g/data/ma05/BARRA_R/analysis/acum_proc"

# ensemble_access=['e01','e02','e03','e04','e05','e06','e07','e08','e09','e10','e11']
# ensemble=[]
# for i in range(args.ensemble):
#     ensemble.append(ensemble_access[i])
    
# ensemble=['e01','e02','e03','e04','e05','e06','e07','e08','e09','e10','e11']

# leading_time=217
# leading_time_we_use=31


# init_date=date(1970, 1, 1)
# start_date=date(1990, 1, 1)
# end_date=date(1990,12,31) #if 929 is true we should substract 1 day
# dates=[start_date + timedelta(x) for x in range((end_date - start_date).days + 1)]

# domain = [111.975, 156.275, -44.525, -9.975]

# domain = [111.975, 156.275, -44.525, -9.975]

class ACCESS_BARRA_vdsr(Dataset):
    '''

2.using my net to train one channel to one channel.
   
    '''
    def __init__(self,start_date=date(1990, 1, 1),end_date=date(1990,12 , 31),regin="AUS",lr_transform=None,hr_transform=None,shuffle=True,args=None):
        print("=> BARRA_R & ACCESS_S1 loading")
        print("=> from "+start_date.strftime("%Y/%m/%d")+" to "+end_date.strftime("%Y/%m/%d")+"")
        self.file_BARRA_dir = args.file_BARRA_dir
        self.file_ACCESS_dir = args.file_ACCESS_dir
        self.args=args
        
        self.lr_transform = lr_transform
        self.hr_transform = hr_transform

        self.start_date = start_date
        self.end_date = end_date
        
        self.regin = regin
        self.leading_time_we_use=args.leading_time_we_use

        self.ensemble_access=['e01','e02','e03','e04','e05','e06','e07','e08','e09','e10','e11']
        self.ensemble=[]
        for i in range(args.ensemble):
            self.ensemble.append(self.ensemble_access[i])
                
        self.dates = self.date_range(start_date, end_date)
        
        
        self.filename_list=self.get_filename_with_time_order(args.file_ACCESS_dir+"pr/daily/")
        if not os.path.exists(args.file_ACCESS_dir+"pr/daily/"):
            print(args.file_ACCESS_dir+"pr/daily/")
            print("no file or no permission")
        
        
        _,_,date_for_BARRA,time_leading=self.filename_list[0]
        if shuffle:
            random.shuffle(self.filename_list)
        
        
        if not os.path.exists("/g/data/ma05/BARRA_R/v1/forecast/spec/accum_prcp/1990/01/accum_prcp-fc-spec-PT1H-BARRA_R-v1-19900109T0600Z.sub.nc"):
            print(self.file_BARRA_dir)
            print("no file or no permission!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        data_high=dpt.read_barra_data_fc_get_lat_lon(self.file_BARRA_dir,date_for_BARRA)
        self.lat=data_high[1]
        self.lon=data_high[1]
        self.shape=(79,94)
        if self.args.dem:
            data_dem=dpt.add_lat_lon( dpt.read_dem(args.file_DEM_dir+"dem-9s1.tif"))
            self.dem_data=dpt.interp_tensor_2d(dpt.map_aust_old(data_dem,xrarray=False) ,self.shape )
        

        
    def __len__(self):
        return len(self.filename_list)
    

    def date_range(self,start_date, end_date):
        """This function takes a start date and an end date as datetime date objects.
        It returns a list of dates for each date in order starting at the first date and ending with the last date"""
        return [start_date + timedelta(x) for x in range((end_date - start_date).days + 1)]

    
    def get_filename_with_no_time_order(self,rootdir):
        '''get filename first and generate label '''
        _files = []
        list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
        for i in range(0,len(list)):
            path = os.path.join(rootdir,list[i])
            if os.path.isdir(path):
                _files.extend(self.get_filename_with_no_time_order(path))
            if os.path.isfile(path):
                if path[-3:]==".nc":
                    _files.append(path)
        return _files
    
    def get_filename_with_time_order(self,rootdir):
        '''get filename first and generate label ,one different w'''
        _files = []
        for en in self.ensemble:
            for date in self.dates:

#                 filename="da_pr_"+date.strftime("%Y%m%d")+"_"+en+".nc"cd
                access_path=rootdir+en+"/"+"da_pr_"+date.strftime("%Y%m%d")+"_"+en+".nc"
#                 print(access_path)
                if os.path.exists(access_path):
                    for i in range(self.leading_time_we_use):
                        if date==self.end_date and i==1:
                            break
                        path=[]
                        path.append(en)
                        barra_date=date+timedelta(i)
                        path.append(date)
                        path.append(barra_date)
                        path.append(i)
                        _files.append(path)
    
    #最后去掉第一行，然后shuffle
        return _files

    def mapping(self,X,min_val=0.,max_val=255.):
        Xmin = np.min(X)
        Xmax = np.max(X)
        #将数据映射到[-1,1]区间 即a=-1，b=1
        a = min_val
        b = max_val
        Y = a + (b-a)/(Xmax-Xmin)*(X-Xmin)
        return Y
        
    def __getitem__(self, idx):
        '''
        from filename idx get id
        return lr,hr
        '''
        t=time.time()
        
        #read_data filemame[idx]
        en,access_date,barra_date,time_leading=self.filename_list[idx]
        

        lr=dpt.read_access_data(self.file_ACCESS_dir,en,access_date,time_leading,"pr")

#         lr=np.expand_dims(lr,axis=2)

#         lr=np.expand_dims(self.mapping(lr),axis=2)
        label=dpt.read_barra_data_fc(self.file_BARRA_dir,barra_date)

        if self.args.psl:
            lr_zg=np.expand_dims(dpt.read_access_data(self.file_ACCESS_dir,en,access_date,time_leading,"zg"),axis=2)

        if self.args.psl:
            lr_psl=dpt.read_access_data(self.file_ACCESS_dir,en,access_date,time_leading,"psl")

        if self.args.tasmax:
            lr_tasmax=np.expand_dims(dpt.read_access_data(self.file_ACCESS_dir,en,access_date,time_leading,"tasmax"),axis=2)


        if self.args.tasmin:
            lr_tasmin=dpt.read_access_data(self.file_ACCESS_dir,en,access_date,time_leading,"tasmin")
            
#         if self.args.channels==1:
#             lr=np.repeat(lr,3,axis=2)
        return self.lr_transform(Image.fromarray(lr)),self.hr_transform(Image.fromarray(label)),torch.tensor(int(barra_date.strftime("%Y%m%d"))),torch.tensor(time_leading)

class ACCESS_BARRA_vdsr_zg(Dataset):
    '''

2.using my net to train one channel to one channel.
   
    '''
    def __init__(self,start_date=date(1990, 1, 1),end_date=date(1990,12 , 31),regin="AUS",lr_transform=None,hr_transform=None,shuffle=True,args=None):
        print("=> BARRA_R & ACCESS_S1 loading")
        print("=> from "+start_date.strftime("%Y/%m/%d")+" to "+end_date.strftime("%Y/%m/%d")+"")
        self.file_BARRA_dir = args.file_BARRA_dir
        self.file_ACCESS_dir = args.file_ACCESS_dir
        self.args=args
        
        self.lr_transform = lr_transform
        self.hr_transform = hr_transform

        self.start_date = start_date
        self.end_date = end_date
        
        self.regin = regin
        self.leading_time_we_use=args.leading_time_we_use

        self.ensemble_access=['e01','e02','e03','e04','e05','e06','e07','e08','e09','e10','e11']
        self.ensemble=[]
        for i in range(args.ensemble):
            self.ensemble.append(self.ensemble_access[i])
                
        self.dates = self.date_range(start_date, end_date)
        
        
        self.filename_list=self.get_filename_with_time_order(args.file_ACCESS_dir+"pr/daily/")
        if not os.path.exists(args.file_ACCESS_dir+"pr/daily/"):
            print(args.file_ACCESS_dir+"pr/daily/")
            print("no file or no permission")
        
        
        _,_,date_for_BARRA,time_leading=self.filename_list[0]
        if shuffle:
            random.shuffle(self.filename_list)
        
        
        if not os.path.exists("/g/data/ma05/BARRA_R/v1/forecast/spec/accum_prcp/1990/01/accum_prcp-fc-spec-PT1H-BARRA_R-v1-19900109T0600Z.sub.nc"):
            print(self.file_BARRA_dir)
            print("no file or no permission!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        data_high=dpt.read_barra_data_fc_get_lat_lon(self.file_BARRA_dir,date_for_BARRA)
        self.lat=data_high[1]
        self.lon=data_high[1]
        self.shape=(79,94)
#        if self.args.dem:
#        data_dem,la,lo=dpt.add_lat_lon( dpt.read_dem(args.file_DEM_dir+"dem-9s1.tif"),xarray=False)
#        self.dem_data=dpt.interp_tensor_2d(dpt.map_aust_old(data_dem,la,lo,xrarray=False) ,self.shape)
        

        
    def __len__(self):
        return len(self.filename_list)
    

    def date_range(self,start_date, end_date):
        """This function takes a start date and an end date as datetime date objects.
        It returns a list of dates for each date in order starting at the first date and ending with the last date"""
        return [start_date + timedelta(x) for x in range((end_date - start_date).days + 1)]

    
    def get_filename_with_no_time_order(self,rootdir):
        '''get filename first and generate label '''
        _files = []
        list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
        for i in range(0,len(list)):
            path = os.path.join(rootdir,list[i])
            if os.path.isdir(path):
                _files.extend(self.get_filename_with_no_time_order(path))
            if os.path.isfile(path):
                if path[-3:]==".nc":
                    _files.append(path)
        return _files
    
    def get_filename_with_time_order(self,rootdir):
        '''get filename first and generate label ,one different w'''
        _files = []
        for en in self.ensemble:
            for date in self.dates:

#                 filename="da_pr_"+date.strftime("%Y%m%d")+"_"+en+".nc"cd
                access_path=rootdir+en+"/"+"da_pr_"+date.strftime("%Y%m%d")+"_"+en+".nc"
#                 print(access_path)
                if os.path.exists(access_path):
                    for i in range(self.leading_time_we_use):
                        if date==self.end_date and i==1:
                            break
                        path=[]
                        path.append(en)
                        barra_date=date+timedelta(i)
                        path.append(date)
                        path.append(barra_date)
                        path.append(i)
                        _files.append(path)
    
    #最后去掉第一行，然后shuffle
        return _files

    def mapping(self,X,min_val=0.,max_val=255.):
        Xmin = np.min(X)
        Xmax = np.max(X)
        #将数据映射到[-1,1]区间 即a=-1，b=1
        a = min_val
        b = max_val
        Y = a + (b-a)/(Xmax-Xmin)*(X-Xmin)
        return Y
    
    def to01(self,lr):
        return (lr-np.min(lr))/(np.max(lr)-np.min(lr))
    
    def __getitem__(self, idx):
        '''
        from filename idx get id
        return lr,hr
        '''
        t=time.time()
        
        #read_data filemame[idx]
        en,access_date,barra_date,time_leading=self.filename_list[idx]
        

        lr=dpt.read_access_data(self.file_ACCESS_dir,en,access_date,time_leading,"pr")
#        self.dem_data=self.to01(self.dem_data)*np.max(lr)
#         lr=np.expand_dims(lr,axis=2)

#         lr=np.expand_dims(self.mapping(lr),axis=2)
        label=dpt.read_barra_data_fc(self.file_BARRA_dir,barra_date)

        lr_zg=dpt.read_access_data(self.file_ACCESS_dir,en,access_date,time_leading,"zg")
        lr_zg=self.to01(lr_zg)
        lr_zg=lr_zg*np.max(lr)
        if self.args.psl:
            lr_psl=dpt.read_access_data(self.file_ACCESS_dir,en,access_date,time_leading,"psl")

        #lr_tasmax=dpt.read_access_data(self.file_ACCESS_dir,en,access_date,time_leading,"tasmax")


        #lr_tasmin=dpt.read_access_data(self.file_ACCESS_dir,en,access_date,time_leading,"tasmin")
        lr_t=dpt.read_access_data(self.file_ACCESS_dir,en,access_date,time_leading,"psl")
        lr_t=self.to01(lr_t)
        lr_t=lr_t*np.max(lr)
#         if self.args.channels==1:
#             lr=np.repeat(lr,3,axis=2)
        return self.lr_transform(Image.fromarray(lr)),self.hr_transform(Image.fromarray(label)),self.lr_transform(Image.fromarray(lr_zg)),self.lr_transform(Image.fromarray(lr_t)),torch.tensor(int(en[1:])),torch.tensor(int(barra_date.strftime("%Y%m%d"))),torch.tensor(time_leading)
