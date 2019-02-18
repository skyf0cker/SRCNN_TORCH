<p align='center'> / SRCNN_TORCH /</p>
<p align='center'> SRCNN 的 Pytorch实现版本</p>
<p align='center'> Build the SRCNN with Pytorch</p>
<p align='center'> v 0.1.0</p>

---

### architecture

We refer to the architecture of ESRGAN to build SRCNN.

- dataloader : To load the data
- model : Describe the specific architecture of SRCNN
- train:The main entrance of the program 

### Detail

#### dataloader

- torch.utils.data.Dataset

  ***The most basic class to load data***

  The main job is to implement the following two member functions

  - ```python
    #  *****************************************************************
    #		how to load data each time and how to get the length of data
    #  *****************************************************************
    
    def __getitem__(self, index):
            img_path, label = self.data[index].img_path, self.data[index].label
            img = Image.open(img_path)
    
            return img, label
        
    def __len__(self):
        return len(self.data)
    ```

    

- torch.utils.data.DataLoader

  ***This class encapsulates the Dataset class.***

  - ```python
    # defination of the DataLoader
    class torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=<function default_collate>, pin_memory=False, drop_last=False)
    ```

    As you can see, there are several main parameters:

    - **Dataset** : The custom Dataset class above.
    - **Collate_fn**: This function is used to package the batch, which is described in detail later.
    - **Num_worker**: Very simple multi-threaded method, as long as it is set to >=1, you can multi-thread pre-read data.

    And we should Implement a function named "\__iter__",we will use the DataLoaderIter class in it.

    ```python
    def __iter__(self):
            return DataLoaderIter(self)
    ```

    

- torch.utils.data.dataloader.DataLoaderIter

  **DataLoadIter is the Iterator of the DataLoader**

  ---

  ```python
  from torch.utils.data import Dataset
  import os
  from PIL import Image
  import numpy as np
  import torchvision.transforms as transform
  
  
  class Imgdataset(Dataset):
  
      def __init__(self, path):
          super(Imgdataset, self).__init__()
          self.data = []
          if os.path.exists(path):
              dir_list = os.listdir(path)
              LR_dir = dir_list[0]
              HR_dir = dir_list[1]
              LR_path = path + '/' + LR_dir
              HR_path = path + '/' + HR_dir
              if os.path.exists(LR_path) and os.path.exists(HR_path):
                  LR_data = os.listdir(LR_path)
                  HR_data = os.listdir(HR_path)
                  self.data = [{'LR':LR_path +'/'+ LR_data[i], 'HR':HR_path +'/'+ HR_data[i]} for i in range(len(LR_data))]
              else:
                  raise FileNotFoundError('path doesnt exist!')
  
      def __getitem__(self, index):
  
          LR_path, HR_path = self.data[index]["LR"], self.data[index]["HR"]
          tran = transform.ToTensor()
  
          LR_img = Image.open(LR_path)
          HR_img = Image.open(HR_path)
          # print(tran(img).shape)
  
          return tran(LR_img), tran(HR_img)
  
      def __len__(self):
  
          return len(self.data)
  
  ```


#### Model

- **conv1**: conv2d(in_channels=3, out_channels=64, kernal_size=9, stride=1)
- **conv2**: conv2d(in_channels=64, out_channels=32, kernal_size=3, stride=1)
- **conv3**:conv2d(in_channels=32, out_channels=3, kernal_size=1, stride=1)
- **ReLU**: Activation function

```python
import torch
import torch.nn as nn


class NetWork(nn.Module):

    def __init__(self):
        super(NetWork, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, stride=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, stride=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, stride=1)

    def forward(self, data):
        out = self.conv1(data)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)

        return out
```

**NOTE:To avoid border effects during training, all the convolutional layers have no padding, and the result we get from the conv3 won't have the same size as the input image, it will be a little bit smaller than the origin picture,so we just calcuate the MSELoss by central overlapping area **



#### train

there are a little bit of points that we should note:

- how to load our data?

  > Not to mention what we said above, we just need to instantiate a DataLoader class with our custom DataSet and then iterate over it.
  >
  > AND WE SHOULD NOTE: 
  >
  > ```python
  > for batch_id ,batch in enumerate(train_data_loader):
  >     #
  >     #	batch is a tensor included twenty LR_data and twenty HR_data (e.g.)
  >     #
  > ```

- how to build our net?

  > just instantiate it! 

- how to train our model?

  > we use Adam to optimize our Loss function:MSELoss.
  >
  > ```python
  > optimizer = optim.Adam(net.parameters(), lr=0.1)
  > loss = nn.MSELoss()
  > ```

- how to use cuda?

  > ```python
  > if not torch.cuda.is_available():
  >     raise Exception('NO GPU!')
  > ```



​    

   
