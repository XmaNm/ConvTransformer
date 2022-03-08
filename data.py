from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
from skimage import io,transform

class myData(Dataset):
    def __init__(self, img1_path, img2_path, pimg_path, transform=None):
        self.img1_path = img1_path
        self.img2_path = img2_path
        self.pimg_path = pimg_path
        self.transform = transform
        self.image1 = os.listdir(self.img1_path)
        self.image2 = os.listdir(self.img2_path)
        self.pimage = os.listdir(self.pimg_path)

    def __len__(self):
        return len(self.pimage)

    def __getitem__(self, index):
        image_index =self.pimage[index]
        img1_path = os.path.join(self.img1_path, image_index)
        img2_path = os.path.join(self.img2_path, image_index)
        pimg_path = os.path.join(self.pimg_path, image_index)
        image1 = io.imread(img1_path)
        image1 = np.expand_dims(image1, axis=-1)
        image2 = io.imread(img2_path)
        image2 = np.expand_dims(image2, axis=-1)

        x = np.concatenate((image1, image2), axis=-1)
        pimage = io.imread(pimg_path)
        sample = {'image': x, 'label':pimage}

        if self.transform:
            sample = self.transform(sample)
        return sample

path1 = r"G:\mine\code\ConvTransformerTimeSeries-master\ConvTransformerTimeSeries-master\roi2\1999"
path2 = r"G:\mine\code\ConvTransformerTimeSeries-master\ConvTransformerTimeSeries-master\roi2\2000"
path3 = r"G:\mine\code\ConvTransformerTimeSeries-master\ConvTransformerTimeSeries-master\roi2\2001"

data = myData(path1, path2, path3, transform=None)
dataloader = DataLoader(data, batch_size=4, shuffle=True)
# for i_batch, batch_data in enumerate(dataloader):
#         print(i_batch)#打印batch编号
#         print(batch_data['image'].size())#打印该batch里面图片的大小
#         print(batch_data['label'])#打印该batch里面图片的标签
