from torchvision.datasets.folder import *
import numpy as np
import torch

class ImageFromFolder(ImageFolder):
    def __init__(self, root, num_data=100000, preprocessing=False, transform=None, target_transform=None,
                 loader=default_loader):

        mag = np.loadtxt(os.path.join(root, 'train_mf.txt'))
        #print(mag[:10], mag.shape)        

        imgs = [(os.path.join(root,'amplified','%06d.png'%(i)),
                 os.path.join(root,'frameA','%06d.png'%(i)),
                 os.path.join(root,'frameB','%06d.png'%(i)),
                 os.path.join(root,'frameC','%06d.png'%(i)),
                 mag[i]) for i in range(num_data)]


        self.root = root
        self.imgs = imgs
        self.samples = self.imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.preproc = preprocessing

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        pathAmp, pathA, pathB, pathC, target = self.samples[index]
        sampleAmp, sampleA, sampleB, sampleC = np.array(self.loader(pathAmp)), np.array(self.loader(pathA)), np.array(self.loader(pathB)), np.array(self.loader(pathC))
      
        # normalize
        sampleAmp = sampleAmp/127.5 - 1.0
        sampleA = sampleA/127.5 - 1.0
        sampleB = sampleB/127.5 - 1.0
        sampleC = sampleC/127.5 - 1.0

        # preprocessing
        if self.preproc:
            sampleAmp = preproc_poisson_noise(sampleAmp)
            sampleA = preproc_poisson_noise(sampleA)
            sampleB = preproc_poisson_noise(sampleB)
            sampleC = preproc_poisson_noise(sampleC)
        """
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        """

        # to torch tensor
        sampleAmp, sampleA, sampleB, sampleC = torch.from_numpy(sampleAmp), torch.from_numpy(sampleA), torch.from_numpy(sampleB), torch.from_numpy(sampleC)
        sampleAmp = sampleAmp.float()
        sampleA = sampleA.float()
        sampleB = sampleB.float()
        sampleC = sampleC.float()

        target = torch.from_numpy(np.array(target)).float()

        # permute from HWC to CHW
        sampleAmp = sampleAmp.permute(2,0,1)
        sampleA = sampleA.permute(2,0,1)
        sampleB = sampleB.permute(2,0,1)
        sampleC = sampleC.permute(2,0,1)

        return sampleAmp, sampleA, sampleB, sampleC, target

def preproc_poisson_noise(image):
    nn = np.random.uniform(0, 0.3) # 0.3
    n = np.random.normal(0.0, 1.0, image.shape)
    n_str = np.sqrt(image + 1.0) / np.sqrt(127.5)
    return image + nn * n * n_str

class ImageFromFolderTest(ImageFolder):
    def __init__(self, root, mag=10.0, mode='static', num_data=300, preprocessing=False, transform=None, target_transform=None, loader=default_loader):
        if mode=='static':
            imgs = [(root+'_%06d.png'%(1),
                     root+'_%06d.png'%(i+2),
                     mag) for i in range(num_data)]
        elif mode=='dynamic':
            imgs = [(root+'_%06d.png'%(i+1),
                     root+'_%06d.png'%(i+2),
                     mag) for i in range(num_data)]
        else:
            raise ValueError("Unsupported modes %s"%(mode))

        self.root = root
        self.imgs = imgs
        self.samples = self.imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.preproc = preprocessing

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        pathA, pathB, target = self.samples[index]
        #print(pathA, pathB, target)
        sampleA, sampleB = np.array(self.loader(pathA)), np.array(self.loader(pathB))
      
        # normalize
        sampleA = sampleA/127.5 - 1.0
        sampleB = sampleB/127.5 - 1.0

        # preprocessing
        if self.preproc:
            sampleA = preproc_poisson_noise(sampleA)
            sampleB = preproc_poisson_noise(sampleB)
        """
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        """

        # to torch tensor
        sampleA, sampleB = torch.from_numpy(sampleA), torch.from_numpy(sampleB)
        sampleA = sampleA.float()
        sampleB = sampleB.float()

        target = torch.from_numpy(np.array(target)).float()

        # permute from HWC to CHW
        sampleA = sampleA.permute(2,0,1)
        sampleB = sampleB.permute(2,0,1)

        return sampleA, sampleB, target

# Test
if __name__ == '__main__':

    dataset = ImageFromFolder('./../data/train', num_data=100, preprocessing=True)

    imageAmp, imageA, imageB, imageC, mag = dataset.__getitem__(0)

    import matplotlib.pyplot as plt
    plt.imshow(((imageA+1.0)*127.5).astype(np.uint8))
    plt.show()