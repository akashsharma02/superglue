import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from base import BaseDataLoader

from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.distance import cdist


class AidtrDataLoader(BaseDataLoader):
    """
    Load AIDTR dataset using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, resize=[240, 320], num_features=1024, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir
        self.num_features = num_features
        self.resize = resize
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        if self.resize is not None:
            self.transform.Compose([
                self.transform.Resize(self.resize)
            ])

        self.dataset = AidtrDataset(
            self.data_dir, self.num_features, train=training, transform=self.transform)
        super.__init__(self.dataset, batch_size, shuffle,
                       validation_split, num_workers)


class AidtrDataset(Dataset):
    """
    Sample ground correspondences from AIDTR dataset
    NPY files sourced from data_dir contain finetuned superpoint keypoints and descriptors

    """

    def __init__(self, data_dir, train, num_features, transform):

        self.files = []
        self.folder = folder + '_data_rs/'
        self.num_features = num_features
        self.transform = transform
        if train:
            self.data_dir = Path(data_dir) / Path('train_data')
        else:
            self.data_dir = Path(data_dir) / Path('test_data')
        self.data += [data_file for data_file in self.data_file.iterdir()]

        self.matcher = cv2.BFMatcher_create(cv2.NORM_L1, crossCheck=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        data = np.load(self.data[idx], allow_pickle=True)
        filename = self.files[idx]

        kp0 = data[0].squeeze()
        kp1 = data[1].squeeze()
        desc0 = data[2].squeeze()
        desc1 = data[3].squeeze()
        image0 = data[4].astype(np.float64)
        image1 = data[5].astype(np.float64)
        homography = data[6]

        width, height = image0.shape[0], image0.shape[1]

        assert image0.shape == image1.shape, "Incompatible image sized in Dataset"
        assert kp0.shape[1] == kp1.shape[1], "Incompatible keypoint shapes"
        assert kp0.shape[0] == desc0.shape[0]
        assert kp1.shape[0] == desc0.shape[0]

        if self.num_features > 0:
            kp0_num = min(self.num_features, len(kp0))
            kp1_num = min(self.num_features, len(kp1))
            kp0 = kp0[:kp0_num]
            kp1 = kp1[:kp1_num]

        # Coordinates may have 3 dimensions (x, y, score)
        kp0_np = np.array([(kp[0], kp[1]) for kp in kp0])
        kp1_np = np.array([(kp[0], kp[1]) for kp in kp1])
        scores0_np = np.array([kp[2] for kp in kp0]).astype(np.float64)
        scores1_np = np.array([kp[2] for kp in kp1]).astype(np.float64)

        if len(kp0) < 1 or len(kp1) < 1:
            return{
                'keypoints0': torch.zeros([0, 0, 2], dtype=torch.double),
                'keypoints1': torch.zeros([0, 0, 2], dtype=torch.double),
                'descriptors0': torch.zeros([0, 2], dtype=torch.double),
                'descriptors1': torch.zeros([0, 2], dtype=torch.double),
                'image0': image0,
                'image1': image1,
                'filename': filename
            }
        matched = self.matcher.match(desc0, desc1)

        kp1_projected = cv2.perspectiveTransform(
            kp0_np.reshape((1, -1, 2)), homography)[0, :, :]  # why [0, :, :]

        dists = cdist(kp1_projected, kp1_np)

        min1 = np.argmin(dists, axis=0)
        min2 = np.argmin(dists, axis=1)

        min1v = np.min(dists, axis=1)
        min1f = min2[min1v < 3]

        xx = np.where(min2[min1] == np.arange(min1.shape[0]))[0]
        matches = np.intersect1d(min1f, xx)

        missing1 = np.setdiff1d(np.arange(kp0_np.shape[0]), min1[matches])
        missing2 = np.setdiff1d(np.arange(kp1_np.shape[0]), matches)

        visualize = False
        if visualize:
            matches_dmatch = []
            for idx in range(matches.shape[0]):
                dmatch = cv2.DMatch(matches[idx], min2[matches[idx]], 0.0)
                print(
                    "Match {matches[idx]} {min2[matches[idx]]} dist={dists[matches[idx], min2[matches[idx]]]}")
                matches_dmatch.append(dmatch)
            out = cv2.drawMatches(image0, kp1, image1,
                                  kp2, matches_dmatch, None)
            cv2.imshow('a', out)
            cv2.waitKey(0)

        MN = np.concatenate(
            [min1[matches][np.newaxis, :], matches[np.newaxis, :]])
        MN2 = np.concatenate([missing1[np.newaxis, :], (len(
            kp2)) * np.ones((1, len(missing1)), dtype=np.int64)])
        MN3 = np.concatenate(
            [(len(kp1)) * np.ones((1, len(missing2)), dtype=np.int64), missing2[np.newaxis, :]])
        target_matches = np.concatenate([MN, MN2, MN3], axis=1)

        kp0_np = kp0_np.reshape((1, -1, 2))
        kp1_np = kp1_np.reshape((1, -1, 2))
        desc0 = np.transpose(desc0)
        desc1 = np.transpose(desc1)

        if self.transform is not None:
            image0 = self.transform(image0)
            image1 = self.transform(image1)

        return{
            'keypoints0': list(kp0_np),
            'keypoints1': list(kp1_np),
            'descriptors0': list(desc0),
            'descriptors1': list(desc1),
            'scores0': list(scores0_np),
            'scores1': list(scores1_np),
            'image0': image0,
            'image1': image1,
            'target_matches': list(target_matches),
            'filename': filename
        }
