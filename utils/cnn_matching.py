import logging
import os.path
import random
from collections import OrderedDict
from typing import List, Iterable, Tuple, BinaryIO, Union

import matplotlib.pyplot as plt
import numpy as np
import skimage
import skimage.color
import skimage.feature
import skimage.io
import skimage.transform
import sklearn.neighbors
import torch
import torch.nn.functional as F
from numpy.typing import ArrayLike
from torch import nn
from torchvision.models import resnet18, wide_resnet50_2
# device setup
from typing.io import IO

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MatchMeMaker(nn.Module):
    arch = 'resnet18'
    t_d = 448
    d = 100
    max_feature_dist = 0.85

    def __init__(self):
        super().__init__()
        self.feature_extractor = self.init_feature_extractor()
        self.feature_extractor.to(device)
        self.feature_extractor.eval()

        random.seed(1024)
        torch.manual_seed(1024)
        if use_cuda:
            torch.cuda.manual_seed_all(1024)

        self.dim_idxs = torch.tensor(random.sample(range(0, self.t_d), self.d))

        # set model's intermediate outputs
        self.outputs = []

        def hook(module, input, output):
            self.outputs.append(output)

        self.feature_extractor.layer1[-1].register_forward_hook(hook)
        self.feature_extractor.layer2[-1].register_forward_hook(hook)
        self.feature_extractor.layer3[-1].register_forward_hook(hook)

        self.images = []
        self.lengths = []
        self.borders = np.array(tuple(), dtype=int)
        self.embeddings = []
        self.keypoints = []
        self.kd_tree = None

    def save(self, file: Union[str, os.PathLike, BinaryIO, IO[bytes]]):
        state = {
            'feature_extractor': self.feature_extractor.state_dict(),
            'images': self.images,
            'lengths': self.lengths,
            'borders': self.borders,
            'embeddings': self.embeddings,
            'keypoints': self.keypoints,
            'kd_tree': self.kd_tree,
        }
        torch.save(state, file)

    def load_state_dict(self, state_dict: 'OrderedDict[str, torch.Tensor]',
                        strict: bool = True):
        self.feature_extractor.load_state_dict(state_dict['feature_extractor'])
        self.images = state_dict['images']
        self.lengths = state_dict['lengths']
        self.borders = state_dict['borders']
        self.embeddings = state_dict['embeddings']
        self.keypoints = state_dict['keypoints']
        self.kd_tree = state_dict['kd_tree']

    def rebuild_kd_tree(self):
        self.borders = np.cumsum(self.lengths)
        self.kd_tree = sklearn.neighbors.KDTree(np.concatenate(self.embeddings))

    def init_feature_extractor(self):
        if self.arch == 'resnet18':
            return resnet18(pretrained=True, progress=True)
        elif self.arch == 'wide_resnet50_2':
            return wide_resnet50_2(pretrained=True, progress=True)

    @staticmethod
    def embedding_concat(x, y):
        B, C1, H1, W1 = x.size()
        _, C2, H2, W2 = y.size()
        s = int(H1 / H2)
        x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
        x = x.view(B, C1, -1, H2, W2)
        z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
        for i in range(x.size(2)):
            z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
        z = z.view(B, -1, H2 * W2)
        z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

        return z

    def extract_all_features(self, images: np.array) -> np.array:
        """
        Extracts the feature vector from all images.

        :param images: the images as (B, H, W, C) numpy array
        :return: the vectors as (B/2*H/2, d) array
        """
        outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
        # forward pass
        image = torch.as_tensor(images).permute((0, 3, 1, 2)).to(dtype=torch.float)
        self.outputs = []
        with torch.no_grad():
            _ = self.feature_extractor(image.to(next(self.feature_extractor.parameters()).device))
        # get intermediate layer outputs
        for k, v in zip(outputs.keys(), self.outputs):
            outputs[k].append(v.cpu().detach())

        # Merge features into feature vector
        train_outputs_concat = OrderedDict()
        for k, v in outputs.items():
            train_outputs_concat[k] = torch.cat(v, 0)
        embedding_vectors = train_outputs_concat['layer1']
        for layer_name in ['layer2', 'layer3']:
            embedding_vectors = self.embedding_concat(embedding_vectors, train_outputs_concat[layer_name])

        # randomly select d dimension
        embedding_vectors = torch.index_select(embedding_vectors, 1, self.dim_idxs)
        return embedding_vectors.numpy()

    def get_interesting_embeddings(self, images: ArrayLike) -> Tuple[List[np.array], List[np.array]]:
        """
        Gets the embeddings at interest points for each image in the input.

        :param images: the images as (B, H, W, C) numpy array
        :return: a tuple of
            list of embeddings with shape (N_i, d) for i = 1..B
            list of keypoints with shape (N_i, 2) for i = 1..B
        """
        images = np.asarray(images)
        assert len(images.shape) == 4
        keypoints = []
        embeddings = []
        for image in images:
            logger.debug('Finding harris corners.')
            harris_response = skimage.feature.corner_harris(skimage.color.rgb2gray(image), method='k', k=0.05,
                                                            eps=1e-06, sigma=1)
            interest_points = skimage.feature.corner_peaks(harris_response, threshold_rel=0.001, min_distance=5)
            logger.debug('Extracting all embeddings.')
            all_embeddings = self.extract_all_features([image])
            interesting_embeddings = all_embeddings[:, :, interest_points[:, 0] // 4, interest_points[:, 1] // 4]
            # Get features at interest points
            interesting_embeddings = interesting_embeddings.transpose((0, 2, 1))[0]
            keypoints.append(interest_points)
            embeddings.append(interesting_embeddings)
        return keypoints, embeddings

    def add_images_to_db(self, images: Iterable[np.array]):
        for i, image in enumerate(images):
            self.images.append(image)
            logger.info("Getting interest points of {i}th image.")
            keypoints, embeddings = self.get_interesting_embeddings([image])
            self.embeddings += embeddings
            self.keypoints += keypoints
            self.lengths.append([len(im_embeddings) for im_embeddings in embeddings])
        logger.info("Rebuilding KD Tree")
        self.rebuild_kd_tree()

    def count_matches_per_image(self, indices):
        return np.unique(np.digitize(indices, self.borders), return_counts=True)

    def query_images(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Queries the database to find the best image overlap.
        Returns keypoints and match indices.

        :param image: the image as (H, W, C) numpy array, for which a match should be searched
        :return: tuple of:
            matched image as array of shape (H, W, C)
            keypoints in queried image as array of shape (N_q, 2)
            keypoints in queried image as array of shape (N_p, 2)
            match indices as uint-array of shape (M, 2)
        """
        logger.info("Extracting features from test image.")
        query_keypoints, query_features = self.get_interesting_embeddings([image])
        query_keypoints, query_features = query_keypoints[0], query_features[0]
        logger.info("Querying KD-Tree.")
        dists, indices = self.kd_tree.query(query_features, k=1, return_distance=True)
        dists, indices = dists.squeeze(), indices.squeeze()
        logger.info("Searching best image match.")
        image_idx, counts = self.count_matches_per_image(indices)
        image_idx = counts.argmax()
        # Select indices from matched images
        lower = 0 if image_idx == 0 else self.borders[image_idx - 1]
        upper = self.borders[image_idx]
        # Only take keypoints of matched image
        matches_mask = np.logical_and(lower <= indices, indices < upper)
        query_indices = np.argwhere(matches_mask).squeeze()
        match_indices = indices[matches_mask] - self.borders[image_idx]
        dists = dists[matches_mask]
        plt.hist(dists, bins=100)
        plt.show()
        # Filter out matches with high distance
        dists_mask = dists < self.max_feature_dist
        query_indices = query_indices[dists_mask]
        match_indices = match_indices[dists_mask]
        matches12 = np.stack([query_indices, match_indices], axis=1)
        return self.images[image_idx], query_keypoints, self.keypoints[image_idx], matches12


def make_divideable(image, div_factor=16):
    H, W = image.shape[:2]
    W = (W // div_factor) * div_factor
    H = (H // div_factor) * div_factor
    return skimage.transform.resize(image, output_shape=(H, W))


def prepare_image(path):
    image = skimage.io.imread(path)
    image = skimage.transform.rescale(image, [0.25, 0.25, 1.])
    image = make_divideable(image)
    return image


def train() -> MatchMeMaker:
    train_path = os.path.expanduser('~/data/photogrammetry/Uttenhofen Bach_inputs/DJI_0054.JPG')

    use_saved = False
    save_path = './model.pkl'
    if use_saved:
        mmm = MatchMeMaker()
        mmm.load_state_dict(torch.load(save_path))
    else:
        mmm = MatchMeMaker()
        mmm.add_images_to_db([prepare_image(train_path)])
        mmm.save(save_path)
    return mmm


def main():
    test_path = os.path.expanduser('~/data/photogrammetry/Uttenhofen Bach_inputs/DJI_0058.JPG')
    mmm = train()

    test_image = prepare_image(test_path)
    match_image, keypoints1, keypoints2, matches12 = mmm.query_images(test_image)
    matched_keypoints_1 = keypoints1[matches12[:, 0]]
    matched_keypoints_2 = keypoints2[matches12[:, 1]]
    num_matches = matched_keypoints_1.shape[0]
    aligned_matches = np.asarray([np.arange(num_matches), np.arange(num_matches)]).T
    #aligned_matches = np.array([])
    aligned_matches = aligned_matches[:20]

    plt.imshow(test_image)
    plt.show()
    plt.imshow(match_image)
    plt.show()

    fig, axes = plt.subplots()
    # skimage.feature.plot_matches(axes, test_image, match_image, keypoints1, keypoints2, matches12)
    skimage.feature.plot_matches(axes, test_image, match_image, matched_keypoints_1, matched_keypoints_2, aligned_matches)
    fig.show()
    print("finish")


if __name__ == '__main__':
    main()