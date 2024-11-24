from monai.transforms import (
    Compose,
    LoadImaged,
    Resized,
    MapTransform,
    ToNumpyd,
    ToTensord,
    ScaleIntensityd,
    ToTensord
    )
import numpy as np
import cv2
import torch

class Otsud(MapTransform):
    def __init__(self, keys, output_key = None, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.output_key = output_key
        
    def otsu_thresholding(self, input):
        input = input[0]
        _, thresh = cv2.threshold(input, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = np.expand_dims(thresh, 0)
        return thresh
    
    def __call__(self, data):
        for key in self.key_iterator(data):
            if self.output_key is None:
                self.output_key = key
            data[self.output_key] = self.otsu_thresholding(data[key])
        return data
    
class IterativeWatershedd(MapTransform):
    def __init__(self, keys, positive_key, negative_key, iterations = 1, output_key = None, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.output_key = output_key
        self.positive_key = positive_key
        self.negative_key = negative_key
        self.iterations = iterations
        
    def watershed(self, image, pos_marker, neg_marker, iterations):
        im = image[0].astype(np.uint8)
        markers = pos_marker[0].astype(int) + neg_marker[0].astype(int)
        labels = cv2.watershed(np.stack([im,im,im],-1), markers)
        for _ in range(iterations):
            markers=np.zeros_like(markers)
            markers[labels==3]=3
            markers[labels==2]=2
            markers[neg_marker[0]==1]=1
            labels=cv2.watershed(np.stack([im,im,im],-1), markers)
        labels[labels==1]=0
        labels[labels==-1]=0
        labels = np.expand_dims(labels.astype(np.uint8),0)
        return labels
    def __call__(self, data):
        for key in self.key_iterator(data):
            if self.output_key is None:
                self.output_key = key
            data[self.output_key] = self.watershed(data[key], data[self.positive_key], data[self.negative_key], self.iterations)
        return data

class MorphologicalErosiond(MapTransform):
    def __init__(self, keys, kernel_size=(3, 3), iterations=1, output_key = None, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.kernel_size = kernel_size
        self.iterations = iterations
        self.output_key = output_key
        
    def perform_erosion(self, input):
        input = input[0]
        kernel = np.ones(self.kernel_size, np.uint8)
        erosion = cv2.erode(input, kernel, iterations=self.iterations)
        erosion = np.expand_dims(erosion, 0)
        return erosion
    
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            if self.output_key is None:
                self.output_key = key
            d[self.output_key] = self.perform_erosion(d[key])
        return d


class MorphologicalDilationd(MapTransform):
    def __init__(self, keys, kernel_size=(3, 3), iterations=1, output_key = None, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.kernel_size = kernel_size
        self.iterations = iterations
        self.output_key = output_key
        
    def perform_dilation(self, input):
        input = input[0]
        kernel = np.ones(self.kernel_size, np.uint8)
        dilation = cv2.dilate(input, kernel, iterations=self.iterations)
        dilation = np.expand_dims(dilation, 0)
        return dilation
    
    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            if self.output_key is None:
                self.output_key = key
            d[self.output_key] = self.perform_dilation(d[key])
        return d
    
import numpy as np
import cv2
from monai.transforms import MapTransform

class BlackEdgeRemovald(MapTransform):
    def __init__(self, keys, output_key=None, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.output_key = output_key

    def remove_black_edges(self, input_image):
        # Create a copy of the input image to avoid modifying the original
        processed_image = input_image.clone()

        # Create a mask that is True where any channel has intensity close to zero
        combined_mask = (processed_image < 0.03).any(axis=0)
        
        # Set the pixels to the maximum value across all channels where combined_mask is True
        for c in range(processed_image.shape[0]):
            processed_image[c][combined_mask] = processed_image[c].max()  # Set to the max of the current channel
        
        # Convert to NumPy array for morphological operation
        processed_image_np = processed_image.numpy()
        
        # Define a kernel for the morphological closing
        kernel = np.ones((7, 7), np.uint8)  # Adjust kernel size as needed
        
        # Apply closing to each channel
        for c in range(processed_image_np.shape[0]):
            processed_image_np[c] = cv2.morphologyEx(processed_image_np[c], cv2.MORPH_OPEN, kernel)

        # Convert back to the original format if needed
        processed_image = torch.from_numpy(processed_image_np)
        
        return processed_image


    def __call__(self, data):
        for key in self.key_iterator(data):
            # Set the output key based on the original key if no output_key is specified
            output_key = self.output_key or key
            data[output_key] = self.remove_black_edges(data[key])
        return data


    
class KMeansSegmentd(MapTransform):
    def __init__(self, keys, num_clusters=2, output_key='kmeans', mask_key=None, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.num_clusters = num_clusters
        self.output_key = output_key
        self.mask_key = mask_key
    
    def segment(self, input, mask=None):
        input = input[0]
        pixels = input.flatten()
        pixels = np.float32(pixels)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, self.num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        segmented_image = centers[labels.flatten()]
        segmented_image = segmented_image.reshape(input.shape)
        segmented_image = np.expand_dims(segmented_image == segmented_image.max(), 0).astype(np.uint8)
        return segmented_image

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            if self.output_key is None:
                self.output_key = key
            d[self.output_key] = self.segment(d[key], mask=d.get(self.mask_key))
        return d

transforms = Compose([
    LoadImaged(keys=['image'], ensure_channel_first=True),  # Load images
    Resized(keys=['image'], spatial_size=(256, 256)),  # Resize all images to 256x256
    ScaleIntensityd(keys=['image']),
    ToTensord(keys=['image']),
    BlackEdgeRemovald(keys=["image"], output_key='edgeremoved'),
    # ScaleIntensityd(keys=['edgeremoved']),
    # KMeansSegmentd(keys=["edgeremoved"], num_clusters=2, output_key='kmeans')  # Change from list to string
])



