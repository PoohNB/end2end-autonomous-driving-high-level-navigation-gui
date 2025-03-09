# Load model directly
import numpy as np
import cv2
import colorsys
import time

class SegmodelWrapper():

    """
    This class provides methods to:
    - Post-process the label result
    - Generate colors for each class
    - Overlay segmentation results on images
    
    Requirements:
    - Initialization with device setup
    - Application of label mapping with apply_label_mapping(label_mapping, custom_palette) after initialization
    
    Methods:
    - warmup(image_shape=(512,1024,3)): Perform inference and print the average time.
    - generate_colors(num_classes): Generate color for labels.
    - apply_label_mapping(label_mapping, custom_palette=None): Apply label mapping for post-processing.
    - predict(images): Abstract method to be implemented in subclasses.
    - convert_label(seg_maps): Convert original semantic map to desired semantic map based on label_mapping.
    - get_seg_overlay(images, segs, original_size=True): Get the image overlay results.
    - get_seg_images(segs, shape=None): Get the segmentation results as colored images.
    """

   
    def __init__(self):        

        pass


    def warmup(self,image_shape=(512,1024,3),batch=1):
        # warmup
        dummy_images = [np.random.randint(0, 255, image_shape, dtype=np.uint8)]*batch
        self(dummy_images)
        test_times = 5
        st = time.time()
        for _ in range(test_times):
            self(dummy_images)
        print(f"inference time :{(time.time()-st)/test_times:.6f}")

    @staticmethod
    def generate_colors(num_classes):
        hsv_colors = [(i / num_classes, 1.0, 1.0) for i in range(num_classes)]
        rgb_colors = [colorsys.hsv_to_rgb(*color) for color in hsv_colors]
        rgb_colors = [(int(r * 255), int(g * 255), int(b * 255)) for (r, g, b) in rgb_colors]
        return np.array(rgb_colors).astype(np.uint8)
    
    def apply_label_mapping(self,label_mapping,custom_palette = None):
        
        if label_mapping is not None:
            self.labels = list(set(label_mapping.values()))
            assert 0 not in self.labels, "0 is background, label can't be 0"
            num_labels = len(self.labels)
        else: 
            self.labels = list(range(self.default_numlabels))
            num_labels = self.default_numlabels
        self.label_mapping = label_mapping
        self.palette = custom_palette if custom_palette and len(custom_palette) >= num_labels \
            else self.generate_colors(num_labels)


    def __call__(self,images):

        """
        inputs: list of images
        return predict output and overlayed image

        should apply for post process
        if self.label_mapping is not None:
            pred_segs = self.convert_label(pred_segs)  

        """
        raise NotImplementedError("Method 'predict' must be implemented in subclasses")
    
    def convert_label(self,seg_maps):

        new_segmaps = []

        for seg_map in seg_maps:

            # postprocessed = np.full(seg_map.shape,255, dtype=np.uint8)
            postprocessed = np.zeros(seg_map.shape, dtype=np.uint8)

            for k,v in self.label_mapping.items():
                # print(f"k: {k}, type: {type(k)}")
                # print(f"seg_map shape: {seg_map.shape}, dtype: {seg_map.dtype}")
                postprocessed[seg_map == int(k)] = v

            new_segmaps.append(postprocessed)

        return new_segmaps
    
    def _create_color_seg(self, segs: np.ndarray) -> np.ndarray:
        color_seg = np.zeros((segs.shape[0], segs.shape[1], 3), dtype=np.uint8)
        for j, label in enumerate(self.labels):
            color_seg[segs == label, :] = self.palette[j]
        color_seg[segs == 0, :] = [0, 0, 0]
        return color_seg
         
    def get_seg_overlay(self, images: list, segs: list, original_size: bool = True) -> list:
        if len(images) != len(segs):
            raise ValueError("Number of images and segmentation results must be equal")
        
        overlay_images = []
        for img, seg in zip(images, segs):
            color_seg = self._create_color_seg(seg)
            if img.shape[:-1] != seg.shape:
                if original_size:
                    color_seg_resized = cv2.resize(color_seg, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
                    overlay = img * 0.5 + color_seg_resized * 0.5
                else:
                    img_resized = cv2.resize(img, (seg.shape[1], seg.shape[0]))
                    overlay = img_resized * 0.5 + color_seg * 0.5
            else:
                overlay = img * 0.5 + color_seg * 0.5
            overlay_images.append(overlay.astype(np.uint8))
        return overlay_images

    def get_seg_images(self, segs: list, shape: tuple = None) -> list:
        seg_images = []
        for seg in segs:
            color_seg = self._create_color_seg(seg)
            if shape is not None:
                color_seg = cv2.resize(color_seg, shape)
            seg_images.append(color_seg.astype(np.uint8))
        return seg_images
    

