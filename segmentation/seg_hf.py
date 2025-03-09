
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation,\
    AutoImageProcessor,Mask2FormerForUniversalSegmentation
import torch
import numpy as np
import torch


torch.backends.cuda.matmul.allow_tf32 = True

from segmentation.seg_wrapper import SegmodelWrapper

class HFsegWrapper(SegmodelWrapper):

    """
    hugging face model with addition method for visulize and predict
    """
   
    def __init__(self,
                 model_repo,
                 label_mapping=None,
                 crop = None,
                 custom_processor=None,
                 custom_palette=None,
                 fp16 = False,
                 torch_compile = False):

        super().__init__()
        if label_mapping is not None:
            assert isinstance(label_mapping,dict)

        if crop is not None:
            assert len(crop) == 2

        print('===== Segmentation =====')
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("using ",self.device)

        self.torch_dtype = torch.float16 if fp16 and self.device.type == "cuda" else torch.float32

        self.crop = crop
        self.label_mapping = label_mapping
        self.custom_processor = custom_processor
        self.processor,self.model = self._init_preprocess_model(model_repo)
        
        if torch_compile:
            self.model = torch.compile(self.model)
        self.default_numlabels = self.model.config.num_labels
        self.apply_label_mapping(label_mapping,custom_palette)
        self.warmup()

    def _init_preprocess_model(self, repo):
        """
        input: repo
        return: preprocess, model
        """
        raise NotImplementedError("Method '_init_preprocess_model' must be implemented in subclasses")

    def _get_segmaps(self, preds):
        """
        input: preds
        return: segmaps, preds
        """
        raise NotImplementedError("Method '_get_segmaps_outputs' must be implemented in subclasses")
    
    
    def __call__(self,images):

        """
        inputs: list of images
        return predict output and overlayed image
        """
        if self.crop is not None:
            h,w = self.crop
            shape = images[0].shape
            y1,y2, x1,x2 = max(int(shape[0]/2-h/2),0), min(int(shape[0]/2+h/2),shape[0]), max(int(shape[1]/2-w/2),0), min(int(shape[1]/2+w/2),shape[1])
            self.images = [img[y1:y2,x1:x2] for img in images]
        else:
            self.images = images

        with torch.no_grad():
            if self.custom_processor is None:
                inputs = self.processor(self.images,return_tensors="pt").to(self.device,self.torch_dtype)
            else:
                inputs = {'pixel_values':self.processor(torch.stack([self.processor(img) for img in self.images],axis=0))}.to(self.device,self.torch_dtype)

            self.outputs = self.model(**inputs)

        pred_segs = self._get_segmaps(self.outputs)
        
        if self.label_mapping is not None:
            pred_segs = self.convert_label(pred_segs)  

        return pred_segs
    
    
class HF_segFormermodel(HFsegWrapper):
   
    def __init__(self, *args, **kwargs):
        # Call the parent constructor with all arguments
        super().__init__(*args, **kwargs)

    def _init_preprocess_model(self, repo):
        """
        input: repo
        return: preprocess, model
        """
        processor = self.custom_processor or SegformerImageProcessor.from_pretrained(repo)
        model = SegformerForSemanticSegmentation.from_pretrained(repo,torch_dtype=self.torch_dtype,
                                    use_safetensors=True).to(self.device)
        return processor,model

    def _get_segmaps(self, preds):
        """
        input: preds
        return: segmaps, preds
        """
        return preds.logits.argmax(dim=1).cpu()
  
    
         
    

class HF_mask2Formermodel(HFsegWrapper):
   
    def __init__(self, *args, **kwargs):
        # Call the parent constructor with all arguments
        super().__init__(*args, **kwargs)

    def _init_preprocess_model(self, model_repo):
        """
        input: repo
        return: preprocess, model
        """
        processor = AutoImageProcessor.from_pretrained(model_repo) \
                                    if self.custom_processor is None else self.custom_processor
        model = Mask2FormerForUniversalSegmentation.from_pretrained(model_repo,  torch_dtype=self.torch_dtype,
                                    use_safetensors=True).to(self.device)
        return processor,model

    def _get_segmaps(self, preds):
        """
        input: preds
        return: segmaps, preds
        """
        seg_maps = self.processor.post_process_semantic_segmentation(preds,
                                                                 target_sizes=[image.shape[:-1] for image in self.images]
                                                                    )

        seg_maps = [seg.cpu().numpy().astype(np.uint8) for seg in seg_maps]
        return seg_maps
  


  