# # tensorrt==================================================================================================
# # convert the model in full-dimensions mode with an given input shape:
# # /home/lpr/TensorRT-8.6.1.6/bin/trtexec \
# # --onnx=fan-tiny/cityscapes_fan_tiny_hybrid_224.onnx \
# # --saveEngine=fan-tiny/cityscapes_fan_tiny_hybrid_3x3x224x224_onnx_engine.trt \
# # --shapes=input:3x3x224x224


# import numpy as np
# import os
# import pycuda.driver as cuda
# import pycuda.autoinit

# import matplotlib.pyplot as plt
# from PIL import Image

# import tensorrt as trt
# import os
# from segmentation.seg_wrapper import SegmodelWrapper
# import cv2


# os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

# # Check TensorRT version
# # print("TensorRT version:", trt.__version__)
# # assert trt.Builder(trt.Logger())

# import tensorrt as trt
# import numpy as np
# import os

# import pycuda.driver as cuda
# import pycuda.autoinit



# class HostDeviceMem(object):
#     def __init__(self, host_mem, device_mem):
#         self.host = host_mem
#         self.device = device_mem

#     def __str__(self):
#         return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

#     def __repr__(self):
#         return self.__str__()

# class TrtModel(SegmodelWrapper):
    
#     def __init__(self,engine_path,max_batch_size=1,dtype=np.float32):
        
#         self.engine_path = engine_path
#         self.dtype = dtype
#         self.logger = trt.Logger(trt.Logger.WARNING)
#         self.runtime = trt.Runtime(self.logger)
#         self.engine = self.load_engine(self.runtime, self.engine_path)
#         self.max_batch_size = max_batch_size
#         self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
#         self.context = self.engine.create_execution_context()
                          
                
#     @staticmethod
#     def load_engine(trt_runtime, engine_path):
#         trt.init_libnvinfer_plugins(None, "")             
#         with open(engine_path, 'rb') as f:
#             engine_data = f.read()
#         engine = trt_runtime.deserialize_cuda_engine(engine_data)
#         return engine
    
    
#     def allocate_buffers(self):
        
#         inputs = []
#         outputs = []
#         bindings = []
#         stream = cuda.Stream()
        
#         for binding in self.engine:
#             size = trt.volume(self.engine.get_binding_shape(binding)) * self.max_batch_size
#             host_mem = cuda.pagelocked_empty(size, self.dtype)
#             device_mem = cuda.mem_alloc(host_mem.nbytes)
            
#             bindings.append(int(device_mem))

#             if self.engine.binding_is_input(binding):
#                 inputs.append(HostDeviceMem(host_mem, device_mem))
#             else:
#                 outputs.append(HostDeviceMem(host_mem, device_mem))
        
#         return inputs, outputs, bindings, stream
                   
#     def __call__(self,x:np.ndarray,batch_size=1):

        
#         x = x.astype(self.dtype)
        
#         np.copyto(self.inputs[0].host,x.ravel())
        
#         for inp in self.inputs:
#             cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
        
#         self.context.execute_async(batch_size=batch_size, bindings=self.bindings, stream_handle=self.stream.handle)
#         for out in self.outputs:
#             cuda.memcpy_dtoh_async(out.host, out.device, self.stream) 
            
        
#         self.stream.synchronize()
#         return [out.host.reshape(batch_size,-1) for out in self.outputs]


# # class segFormerTRT(SegmodelWrapper):

# #     def __init__(self,
# #                  model_path,
# #                  label_mapping=None,
# #                  preprocessor=None,
# #                  custom_palette=None,
# #                  num_classes = 20):

# #         # load tensorrt engine =====================
# #         print("TensorRT version:", trt.__version__)
# #         TRT_LOGGER = trt.Logger()
# #         engine_file_path = model_path
# #         assert os.path.exists(engine_file_path)
# #         print("Reading engine from file {}".format(engine_file_path))
# #         with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
# #             self.engine = runtime.deserialize_cuda_engine(f.read())

# #         self.input_name = self.engine.get_tensor_name(0)

# #         self.input_shape = tuple(self.engine.get_tensor_shape(self.input_name))

# #         self.context = self.engine.create_execution_context()
# #         self.context.set_input_shape(self.input_name, self.input_shape)
# #         assert self.context.all_binding_shapes_specified

# #         self.preprocessor = preprocessor or self.preprocess

# #         self.default_numlabels = num_classes
# #         self.apply_label_mapping(label_mapping,custom_palette)
# #         self.warmup()
    

# #     def predict(self, images):
# #         preprocessed_images = self.preprocessor(images)
# #         print(f"Preprocessed images shape: {preprocessed_images.shape}")

# #         bindings = []
# #         for binding in self.engine:
# #             size = trt.volume(self.context.get_tensor_shape(binding))
# #             dtype = trt.nptype(self.engine.get_tensor_dtype(binding))
# #             print(f"Binding: {binding}, size: {size}, dtype: {dtype}")
# #             if binding == self.input_name:
# #                 input_buffer = np.ascontiguousarray(preprocessed_images)
# #                 input_memory = cuda.mem_alloc(preprocessed_images.nbytes)
# #                 print(f"Input buffer size: {preprocessed_images.nbytes}")
# #                 bindings.append(int(input_memory))
# #             else:
# #                 output_buffer = cuda.pagelocked_empty(size, dtype)
# #                 output_memory = cuda.mem_alloc(output_buffer.nbytes)
# #                 print(f"Output buffer size: {output_buffer.nbytes}")
# #                 bindings.append(int(output_memory))

# #         stream = cuda.Stream()
# #         # Transfer input data to the GPU.
# #         cuda.memcpy_htod_async(input_memory, input_buffer, stream)
# #         # Run inference
# #         self.context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
# #         # Transfer prediction output from the GPU.
# #         cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
# #         # Synchronize the stream
# #         stream.synchronize()

# #         pred_segs = np.reshape(output_buffer,(self.input_shape[0],*self.input_shape[-2:]))

# #         if self.label_mapping is not None:
# #             pred_segs = self.convert_label(pred_segs)  

# #         return pred_segs
        

# #     def preprocess(self,images):
# #         """
# #         input : list of bgr image

# #         return : preprocessed batch of image => numpy array (batch,ch,h,w)
        
# #         """

# #         # Resize the image to match the specified input dimensions (1024x1024)
# #         preprocessed_images = []

# #         for image in images:
# #             input_width, input_height = self.input_shape[-2:]
# #             resized_image = cv2.resize(image, (input_width, input_height))
# #             resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

# #             # Convert the image to float32 and apply color format adjustment
# #             resized_image = resized_image.astype(np.float32)
# #             resized_image -= np.array([123.675, 116.28, 103.53])

# #             # Scale the image data by the net-scale-factor
# #             net_scale_factor = 0.01735207357279195
# #             resized_image *= net_scale_factor

# #             # Expand dimensions to match the model input shape (3, 1024, 1024)
# #             preprocessed_image = np.transpose(resized_image, (2, 0, 1))  # (H, W, C) to (C, H, W)

# #             # Convert the preprocessed image to a format suitable for inference (e.g., to Tensor)
# #             preprocessed_images.append(preprocessed_image)

# #         return np.stack(preprocessed_images,axis=0)
    

# #     def postprocess(self,datas):
# #         """
# #         input : logits data (segmentation result)

# #         return : pil image of semantic segmentation 

# #         """
# #         imgs = []
# #         for data in datas:
# #             img = Image.fromarray(data.astype('uint8'), mode='P')
# #             img.putpalette(self.palette)
# #             imgs.append(img)
# #         return imgs
    

    






