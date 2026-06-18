# Copyright (c) Qualcomm Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

# =============================================================================
#
# Copyright (c) 2024, Qualcomm Innovation Center, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# =============================================================================

import numpy as np
from PIL import Image
import requests
import os
 
def preprocess(img_path, nhwc = False):
    """
    Preprocesses an image for a neural network model.
 
    Args:
        img_path (str): Path to the input image or image web link.
        nhwc (bool, optional): If True, reshapes the output in NHWC format (channels last).
            Otherwise, uses NCHW format (channels first). Default is False.
 
    Returns:
        np.ndarray: Preprocessed image as a NumPy array.
    """
    if os.path.exists(img_path):
        img = Image.open(img_path)
    else:
        response = requests.get(img_path, stream=True)
        response.raw.decode_content = True
        # reading the image
        img = Image.open(response.raw)
        
    # Resize the image to 224x224
    img = img.resize((224, 224))
    # Convert image to NumPy array and normalize
    img = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    # Transpose to NCHW format
    img = img.transpose(2, 0, 1)
    if nhwc :
        # Transpose to NHWC format if needed
        img = img.transpose(1,2, 0)
    img = np.expand_dims(img,axis=0)

    return img.astype(np.float32)
   
 
def postprocess(output,label_text_path=None):
    """
    Post-processes the output of a neural network model for image classification.
 
    Args:
        output (str or np.ndarray): The output of the neural network model. If a string(path),
                                    it is assumed to be a binary file containing the output probabilities.
        label_path (str, optional): Path to the label text file. If not provided, defaults to "synset.txt".
 
    Returns:
        None: Prints the top-5 predictions along with their corresponding class labels and probabilities.
    """
 
    if label_text_path is None:
        label_text_path = "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/apidoc/imagenet_classes.txt"
    response = requests.get(label_text_path, stream=True)
    response.raw.decode_content = True
    # with open(label_text_path, 'r') as file:
    labels = [str(s.strip()) for s in response.raw]
    if isinstance(output, str):
        output = np.fromfile(output, dtype=np.float32)
        probabilities = np.exp(output) / np.sum(np.exp(output), axis=0)
    else:
        probabilities = np.exp(output[0][0]) / np.sum(np.exp(output[0][0]), axis=0)
    print("\n********************************\nTop-5 predictions:")
    scores = np.squeeze(probabilities)
    sorted_probability = np.argsort(scores)[::-1]
    for i in sorted_probability[0:5]:
        print(' class=%s ; probability=%f100' %(labels[i],scores[i]))
