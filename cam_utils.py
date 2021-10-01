import numpy as np
import cv2
import torch
from torch import functional as F
import matplotlib.pyplot as plt

def returnCAM(feature_conv, weight_softmax, class_idx, size_upsample = (128, 128)):
    # generate the class activation maps upsample to 128x128
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def get_param_weights(model_t):

  finalconv_name = 'layer4'
  # hook the feature extractor
  features_blobs = []
  def hook_feature(module, input, output):
      features_blobs.append(output.data.cpu().numpy())

  model_t._modules.get(finalconv_name).register_forward_hook(hook_feature)

  # get the softmax weight
  params = list(model_t.parameters())
  weight_softmax = np.squeeze(params[-2].data.cpu().numpy())

  return features_blobs, params, weight_softmax


def draw_activation_map(model, image, label, train_dataset, width = 128, height = 128):
    features_blobs, params, weight_softmax = get_param_weights(model)

    # image, label  = next(iter(dataloaders['val']))
    model.eval()
    #
    model.cpu()
    #
    scores = model(image)  # get the raw scores
    probs = F.softmax(scores,
                      dim=1).data.squeeze()  # use softmax to generate the probability distribution for the scores
    probs, idx = probs.sort(0,
                            True)  # sort the probability distribution in descending order, and idx[0] is the predicted class
    print('sum of probabilities: %.0f' % torch.sum(probs).numpy())
    print('true class: ', train_dataset.class_names[label])
    print('predicated class: ', train_dataset.class_names[idx[0].numpy()])

    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)

    image = image.reshape((3, width, height)).numpy().transpose((1, 2, 0))

    # print('original image shape: ', image.reshape((3, 128, 128)).numpy().transpose((1,2,0)).shape)
    # print('heatmap.shape:', heatmap.shape)
    # image = image.reshape((3, 128, 128)).numpy().transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image * std + mean
    image = np.clip(image, 0, 1)

    heatmap = np.interp(heatmap, [0, 255], [0, 1])

    result = 0.5 * heatmap + 0.5 * image
    
    plt.imshow(image)
    plt.show()
    # plt.imshow(heatmap)
    # plt.show()

    plt.imshow(result)
    plt.show()