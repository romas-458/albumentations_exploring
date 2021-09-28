from __future__ import print_function, division

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def classnamedot(str):
    return str.split('/')[-1].split('.')[-2]

def visualize_model(model, dataloaders, device, class_names, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def show_image_grid(model, dataloaders, device, class_names, image_datasets, num_images=24, cols=4):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()
    print('start')

    with torch.no_grad():
        for i, (inputs, labels, path) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            cur_batch_size = inputs.size()[0]
            # rows = len(images_filepaths) // cols
            rows = cur_batch_size // cols

            figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
            for j in range(inputs.size()[0]):

                images_so_far += 1
                abc = inputs.cpu().data[j]
                ## to numpy
                npimg = abc.numpy()
                npimg = np.interp(npimg, (npimg.min(), npimg.max()), (0, 1))
                npimg = np.transpose(npimg, (1, 2, 0))
                # plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
                color = "green" if class_names[preds[j]] == class_names[labels[j]] else "red"
                ax.ravel()[j].imshow(npimg)
                ax.ravel()[j].set_title('predicted: {} \n {}'.format(class_names[preds[j]], classnamedot(path[j])),
                                        color=color)
                ax.ravel()[j].set_axis_off()

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
            plt.tight_layout()
            plt.show()

        model.train(mode=was_training)


def vis_model(model, dataloaders, device, class_names, num_images=2):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels, _) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                color = "green" if class_names[preds[j]] == class_names[labels[j]] else "red"
                # ax.set_title('predicted: {} real {}'.format(class_names[preds[j]], class_names[labels[j]]), color = color)
                ax.set_title('predicted: {}'.format(class_names[preds[j]]), color=color)

                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return

        model.train(mode=was_training)

def display_image_grid(images_filepaths, predicted_labels=(), cols=4):
    rows = len(images_filepaths) // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    for i, image_filepath in enumerate(images_filepaths):
        print(image_filepath)
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        true_label = os.path.normpath(image_filepath).split(os.sep)[-2]
        predicted_label = predicted_labels[i] if predicted_labels else true_label
        color = "green" if true_label == predicted_label else "red"
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_title(predicted_label, color=color)
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()