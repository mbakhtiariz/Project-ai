def eval_UNet():
    return NotImplementedError


'''
def imshow(original, predection, mask):

    images_batch = original
    anno_images_batch = mask
    pred_batch = predection

    grid = torchvision.utils.make_grid(images_batch, nrow=batch_size)
    grid2 = torchvision.utils.make_grid(anno_images_batch, nrow=batch_size)
    grid3 = torchvision.utils.make_grid(pred_batch, nrow=batch_size)

    print('grid.shape: ', grid.shape)
    print('grid T . shape: ', grid.numpy().transpose((1, 2, 0)).shape)

    # plot image and image_anno
    ax = plt.subplot(3, 1, 1)
    ax.axis('off')
    ax.set_title('Input batch')
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    # plot image and image_anno
    ax = plt.subplot(3, 1, 2)
    ax.axis('off')
    ax.set_title('mask')
    plt.imshow(100*grid2.numpy().transpose((1, 2, 0)))

    # plot image and image_anno
    ax = plt.subplot(3, 1, 3)
    ax.axis('off')
    ax.set_title('Input batch')
    plt.imshow(100*grid3.numpy().transpose((1, 2, 0)))
    plt.title('Pred')

#---------------------EVAL ----------------------
# have to change this later to read from best_hyper_param file
model = UNet(hyper_params).to(device)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
seg_correct = seg_total = seg_intersection = seg_union = 0.
cls_correct = cls_total = 0.
draw_flag = False
with torch.no_grad():
    for batch_index, sampled_batch in enumerate(test_loader):
        images = Variable(sampled_batch['image'].to(device).float())
        seg_labels = Variable(sampled_batch['image_anno'].to(device).float())
        cls_labels = Variable(sampled_batch['GlaS'].to(device).float())

        seg_out, cls_out = model(images)

        #cls_out = torch.squeeze(cls_out, dim=1)

        _, seg_pred = torch.max(seg_out.data, 1)
        _, cls_pred = torch.max(cls_out.data, 1)

        #seg_pred = F.sigmoid(seg_pred)
        #cls_pred = F.sigmoid(cls_pred)

        seg_total += seg_labels.size(0) * seg_labels.size(1) * seg_labels.size(2)
        seg_correct += (seg_pred == seg_labels).sum().item()

        cls_total += cls_labels.size(0)
        cls_correct += (cls_pred == cls_labels).sum().item()

        seg_intersection += (seg_pred * seg_labels).sum()
        seg_union += seg_pred.sum() + seg_labels.sum()


        if batch_index == 2:
            draw_flag = True
            plt.figure()
            imshow(images[0], seg_pred[0], seg_labels[0])
            plt.axis('off')
            plt.ioff()




seg_acc_test = 100.0 * seg_correct / seg_total
seg_dice_test = 2.0 * float(seg_intersection) / float(seg_union)

cls_acc_test = 100.0 * cls_correct / cls_total
print('intersection: ', seg_intersection)
print('union: ', seg_union)
print('Accuracy of the network on the test images: ' , seg_acc_test)
print('Dice of the network on the test images:', seg_dice_test)
print('classification Accuracy of the network on the test images: ', cls_acc_test)


if draw_flag:
    plt.show()

'''