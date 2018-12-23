import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from mnist import MNIST
data = MNIST(data_dir="data/MNIST/")

img_size = data.img_size
img_size_flat = data.img_size_flat

img_shape = data.img_shape
num_classes = data.num_classes
num_channels = data.num_channels
tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)

net = x_image

net = tf.layers.conv2d(inputs=net, name='layer_conv1', padding='same',
                       filters=32, kernel_size=5, activation=tf.nn.relu)
net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

net = tf.layers.conv2d(inputs=net, name='layer_conv2', padding='same',
                       filters=64, kernel_size=5, activation=tf.nn.relu)
net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

net = tf.contrib.layers.flatten(net)

net = tf.layers.dense(inputs=net, name='layer_fc1',
                      units=1024, activation=tf.nn.relu)
net = tf.layers.dense(inputs=net, name='layer_fc_out',
                      units=num_classes, activation=None)
logits = net
y_pred = tf.nn.softmax(logits=logits)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true,
                                                           logits=logits)

loss = tf.reduce_mean(cross_entropy)
gradient = tf.gradients(loss, x)[0]

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)


y_pred_cls = tf.argmax(y_pred, axis=1)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

batch_size = 256

def predict_cls(images, labels, cls_true):
    num_images = len(images)
    cls_pred = np.zeros(shape=num_images, dtype=np.int)

    i = 0

    while i < num_images:
        j = min(i + batch_size, num_images)
       
        feed_dict = {x: images[i:j, :],
                     y_true: labels[i:j, :]}
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)
        i = j
    correct = (cls_true == cls_pred)

    return correct, cls_pred

def predict_cls_test():
    return predict_cls(images = data.x_test,
                       labels = data.y_test,
                       cls_true = data.y_test_cls)
    
def cls_accuracy(correct):
    correct_sum = correct.sum()
    acc = float(correct_sum) / len(correct)

    return acc, correct_sum

def print_test_accuracy():  
    correct, cls_pred = predict_cls_test()
    acc, num_correct = cls_accuracy(correct)
    num_images = len(correct)
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, num_correct, num_images))

def plot_images(image, noisy_image,
                name_source, name_target,
                score_source, score_source_org, score_target):
    
    # Create figure with sub-plots.
    fig, axes = plt.subplots(1, 2, figsize=(10,10))

    # Adjust vertical spacing.
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    # Use interpolation to smooth pixels?
    smooth = True
    
    # Interpolation type.
    if smooth:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'

    # Plot the original image.
    # Note that the pixel-values are normalized to the [0.0, 1.0]
    # range by dividing with 255.
    ax = axes.flat[0]
    ax.imshow(image.reshape([28, 28]), interpolation=interpolation)
    msg = "Original Image:\n{0} ({1:.2%})"
    xlabel = msg.format(name_source, score_source_org)
    ax.set_xlabel(xlabel)

    # Plot the noisy image.
    ax = axes.flat[1]
    ax.imshow(noisy_image.reshape([28, 28]), interpolation=interpolation)
    msg = "Image + Noise:\n{0} ({1:.2%})\n{2} ({3:.2%})"
    xlabel = msg.format(name_source, score_source, name_target, score_target)
    ax.set_xlabel(xlabel)


    # Remove ticks from all the plots.
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()
    
    
iter_num = 10

def top_five(precon, orig_label):
    count = 0
    for i in range(len(precon)):
        five_acc=np.argsort(precon[i])[-5:]
        
        if orig_label[i] in five_acc:
            count +=1
        #print("The result is")
        #print(orig_label[i],five_acc)
    acc = count/len(orig_label)
    return acc

def top_one(precls, orig_label):
    count = 0
    for i in range(len(orig_label)):
        if precls[i]==orig_label[i]:
            count+=1
    acc = count/ len(orig_label)
    return acc

def basic_iter(image, label, stsize):
    
    orig_labels=[]
    #adver_image =np.zeros(shape=len(image), dtype=np.int)
    for i in range(len(label)):
        orig_labels.append(np.argmax(label[i]))
        
    #print("The orginal labels are:",orig_labels)
    adver_image = image.copy()
    #org_cls,org_con = session.run([y_pred_cls,y_pred], feed_dict = 
    #                                 {x: image, y_true: label})
    
    
    
    for i in range(1,iter_num+1): 
        #print("iteration ", i) 
        pred_cls,pred_con, grad= session.run([y_pred_cls,y_pred, gradient], feed_dict = 
                                     {x: adver_image, y_true: label})
        #print("predict loss is", pred_loss)
        
        #adver_image = adver_image + stsize * np.sign(grad)
        #adver_image = np.clip(adver_image, 0, 1)
        noise = adver_image + 0.06 * np.sign(grad)
        adver_image = np.clip(noise, adver_image-stsize, adver_image+stsize)
    #print("The predict class is ", pred_cls)
    
    top_one_acc = top_one(pred_cls, orig_labels)
    
    top_five_acc=top_five(pred_con, orig_labels)
    return top_one_acc, top_five_acc
        
        
    
    '''
    print(pred_con)
    score_source= pred_con[0][org_cls][0]
    score_source_org = org_con[0][org_cls][0]
    score_target= pred_con[0][pred_cls][0]
    
    print("The score is ", score_source)
    print("The source orgin score is ", score_source_org)
    print("The result score is ", score_target)
    plot_images(image, adver_image, org_cls, pred_cls, score_source, score_source_org, score_target )   
    '''
        

def least_like_iter(image, label, stsize):
    logits_ = session.run(logits, feed_dict={x:image, y_true:label})
    least_like_class = np.argmin(logits_,1)
    #print("The least like class is ", least_like_class)
    one_hot_label = np.zeros((len(image), 10))
    one_hot_label[np.arange(len(image)), least_like_class]=1.0
    
    orig_labels=[]
    for i in range(len(label)):
        orig_labels.append(np.argmax(label[i]))
        
    #print("The orginal labels are:",orig_labels)
    adver_image = image.copy()
    
    for i in range(1,iter_num+1): 
       #print("iteration ", i) 
       pred_cls,pred_con, grad= session.run([y_pred_cls,y_pred, gradient], feed_dict = 
                                     {x: adver_image, y_true:one_hot_label })
        #print("predict loss is", pred_loss)
       #adver_image = adver_image - stsize * np.sign(grad)
       noise = adver_image - 0.06 * np.sign(grad)
       adver_image = np.clip(noise, adver_image-stsize, adver_image+stsize)
    
    top_one_acc = top_one(pred_cls, orig_labels)
    
    top_five_acc=top_five(pred_con, orig_labels)
    return top_one_acc, top_five_acc
   
def fgsm(image, label, stsize):
    orig_labels=[]
    for i in range(len(label)):
        orig_labels.append(np.argmax(label[i]))
    #print("orginal classes are", orig_labels)
    grad = session.run(gradient, feed_dict = 
                                     {x: image, y_true: label})
   
    adver_image = image + stsize * np.sign(grad)
    pred_cls,pred_con = session.run([y_pred_cls,y_pred], feed_dict = 
                                     {x: adver_image, y_true: label})
    #print("pred classes are", pred_cls)
    top_one_acc = top_one(pred_cls, orig_labels)
    
    top_five_acc=top_five(pred_con, orig_labels)
    return top_one_acc, top_five_acc

save_dir = 'checkpoints/'
saver = tf.train.Saver()
session = tf.Session()
saver.restore(sess=session, save_path=save_dir)



attack_idx = np.random.choice(len(data.y_test), 300)
attack_img = data.x_test[attack_idx]
attack_label = data.y_test[attack_idx]
att_label_num = data.y_test_cls[attack_idx]


#topone, topfive = basic_iter(attack_img, attack_label,0.06)



#basic_one=[]
basic_five = []
iter_five_acc=[]
#fgsm_acc = []

step_range =[0.005*i for i in range(1,16)]
for step_size in step_range:
    print(step_size)
    one_acc, five_acc = basic_iter(attack_img, attack_label, step_size)
    iter_one, iter_five = least_like_iter(attack_img, attack_label, step_size)
    
    #fgsm_one, fgsm_five = fgsm(attack_img, attack_label, step_size)
    
    #basic_one.append(one_acc)
    basic_five.append(five_acc)
    iter_five_acc.append(iter_five)
    #fgsm_acc.append(fgsm_five)

plt.plot(step_range, basic_five,'--x', label='basic iter adv.')
plt.plot(step_range, iter_five_acc, '--o', label='least like adv.')

#plt.plot(step_range, basic_five,'--g*', label ='Top five accuracy')
#plt.title("The accuracy under basic iterative attack")

plt.xlabel('epsilon')
plt.ylabel('accuracy')
plt.legend()
plt.show()

'''
fgsmacc=[]
basicacc=[]
llaccuracy=[]


step_range =[0.01*i for i in range(1,21)]
for step_size in step_range:
    print(step_size)
    fgsmacc.append(fgsm(attack_img, attack_label, step_size))
    basicacc.append(basic_iter(attack_img, attack_label, step_size))
    llaccuracy.append(least_like_iter(attack_img, attack_label, step_size))
print(fgsmacc)
print(basicacc)
print(llaccuracy)

plt.plot(step_range,basicacc,'--x', label='basic iter adv.')
plt.plot(step_range,llaccuracy,'--o', label='least likely adv.')
plt.plot(step_range, fgsmacc, '--*', label='fgsm adv.')

plt.xlabel('epsilon')
plt.ylabel('accuracy')
plt.legend()
plt.show()
#least_like_iter(attack_img, attack_label,0.02)

#print_test_accuracy()

#session.run(tf.global_variables_initializer())
'''