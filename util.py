import numpy as np
import matplotlib.pyplot as plt
def load_data(path):
    data = np.load(path)
    x_train, y_train = data['x_train'], data['y_train']
    x_test, y_test = data['x_test'], data['y_test']
    data.close()
    return (x_train, y_train), (x_test, y_test)
"""
def draw_some_sample(x_train):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    for i in range(5):
        fig=plt.figure(frameon=False)
        fig.set_size_inches(0.56,0.56)
        ax=plt.Axes(fig,[0.,0.,1.,1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(x_train[i], cmap=plt.cm.gray,interpolation='nearest')
        plt.show()


def draw_some_sample_together(x_train):
    fig=plt.figure(figsize=(16,4))
    column=16
    rows=4
    for i in range(1,column*rows+1):
        ax=fig.add_subplot(rows,column,i)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(x_train[i-1],cmap=plt.cm.gray,interpolation='nearest')
        fig.show()


def print_one_sample_pixels(sample):
    for i in range(28):
        line=''
        for j in range(28):
            if sample[i][j]==0:
                line=line+'/t'+str(sample[i][j])
            else:
                line=line+'/t'+str(sample[i][j])
        line=line+'\n'
        print(line)
"""

def draw_one_sample(sample):
    fig=plt.figure(frameon=False)
    fig.set_size_inches(0.56,0.56)
    ax=plt.Axes(fig,[0.,0.,1.,1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(sample,cmap=plt.cm.gray,interpolation='nearest')
    plt.show()

def draw_training_performance_graph(history,epochs):
    figure,(ax1,ax2)=plt.subplots(1,2,figsize=(10,4))
    figure.suptitle('Training Performance',fontsize=12)
    figure.subplots_adjust(top=1.00,wspace=0.4)
    epoch_list=list(range(1,epochs+1))

    ax1.plot(epoch_list,history.history['accuracy'],label='Training Accuracy')
    val_accuracy=history.history['val_accuracy']
    ax1.plot(epoch_list,val_accuracy,label='Validation Accuracy')
    ax1.grid(linewidth=0.25)
    ax1.set_xticks(np.arange(0,epochs+1,5))
    ax1.set_ylabel('Accuracy Value')
    ax1.set_xlabel('Epoch #')
    ax1.set_title('Accuracy')
    ax1.legend(loc='best')

    ax2.plot(epoch_list,history.history['loss'],label='Training Loss')
    ax2.plot(epoch_list,history.history['val_loss'],label='Validation Loss')
    ax2.grid(linewidth=0.25)
    ax2.set_xticks(np.arange(0,epochs+1,5))
    ax2.set_ylabel('Loss Value')
    ax2.set_xlabel('Epoch #')
    ax2.set_title('Loss')
    ax2.legend(loc='best')

    figure.show()
