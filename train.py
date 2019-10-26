from common import *
from models import *
from utils import *
from data_loader import *

def train_autoencoder(autoencoder, optimizer, criterion, data_loader, number_of_epochs=1, name='main', verbose=False):
    print('Training %s ...'%(name))
    for epoch in range(number_of_epochs):

        running_loss = 0.0
        autoencoder.train()
        for batch_index, (in_images, aspect_image) in enumerate(data_loader):

            in_images = to_var(in_images)
            out_images = autoencoder(in_images)

            loss = criterion(out_images, aspect_image)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.data.numpy()
            if batch_index % 100==0 and verbose:
                print('epoch %d loss: %.5f' % (epoch, running_loss/((batch_index + 1))))
            if batch_index != 0 and batch_index % 1000 == 0:
                break
    print('Done training %s'%(name))

def init_autoencoder_mixture(train_loader, test_loader, n_clusters=10):
    autoencoder_mixture = {}

    autoencoder_main = nn.Sequential(Encoder(), Decoder())
    criterion = nn.BCELoss()
    optimizer = optim.Adam(autoencoder_main.parameters(), lr=1e-3)
    number_of_epochs = 5

    train_autoencoder(autoencoder_main , optimizer, criterion, train_loader, number_of_epochs, name='main')

    dataiter = iter(test_loader)
    in_images = dataiter.next()[0]
    autoencoder_main.eval()
    init_data = autoencoder_main[0](to_var(in_images)).data.view(in_images.shape[0],-1).numpy()

    autoencoder_mixture['autoencoder_main'] = autoencoder_main

    print('Clustering ...')

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(init_data)
    print('Done Clustering !!')
    for cluster in range(n_clusters):
        autoencoder_mixture[cluster] = {}
        ds = AutoEncoderDataset(in_images[kmeans.labels_==cluster])
        autoencoder_mixture[cluster]['autoencoder'] = nn.Sequential(Encoder(), Decoder())
        optimizer = optim.Adam(autoencoder_mixture[cluster]['autoencoder'].parameters(), lr=1e-2)
        criterion = nn.BCELoss()
        data_loader = DataLoader(ds, batch_size=4, shuffle=True)

        train_autoencoder(autoencoder_mixture[cluster]['autoencoder'],
                          optimizer,
                          criterion,
                          data_loader,
                          number_of_epochs = 10,
                          name='cluster_'+ str(cluster))

        test_image = iter(data_loader).next()[0]
        autoencoder_mixture[cluster]['autoencoder'].eval()
        recon_image = autoencoder_mixture[cluster]['autoencoder'](to_var(test_image))
        plt.subplot(10, 10, cluster + 1)
        plt.imshow(test_image[0].numpy().squeeze(0))
        plt.subplot(10, 10, cluster + 11)
        plt.imshow(recon_image[0].data.numpy().squeeze(0))

    autoencoder_mixture['cluster_net'] = ClusterNet()
    plt.show()
    return autoencoder_mixture
def get_mixure_output(autoencoder_mixture, images, n_clusters=10):
    output = []
    for cluster in range(n_clusters):
        output.append(autoencoder_mixture[cluster]['autoencoder'](images))
    return output
def autoencoder_mixure_loss_fn(images, mixure_outputs, clustering_net_output, n_clusters=10):
    loss = 0
    for cluster in range(n_clusters):
        mse = -((mixure_outputs[cluster] - images)**2).mean(axis=(1, 2, 3))
        loss += clustering_net_output[:, cluster] * torch.exp(mse)
    loss = -torch.log(loss).sum()
    return loss
def train_autoencoder_mixure(autoencoder_mixture, data_loader, number_of_epochs=1, n_clusters = 10, name='main', verbose=False):
    print('Training %s ...'%(name))

    params = list(autoencoder_mixture['cluster_net'].parameters())

    for cluster in range(n_clusters):
        params += list(autoencoder_mixture[cluster]['autoencoder'].parameters())
        autoencoder_mixture[cluster]['autoencoder'].train()

    optimizer = optim.Adam(params, lr=1e-2)
    number_of_epochs = 5

    for epoch in range(number_of_epochs):
        running_loss = 0.0
        autoencoder_mixture['cluster_net'].train()

        for batch_index, (in_images, labels) in enumerate(data_loader):

            in_images = to_var(in_images)
            mixure_outputs = get_mixure_output(autoencoder_mixture, in_images, n_clusters=10)
            clustering_net_outputs = autoencoder_mixture['cluster_net'](in_images)
            loss = autoencoder_mixure_loss_fn(in_images, mixure_outputs, clustering_net_outputs, n_clusters=10)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.data.numpy()
            if batch_index % 100==0 and verbose:
                print('epoch %d loss: %.5f batch: %d' % (epoch, running_loss/((batch_index + 1)), (batch_index + 1)*batch_size))
            if batch_index != 0 and batch_index % 1000 == 0:
                break
if __name__ == '__main__':
    batch_size = 8
    train_loader = DataLoader(
    torchvision.datasets.MNIST('data/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()
                             ])),
    batch_size=batch_size, shuffle=True)

    test_loader = DataLoader(
    torchvision.datasets.MNIST('data/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()
                             ])),
    batch_size=2000, shuffle=False)

    autoencoder_mixture = init_autoencoder_mixture(train_loader, test_loader, n_clusters=10)
    train_autoencoder_mixure(autoencoder_mixture,
                         train_loader,
                         number_of_epochs=1,
                         name='autoencoder mixture', verbose=True)
