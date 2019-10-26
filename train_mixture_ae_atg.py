from common import *
from utils import *
from models import *
from data_loader import *
from train import *

image_size = 128
data_path = glob.glob('data/real_aspects/*')
atg_ds = ATGDataset(data_path, image_size=image_size)
observation_loader = DataLoader( atg_ds, batch_size=1, shuffle=False)
atg_ds_test = ATGDataset(data_path, image_size=image_size)
observation_loader_test = DataLoader( atg_ds_test, batch_size=1, shuffle=True)

transformer = transforms.Compose([
                       transforms.ToPILImage(),
                       transforms.ColorJitter(brightness=[0.8, 1.2], contrast=0.0, saturation=0.0, hue=0.0),
                       transforms.Resize(image_size),
                       transforms.RandomCrop(image_size, pad_if_needed=True, fill=0, padding_mode='constant'),
                       transforms.ToTensor()])
aspect_count = 0
autoencoder_mixture = {}
autoencoder_mixture[aspect_count] = {}
autoencoder_mixture[aspect_count]['autoencoder'] = nn.Sequential(Encoder(), Decoder())
autoencoder_mixture[aspect_count]['recon_error'] = 0

final_aspect_nodes = []
dataiter = iter(observation_loader_test)

for obs_index, image in enumerate(observation_loader):

    recon_loss, recon_loss_norm = get_reconstruction_loss_with_all_ae(image,
                                                     autoencoder_mixture,
                                                     loss_fn = torch.nn.functional.mse_loss)
    print(recon_loss, recon_loss_norm, recon_loss_norm .min())
    print('[Observation: %d Num Aspects: %d min recon loss: %.4f]'%(obs_index, aspect_count+1, np.min(recon_loss_norm)))

    if recon_loss_norm.min() > 0.015:

        if obs_index != 0:
            aspect_count += 1
            print('Creating aspect_%d'%(aspect_count))
        print()
        final_aspect_nodes.append(image)

        autoencoder_mixture[aspect_count] = {}
        autoencoder_mixture[aspect_count]['autoencoder'] = nn.Sequential(Encoder(), Decoder())
        gen_images = generate_random_versions_of_image(image.squeeze(0), transformer, n_versions=300)
        ds = AutoEncoderDataset(gen_images, aspect_image=image)
        optimizer = optim.Adam(autoencoder_mixture[aspect_count]['autoencoder'].parameters(), lr=1e-3)
        criterion = nn.BCELoss()
        data_loader = DataLoader(ds, batch_size=4, shuffle=True)

        train_autoencoder(autoencoder_mixture[aspect_count]['autoencoder'],
                          optimizer,
                          criterion,
                          data_loader,
                          number_of_epochs=5,
                          name='aspect_autoencoder_' + str(aspect_count), verbose=True)


        autoencoder_mixture[aspect_count]['autoencoder'].eval()

        test_image = to_var(gen_images[0].unsqueeze(0))
        test_image_recon = autoencoder_mixture[aspect_count]['autoencoder'](test_image)

        recon_error = torch.nn.functional.mse_loss(test_image_recon, test_image)
        autoencoder_mixture[aspect_count]['recon_error'] = recon_error.data.sum()

        random_test_image = to_var(dataiter.next())
        random_test_image_recon = autoencoder_mixture[aspect_count]['autoencoder'](random_test_image)

        test_recon_and_image = torch.stack([test_image.data,
                                            test_image_recon.data,
                                            random_test_image,
                                            random_test_image_recon]).squeeze(1).data
        fig=plt.figure(figsize=(18, 16), dpi= 100, facecolor='w', edgecolor='k')
        imshow(make_grid(test_recon_and_image), True)



final_aspect_nodes = torch.stack(final_aspect_nodes).squeeze(1).data
fig=plt.figure(figsize=(18, 16), dpi= 100, facecolor='w', edgecolor='k')
imshow(torchvision.utils.make_grid(final_aspect_nodes), True)

for obs_index, image in enumerate(observation_loader):
        belief = belief_for_observation(image,
                               autoencoder_mixture,
                               torch.nn.functional.mse_loss)


        plt.subplot(1, 2, 1)
        plt.imshow(image.numpy().squeeze(0).transpose(1, 2, 0))
        plt.title('Input image')
        plt.subplot(1, 2, 2)
        plt.bar(np.arange(belief.shape[0]), belief)
        plt.title('belief')
        plt.xlabel('aspect')
        plt.ylabel('belief')
        fig=plt.figure(figsize=(18, 16), dpi= 100, facecolor='w', edgecolor='k')
        imshow(make_grid(final_aspect_nodes), False)
        plt.title('Aspect nodes')
        plt.show()
        if obs_index == 5:
            break
