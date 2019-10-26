from common import *

def to_var(x, volatile=False):
	if torch.cuda.is_available():
		x = x.cuda()

	return Variable(x, volatile=volatile)

def imshow(img, display=False):
    npimg = img.cpu().numpy()
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig('autoencoder_output.png')
    if display:
        plt.show()
def generate_random_versions_of_image(image, transformer, n_versions=10):
    output = []
    for i in range(n_versions):
        output.append(transformer(image))

    return torch.stack(output)
def get_reconstruction_loss_with_all_ae(image, autoencoder_mixture, loss_fn):
    recon_loss_mix = []
    recon_loss_mix_normalized = []

    for aspect, aspect_param in autoencoder_mixture.items():
        image = to_var(image)
        recon_image = aspect_param['autoencoder'](image)
        recon_loss  = loss_fn(recon_image, image).data.sum()
        recon_loss_mix.append(recon_loss)
        recon_loss_mix_normalized.append(abs(recon_loss - aspect_param['recon_error']))
    return np.array(recon_loss_mix), np.array(recon_loss_mix_normalized)
def belief_for_observation(image, autoencoder_mixture, loss_fn):
    belief = 1./get_reconstruction_loss_with_all_ae(image, autoencoder_mixture, loss_fn)[0]
    belief /= belief.sum()
    return belief
    
