import torch 

# discriminator loss
def compute_discriminator_loss(
    real, # real image
    fake, # fake image
    discriminator, # discriminator
    adversarial_loss, # loss function to be mininized 
):

    # compute adversarial loss for real and fake images    
    real_pred = discriminator(real)
    real_loss = adversarial_loss(real_pred, torch.ones_like(real_pred)) # real images with labels 1 
    fake_pred = discriminator(fake)
    fake_loss = adversarial_loss(fake_pred, torch.zeros_like(fake_pred)) # fake images with labels 0

    # compute discriminator loss as the mean of real and fake loss
    discriminator_loss = 0.5 * (real_loss + fake_loss)
    
    return discriminator_loss


def compute_generator_adversarial_loss(
    real_x, # real images from x set
    generator_xtoy,  # generator for converting x to y
    discriminator_y, # discriminator for image y
    adversarial_loss,
):
    
    # generate fake y from real x
    fake_y = generator_xtoy(real_x)

    # compute generator adversarial loss 
    fake_y_pred = discriminator_y(fake_y)

    # note: set up lablel as one as we hope generated fake image can be close to real images
    generator_adversarial_loss = adversarial_loss(fake_y_pred, torch.ones_like(fake_y_pred))
    
    return generator_adversarial_loss, fake_y


def compute_identity_loss(
    real_x, # real image from x
    generator_ytox, # generator converting y to x
    identity_loss, # loss function for identity property 
):

    # convert real x to fake x    
    fake_identity_x = generator_ytox(real_x)
    # compute identity loss
    generator_identity_loss = identity_loss(real_x, fake_identity_x)
    
    return generator_identity_loss


def compute_cycle_consistency_loss(
    real_x, # real images from x
    fake_y, # fake images for y
    generator_ytox, # generator converting y to x 
    cycle_loss,
):
    
    fake_cycle_x = generator_ytox(fake_y)
    generator_cycle_loss = cycle_loss(real_x, fake_cycle_x)
    
    return generator_cycle_loss


def compute_generator_loss(
    real_1, # image from 1
    real_2, # image from 2
    generator_1to2, # generator converting 1 to 2
    generator_2to1, # generator converting 2 to 1
    discriminator_1, # discriminator for 1
    discriminator_2, # discriminator for 2
    adversarial_loss, # adversarial loss for generator and discriminator
    identity_loss, # identity loss for generator
    cycle_loss, # cycle consistency loss for generator
    lambda_identity=0.1, # prefactor for identity loss
    lambda_cycle=10, # prefactor for cycle consistency loss
):

    # compute generator adversarial loss 
    # adversaial loss for 1 to 2
    adversarial_loss_1to2, fake_2 = compute_generator_adversarial_loss(
        real_1,
        generator_1to2,
        discriminator_2,
        adversarial_loss,
    )
    # adversaial loss for 2 to 1
    adversarial_loss_2to1, fake_1 = compute_generator_adversarial_loss(
        real_2,
        generator_2to1,
        discriminator_1,
        adversarial_loss,
    )
    
    # compute generator identity loss
    # identity loss for 1 to 1
    identity_loss_1 = compute_identity_loss(
        real_1,
        generator_2to1,
        identity_loss,
    )
    # identity loss for 2 to 2
    identity_loss_2 = compute_identity_loss(
        real_2,
        generator_1to2,
        identity_loss,
    )

    # compute generator cycle consistency loss
    # cycle loss for 1 to 2 to 1
    cycle_loss_1to2to1 = compute_cycle_consistency_loss(
        real_1,
        fake_2,
        generator_2to1,
        cycle_loss,
    )
    # cycle loss for 2 to 1 to 2
    cycle_loss_2to1to2 = compute_cycle_consistency_loss(
        real_2,
        fake_1,
        generator_1to2,
        cycle_loss,
    )
    
    # compute total generator loss
    generator_loss = adversarial_loss_1to2 + adversarial_loss_2to1 + \
        lambda_identity * (identity_loss_1 + identity_loss_2) + \
        lambda_cycle * (cycle_loss_1to2to1 + cycle_loss_2to1to2)
    
    return generator_loss, fake_1, fake_2
