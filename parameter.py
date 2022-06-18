class hyper_param:
    image_size = (256, 256)
    prob_adjust_brightness = 0.2
    prob_adjust_contrast = 0.2
    prob_adjust_gamma = 0.2
    prob_rotate_90 = 0.3
    prob_rotate_180 = 0.3
    prob_rotate_270 = 0.3
    prob_hflip = 0.3
    prob_vflip = 0.3

    # Training Parameter
    num_epochs = 30
    lr = 5e-4
    batch_size = 2
