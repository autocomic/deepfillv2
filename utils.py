from deepfillv2 import network

# ----------------------------------------
#                 Network
# ----------------------------------------
def create_generator(opt):
    # Initialize the networks
    generator = network.GatedGenerator(opt)
    print("-- Generator is created! --")
    network.weights_init(
        generator, init_type=opt.init_type, init_gain=opt.init_gain
    )
    print("-- Initialized generator with %s type --" % opt.init_type)
    return generator
